# Multi-predicate span-based SRL based on the e2e-coref model.

import math
import numpy as np
import operator
import os
import random
import tensorflow as tf

import util
from lsgn_data import LSGNData

from embedding_helper import get_embeddings
from input_utils import *
from model_utils import *


class SRLModel(object):
  def __init__(self, lsgn_data, config):
    self.config = config
    self.data = lsgn_data 
    # TODO: Make labels_dict = None at test time.
    self.predictions, self.loss = self.get_predictions_and_loss(
        self.data.input_dict, self.data.labels_dict)

    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    self.reset_global_step = tf.assign(self.global_step, 0)
    learning_rate = tf.train.exponential_decay(
        self.config["learning_rate"], self.global_step, self.config["decay_frequency"],
        self.config["decay_rate"], staircase=True)
    trainable_params = tf.trainable_variables()
    gradients = tf.gradients(self.loss, trainable_params)
    gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
    optimizers = {
      "adam" : tf.train.AdamOptimizer,
      "sgd" : tf.train.GradientDescentOptimizer
    }
    optimizer = optimizers[self.config["optimizer"]](learning_rate)
    self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params),
                                              global_step=self.global_step)
    # For debugging.
    # for var in tf.trainable_variables():
    #  print var

  def get_predictions_and_loss(self, inputs, labels):
    # This little thing got batched.
    is_training = inputs["is_training"][0]
    self.dropout = 1 - (tf.to_float(is_training) * self.config["dropout_rate"])
    self.lexical_dropout = 1 - (tf.to_float(is_training) * self.config["lexical_dropout_rate"])
    self.lstm_dropout = 1 - (tf.to_float(is_training) * self.config["lstm_dropout_rate"])
  
    sentences = inputs["tokens"] 
    text_len = inputs["text_len"]  # [num_sentences]
    context_word_emb = inputs["context_word_emb"]  # [num_sentences, max_sentence_length, emb]
    head_word_emb = inputs["head_word_emb"]  # [num_sentences, max_sentence_length, emb]
    num_sentences = tf.shape(context_word_emb)[0]
    max_sentence_length = tf.shape(context_word_emb)[1]
    context_emb, head_emb, self.lm_weights, self.lm_scaling = get_embeddings(
        self.data, sentences, text_len, context_word_emb, head_word_emb, inputs["char_idx"],
        inputs["lm_emb"], self.lexical_dropout)  # [num_sentences, max_sentence_length, emb]
    
    context_outputs = lstm_contextualize(
        context_emb, text_len, self.config, self.lstm_dropout)  # [num_sentences, max_sentence_length, emb]

    # [num_sentences, max_num_candidates], ...
    candidate_starts, candidate_ends, candidate_mask = get_span_candidates(
        text_len, max_sentence_length, self.config["max_arg_width"])
    flat_candidate_mask = tf.reshape(candidate_mask, [-1])  # [num_sentences, max_num_candidates]
    batch_word_offset = tf.expand_dims(tf.cumsum(text_len, exclusive=True), 1)  # [num_sentences, 1]
    flat_candidate_starts = tf.boolean_mask(
        tf.reshape(candidate_starts + batch_word_offset, [-1]), flat_candidate_mask)  # [num_candidates]
    flat_candidate_ends = tf.boolean_mask(
        tf.reshape(candidate_ends + batch_word_offset, [-1]), flat_candidate_mask)  # [num_candidates]

    text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length)  # [num_sentences, max_sentence_length]
    flat_context_outputs = flatten_emb_by_sentence(context_outputs, text_len_mask)  # [num_doc_words]
    flat_head_emb = flatten_emb_by_sentence(head_emb, text_len_mask)  # [num_doc_words]
    doc_len = util.shape(flat_context_outputs, 0)

    candidate_span_emb, head_scores, span_head_emb, head_indices, head_indices_log_mask = get_span_emb(
        flat_head_emb, flat_context_outputs, flat_candidate_starts, flat_candidate_ends,
        self.config, self.dropout
    )  # [num_candidates, emb], [num_candidates, max_span_width, emb], [num_candidates, max_span_width]

    num_candidates = util.shape(candidate_span_emb, 0)
    max_num_candidates_per_sentence = util.shape(candidate_mask, 1)
    candidate_span_ids = tf.sparse_to_dense(
        sparse_indices=tf.where(tf.equal(candidate_mask, True)),
        output_shape=tf.cast(tf.stack([num_sentences, max_num_candidates_per_sentence]), tf.int64),
        sparse_values=tf.range(num_candidates, dtype=tf.int32),
        default_value=0,
        validate_indices=True)  # [num_sentences, max_num_candidates]

    spans_log_mask = tf.log(tf.to_float(candidate_mask))  # [num_sentences, max_num_candidates]
    predict_dict = {"candidate_starts": candidate_starts, "candidate_ends": candidate_ends}
    if head_scores is not None:
      predict_dict["head_scores"] = head_scores

    # Compute SRL representation.
    flat_candidate_arg_scores = get_unary_scores(
        candidate_span_emb, self.config, self.dropout, 1, "arg_scores")  # [num_candidates,]
    candidate_arg_scores = tf.gather(
        flat_candidate_arg_scores, candidate_span_ids) + spans_log_mask  # [num_sents, max_num_candidates] 

    # [num_sentences, max_num_args], ... [num_sentences,], [num_sentences, max_num_args] 
    arg_starts, arg_ends, arg_scores, num_args, top_arg_indices = get_batch_topk(
        candidate_starts, candidate_ends, candidate_arg_scores, self.config["argument_ratio"], text_len,
        max_sentence_length, sort_spans=False, enforce_non_crossing=False)

    candidate_pred_ids = tf.tile(tf.expand_dims(tf.range(max_sentence_length), 0),
                                 [num_sentences, 1])  # [num_sentences, max_sentence_length]
    candidate_pred_emb = context_outputs  # [num_sentences, max_sentence_length, emb]
    candidate_pred_scores = get_unary_scores(
        candidate_pred_emb, self.config, self.dropout, 1, "pred_scores"
    ) + tf.log(tf.to_float(text_len_mask))  # [num_sentences, max_sentence_length]

    if self.config["use_gold_predicates"]:
      predicates = inputs["gold_predicates"]
      num_preds = inputs["num_gold_predicates"]
      pred_scores = tf.zeros_like(predicates, dtype=tf.float32)
      top_pred_indices = predicates 
    else:
      # [num_sentences, max_num_preds] ... [num_sentences,]
      predicates, _, pred_scores, num_preds, top_pred_indices = get_batch_topk(
          candidate_pred_ids, candidate_pred_ids, candidate_pred_scores, self.config["predicate_ratio"],
          text_len, max_sentence_length, sort_spans=False, enforce_non_crossing=False)

    arg_span_indices = batch_gather(candidate_span_ids, top_arg_indices)  # [num_sentences, max_num_args]
    arg_emb = tf.gather(candidate_span_emb, arg_span_indices)  # [num_sentences, max_num_args, emb]
    pred_emb = batch_gather(candidate_pred_emb, top_pred_indices)  # [num_sentences, max_num_preds, emb]
    max_num_args = util.shape(arg_scores, 1)
    max_num_preds = util.shape(pred_scores, 1)

    # Compute SRL loss.
    srl_labels = get_srl_labels(
        arg_starts, arg_ends, predicates, labels, max_sentence_length
    )  # [num_sentences, max_num_args, max_num_preds]
    srl_scores = get_srl_scores(
        arg_emb, pred_emb, arg_scores, pred_scores, len(self.data.srl_labels), self.config, self.dropout
    )  # [num_sentences, max_num_args, max_num_preds, num_labels]
    srl_loss = get_srl_softmax_loss(
        srl_scores, srl_labels, num_args, num_preds)  # [num_sentences, max_num_args, max_num_preds]
    predict_dict.update({
      "candidate_arg_scores": candidate_arg_scores,
      "candidate_pred_scores": candidate_pred_scores,
      "arg_starts": arg_starts,
      "arg_ends": arg_ends,
      "predicates": predicates,
      "arg_scores": arg_scores,  # New ...
      "pred_scores": pred_scores,
      "num_args": num_args,
      "num_preds": num_preds,
      "arg_labels": tf.argmax(srl_scores, -1), # [num_sentences, num_args, num_preds]
      "srl_scores": srl_scores
    })

    tf.summary.scalar("SRL_loss", srl_loss)
    loss = srl_loss
    return predict_dict, loss


       



