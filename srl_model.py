# Multi-predicate span-based SRL based on the e2e-coref model.

import datetime
import h5py
import json
import math
import numpy as np
import operator
import os
import random
import tensorflow as tf
import threading
import time

import util
import conll
import metrics as coref_metrics

import debug_utils
import inference_utils
from input_utils import *
from model_utils import *
import srl_eval_utils


class SRLModel(object):
  def __init__(self, config):
    self.config = config
    self.context_embeddings = util.EmbeddingDictionary(config["context_embeddings"])
    self.head_embeddings = util.EmbeddingDictionary(config["head_embeddings"],
                                                    maybe_cache=self.context_embeddings)
    self.char_embedding_size = config["char_embedding_size"]
    self.char_dict = util.load_char_dict(config["char_vocab_path"])
    
    self.genres = { g:i for i,g in enumerate(config["genres"]) }
 
    if config["lm_path"]:
      self.lm_file = h5py.File(self.config["lm_path"], "r")
      self.lm_layers = self.config["lm_layers"]
      self.lm_size = self.config["lm_size"]
    else:
      self.lm_file = None
      self.lm_layers = 0
      self.lm_size = 0

    # Other configs.
    self.use_entity_model = config["refresh_srl_scores"] or config["refresh_antecedent_scores"]
    if self.use_entity_model:
      print "Using entity model."

    self.adjunct_roles, self.core_roles = split_srl_labels(config["srl_labels"])
    print self.adjunct_roles, self.core_roles
    self.srl_labels_inv  = [""] + self.adjunct_roles + self.core_roles
    print len(self.srl_labels_inv), len(self.adjunct_roles), len(self.core_roles)
    self.srl_labels = { l:i for i,l in enumerate(self.srl_labels_inv) }

    self.const_labels = { l:i for i,l in enumerate([""] + config["const_labels"]) }
    self.const_labels_inv = [""] + config["const_labels"]
    self.ner_labels = { l:i for i,l in enumerate([""] + config["ner_labels"]) }
    self.ner_labels_inv = [""] + config["ner_labels"]

    self.eval_data = None  # Load eval data lazily.

    # Need to make sure they are in the same order as input_names + label_names
    self.input_props = [
      (tf.float32, [None, self.context_embeddings.size]), # Context embeddings.
      (tf.float32, [None, self.head_embeddings.size]), # Head embeddings.
      (tf.float32, [None, self.lm_size, self.lm_layers]), # LM embeddings.
      (tf.int32, [None, None]), # Character indices.
      (tf.int32, []),  # Text length.
      (tf.int32, [None]),  # Speaker IDs.
      (tf.int32, []),  # Genre.
      # (tf.int32, []),  # Word offset.
      (tf.int32, []),  # Document ID.
      (tf.bool, []),  # Is training.
      (tf.int32, [None]),  # Gold predicate ids (for input).
      (tf.int32, []),  # Num gold predicates (for input).
      (tf.int32, [None]),  # Predicate ids (length=num_srl_relations).
      (tf.int32, [None]),  # Argument starts.
      (tf.int32, [None]),  # Argument ends.
      (tf.int32, [None]),  # SRL labels.
      (tf.int32, []),  # Number of SRL relations.
      (tf.int32, [None]),  # Constituent starts.
      (tf.int32, [None]),  # Constituent ends.
      (tf.int32, [None]),  # Constituent labels.
      (tf.int32, []),  # Number of  constituent spans.
      (tf.int32, [None]),  # NER starts.
      (tf.int32, [None]),  # NER ends.
      (tf.int32, [None]),  # NER labels.
      (tf.int32, []),  # Number of NER spans.
      (tf.int32, [None]),  # Coref mention starts.
      (tf.int32, [None]),  # Coref mention ends.
      (tf.int32, [None]),  # Coref cluster ids.
      (tf.int32, []),  # Number of coref mentions.
    ]

    # Names for the "given" tensors.
    self.input_names = [
        "context_word_emb", "head_word_emb", "lm_emb", "char_idx", "text_len", #"word_offset",
        "speaker_ids", "genre", "doc_id", "is_training",
        "gold_predicates", "num_gold_predicates",
    ]
    # Names for the "gold" tensors.
    self.label_names = [
        "predicates", "arg_starts", "arg_ends", "arg_labels", "srl_len",
        "const_starts", "const_ends", "const_labels", "const_len",
        "ner_starts", "ner_ends", "ner_labels", "ner_len",
        "coref_starts", "coref_ends", "coref_cluster_ids", "coref_len",
    ]
    # Name for predicted tensors.
    self.predict_names = [
        "candidate_starts", "candidate_ends", "candidate_arg_scores", "candidate_pred_scores",
        "arg_starts", "arg_ends", "predicates", "num_args", "num_preds", "arg_labels", "srl_scores", "ner_scores",
        "const_scores", "arg_scores", "pred_scores",
        "candidate_mention_starts", "candidate_mention_ends", "candidate_mention_scores", "mention_starts",
        "mention_ends", "antecedents", "antecedent_scores",
        "srl_head_scores", "coref_head_scores", "ner_head_scores", "entity_gate", "antecedent_attn"
    ]

    self.batch_size = self.config["batch_size"]
    dtypes, shapes = zip(*self.input_props)
    if self.batch_size > 0 and self.config["max_tokens_per_batch"] < 0:
      # Use fixed batch size if number of words per batch is not limited (-1).
      self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in self.input_props]
      queue = tf.PaddingFIFOQueue(capacity=self.batch_size * 2, dtypes=dtypes, shapes=shapes)
      self.enqueue_op = queue.enqueue(self.queue_input_tensors)
      self.input_tensors = queue.dequeue_many(self.batch_size)
    else:
      # Use dynamic batch size.
      new_shapes = [[None] + shape for shape in shapes]
      self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in zip(dtypes, new_shapes)]
      queue = tf.PaddingFIFOQueue(capacity=2, dtypes=dtypes, shapes=new_shapes)
      self.enqueue_op = queue.enqueue(self.queue_input_tensors)
      self.input_tensors = queue.dequeue()

    num_features = len(self.input_names)
    input_dict = dict(zip(self.input_names, self.input_tensors[:num_features]))
    labels_dict = dict(zip(self.label_names, self.input_tensors[num_features:]))

    # TODO: Make labels_dict = None at test time.
    self.predictions, self.loss = self.get_predictions_and_loss(input_dict, labels_dict)
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
    self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)
    # For debugging.
    # for var in tf.trainable_variables():
    #  print var

  def start_enqueue_thread(self, session):
    with open(self.config["train_path"], "r") as f:
      train_examples = [json.loads(jsonline) for jsonline in f.readlines()]

    populate_sentence_offset(train_examples)
    def _enqueue_loop():
      adaptive_batching = (self.config["max_tokens_per_batch"] > 0)
      while True:
        random.shuffle(train_examples)
        doc_examples = []  # List of list of examples.
        cluster_id_offset = 0
        num_sentences = 0
        num_mentions = 0
        for doc_id, example in enumerate(train_examples):
          doc_examples.append([])
          for e in self.split_document_example(example):
            e["doc_id"] = doc_id + 1
            e["cluster_id_offset"] = cluster_id_offset
            doc_examples[-1].append(e)
            num_mentions += len(e["coref"]) 
          cluster_id_offset += len(example["clusters"])
          num_sentences += len(doc_examples[-1])
        print ("Load {} training documents with {} sentences and a total of {} clusters and {} mentions.".format(
            doc_id, num_sentences, cluster_id_offset, num_mentions))

        tensor_names = self.input_names + self.label_names
        batch_buffer = []
        num_tokens_in_batch = 0
        for examples in doc_examples:
          tensor_examples = [self.tensorize_example(e, is_training=True) for e in examples]
          if self.config["batch_size"] == -1:
            # Random truncation.
            num_sents = len(tensor_examples)
            max_training_sents = self.config["max_training_sentences"]
            if num_sents > max_training_sents:
              sentence_offset = random.randint(0, num_sents - max_training_sents)
              tensor_examples = tensor_examples[sentence_offset:sentence_offset + max_training_sents]
            batched_tensor_examples = [pad_batch_tensors(tensor_examples, tn) for tn in tensor_names]
            feed_dict = dict(zip(self.queue_input_tensors, batched_tensor_examples))
            session.run(self.enqueue_op, feed_dict=feed_dict)
          elif adaptive_batching:
            for tensor_example in tensor_examples:
              num_tokens = tensor_example["text_len"]
              if len(batch_buffer) >= self.config["batch_size"] or (
                  num_tokens_in_batch + num_tokens > self.config["max_tokens_per_batch"]):
                batched_tensor_examples = [pad_batch_tensors(batch_buffer, tn) for tn in tensor_names]
                feed_dict = dict(zip(self.queue_input_tensors, batched_tensor_examples))
                session.run(self.enqueue_op, feed_dict=feed_dict)
                batch_buffer = []
                num_tokens_in_batch = 0
              batch_buffer.append(tensor_example)
              num_tokens_in_batch += num_tokens
          else:
            for tensor_example in tensor_examples:
              feed_dict = dict(zip(self.queue_input_tensors, [tensor_example[tn] for tn in tensor_names]))
              session.run(self.enqueue_op, feed_dict=feed_dict)
        # Clear out the batch buffer after each epoch, to avoid the potential danger where the first document
        # in the next batch is the same one as the last document in the previous batch.
        if len(batch_buffer) > 0:
          batched_tensor_examples = [pad_batch_tensors(batch_buffer, tn) for tn in tensor_names]
          feed_dict = dict(zip(self.queue_input_tensors, batched_tensor_examples))
          session.run(self.enqueue_op, feed_dict=feed_dict)
            
    enqueue_thread = threading.Thread(target=_enqueue_loop)
    enqueue_thread.daemon = True
    enqueue_thread.start()

  def split_document_example(self, example):
    """ Split document-based samples into sentence-based samples.
    """
    clusters = example["clusters"]
    gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
    cluster_ids = {}
    for cluster_id, cluster in enumerate(clusters):
      for mention in cluster:
        cluster_ids[tuple(mention)] = cluster_id + 1

    sentences = example["sentences"]
    split_examples = []
    word_offset = 0

    if "speakers" in example:
      speakers = example["speakers"]
      flat_speakers = util.flatten(example["speakers"])
      speaker_dict = { s:i for i,s in enumerate(set(flat_speakers)) }
      speaker_ids = []
      for sent_speakers in speakers:
        speaker_ids.append([speaker_dict[s] for s in sent_speakers])
    else:
      speaker_ids = [[0] for _ in sentences]
  
    if "genre" in example:
      genre_id = self.genres[example["doc_key"][:2]]
    else:
      genre_id = 0

    for i, sentence in enumerate(sentences):
      text_len = len(sentence)
      coref_mentions = []
      for start, end in gold_mentions:
        if word_offset <= start and end < word_offset + text_len:
          coref_mentions.append([start, end, cluster_ids[(start, end)]])

      sent_example = {
        "sentence": sentence,
        "doc_key": example["doc_key"],
        "speaker_ids": speaker_ids[i],
        "genre": genre_id,
        "sent_id": i,
        "constituents": example["constituents"][i],
        "ner": example["ner"][i],
        "srl": example["srl"][i],
        "coref": coref_mentions,
        "word_offset": word_offset, 
        "sent_offset": example["sent_offset"]  # Sentence offset for the same doc ID.
      }
      word_offset += text_len
      split_examples.append(sent_example)
    return split_examples

  def tensorize_example(self, example, is_training):
    """ Tensorize examples and caching embeddings.
    """
    sentence = example["sentence"]
    doc_key = example["doc_key"]
    sent_id = example["sent_id"]  # Number of sentence in the document.
    word_offset = example["word_offset"]
    text_len = len(sentence)

    lm_doc_key = None
    lm_sent_key = None
    if self.lm_file and "ontonotes" in self.config["lm_path"]:
      idx = doc_key.rfind("_")
      lm_doc_key = doc_key[:idx] + "/" + str(example["sent_offset"] + sent_id)
    elif self.lm_file and "conll05" in self.config["lm_path"]:
      lm_doc_key = doc_key[1:]  # "S1234" -> "1234"
    else:
      lm_doc_key = doc_key
      lm_sent_key = str(sent_id)
    lm_emb = load_lm_embeddings_for_sentence(self.lm_file, self.lm_layers, self.lm_size, lm_doc_key, lm_sent_key)
    max_word_length = max(max(len(w) for w in sentence), max(self.config["filter_widths"]))
    context_word_emb = np.zeros([text_len, self.context_embeddings.size])
    head_word_emb = np.zeros([text_len, self.head_embeddings.size])
    char_index = np.zeros([text_len, max_word_length])
    for j, word in enumerate(sentence):
      context_word_emb[j] = self.context_embeddings[word]
      head_word_emb[j] = self.head_embeddings[word]
      char_index[j, :len(word)] = [self.char_dict[c] for c in word]

    const_starts, const_ends, const_labels = (
        tensorize_labeled_spans(example["constituents"], self.const_labels))
    ner_starts, ner_ends, ner_labels = (
        tensorize_labeled_spans(example["ner"], self.ner_labels))
    coref_starts, coref_ends, coref_cluster_ids = (
        tensorize_labeled_spans(example["coref"], label_dict=None))
    predicates, arg_starts, arg_ends, arg_labels = (
        tensorize_srl_relations(example["srl"], self.srl_labels, filter_v_args=self.config["filter_v_args"]))
    # For gold predicate experiment.
    #gold_predicates = np.unique(predicates - word_offset)
    gold_predicates = get_all_predicates(example["srl"]) - word_offset
    example_tensor = {
      # Inputs.
      "context_word_emb": context_word_emb,
      "head_word_emb": head_word_emb,
      "lm_emb": lm_emb,
      "char_idx": char_index,
      "text_len": text_len,
      "speaker_ids": np.array(example["speaker_ids"]),
      "genre": example["genre"],
      "doc_id": example["doc_id"],
      "is_training": is_training,
      "gold_predicates": gold_predicates,
      "num_gold_predicates": len(gold_predicates),
      # Labels.
      "const_starts": const_starts - word_offset, 
      "const_ends": const_ends - word_offset, 
      "const_labels": const_labels,
      "ner_starts": ner_starts - word_offset,
      "ner_ends": ner_ends - word_offset,
      "ner_labels": ner_labels,
      "predicates": predicates - word_offset,
      "arg_starts": arg_starts - word_offset,
      "arg_ends": arg_ends - word_offset,
      "arg_labels": arg_labels,
      "coref_starts": coref_starts - word_offset,
      "coref_ends": coref_ends - word_offset,
      "coref_cluster_ids": coref_cluster_ids + example["cluster_id_offset"],
      "srl_len": len(predicates),
      "const_len": len(const_starts),
      "ner_len": len(ner_starts),
      "coref_len": len(coref_starts)
    }
    return example_tensor

  def get_embeddings(self, context_word_emb, head_word_emb, char_index, lm_emb):
    context_emb_list = [context_word_emb]
    head_emb_list = [head_word_emb]

    num_sentences = tf.shape(context_word_emb)[0]
    max_sentence_length = tf.shape(context_word_emb)[1]

    if self.config["char_embedding_size"] > 0:
      char_emb = tf.gather(
          tf.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]]),
          char_index)  # [num_sentences, max_sentence_length, max_word_length, emb]
      flattened_char_emb = tf.reshape(
          char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2),
          util.shape(char_emb, 3)])  # [num_sentences * max_sentence_length, max_word_length, emb]
      flattened_aggregated_char_emb = util.cnn(
          flattened_char_emb, self.config["filter_widths"],
          self.config["filter_size"])  # [num_sentences * max_sentence_length, emb]
      aggregated_char_emb = tf.reshape(
          flattened_aggregated_char_emb, [num_sentences, max_sentence_length,
          util.shape(flattened_aggregated_char_emb, 1)]) # [num_sentences, max_sentence_length, emb]
      context_emb_list.append(aggregated_char_emb)
      head_emb_list.append(aggregated_char_emb)

    if self.lm_file:
      lm_emb_size = util.shape(lm_emb, 2)
      lm_num_layers = util.shape(lm_emb, 3)
      with tf.variable_scope("lm_aggregation"):
        self.lm_weights = tf.nn.softmax(tf.get_variable("lm_scores", [self.lm_layers],
                                        initializer=tf.constant_initializer(0.0)))
        self.lm_scaling = tf.get_variable("lm_scaling", [], initializer=tf.constant_initializer(1.0))
      flattened_lm_emb = tf.reshape(
          lm_emb, [num_sentences * max_sentence_length * lm_emb_size, lm_num_layers]
      )  # [num_sentences * max_sentence_length * emb, layers]
      flattened_aggregated_lm_emb = tf.matmul(
          flattened_lm_emb, tf.expand_dims(self.lm_weights, 1)) # [num_sentences * max_sentence_length * emb, 1]
      aggregated_lm_emb = tf.reshape(
          flattened_aggregated_lm_emb, [num_sentences, max_sentence_length, lm_emb_size])
      aggregated_lm_emb *= self.lm_scaling
      context_emb_list.append(aggregated_lm_emb)

    # Concatenate and apply dropout.
    context_emb = tf.concat(context_emb_list, 2)  # [num_sentences, max_sentence_length, emb]
    head_emb = tf.concat(head_emb_list, 2)  # [num_sentences, max_sentence_length, emb]
    context_emb = tf.nn.dropout(context_emb, self.lexical_dropout)
    head_emb = tf.nn.dropout(head_emb, self.lexical_dropout)
    return context_emb, head_emb
    
  def get_predictions_and_loss(self, inputs, labels):
    # This little thing got batched.
    is_training = inputs["is_training"][0]
    self.dropout = 1 - (tf.to_float(is_training) * self.config["dropout_rate"])
    self.lexical_dropout = 1 - (tf.to_float(is_training) * self.config["lexical_dropout_rate"])
    self.lstm_dropout = 1 - (tf.to_float(is_training) * self.config["lstm_dropout_rate"])
   
    context_word_emb = inputs["context_word_emb"]  # [num_sentences, max_sentence_length, emb]
    head_word_emb = inputs["head_word_emb"]  # [num_sentences, max_sentence_length, emb]
    num_sentences = tf.shape(context_word_emb)[0]
    max_sentence_length = tf.shape(context_word_emb)[1]

    context_emb, head_emb = self.get_embeddings(
        context_word_emb, head_word_emb, inputs["char_idx"],
        inputs["lm_emb"])  # [num_sentences, max_sentence_length, emb]
    
    text_len = inputs["text_len"]  # [num_sentences]
    text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length)  # [num_sentences, max_sentence_length]

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

    if self.config["span_score_weight"] > 0:
      flat_span_scores = get_unary_scores(
          candidate_span_emb, self.config, self.dropout, 1, "span_scores")  # [num_candidates,]

    spans_log_mask = tf.log(tf.to_float(candidate_mask))  # [num_sentences, max_num_candidates]
    predict_dict = {"candidate_starts": candidate_starts, "candidate_ends": candidate_ends}

    if head_scores is not None:
      predict_dict["head_scores"] = head_scores

    # Compute SRL representation.
    if self.config["srl_weight"] > 0:
      flat_candidate_arg_scores = get_unary_scores(
          candidate_span_emb, self.config, self.dropout, 1, "arg_scores")  # [num_candidates,]
      if self.config["span_score_weight"] > 0:
        flat_candidate_arg_scores += self.config["span_score_weight"] * flat_span_scores
      candidate_arg_scores = tf.gather(
          flat_candidate_arg_scores, candidate_span_ids) + spans_log_mask  # [num_sentences, max_num_candidates] 

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
        #pred_scores = batch_gather(candidate_pred_scores, predicates)
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

    # Get coref representations.
    if self.config["coref_weight"] > 0:
      candidate_mention_scores = get_unary_scores(
          candidate_span_emb, self.config, self.dropout, 1, "mention_scores")  # [num_candidates]
      if self.config["span_score_weight"] > 0:
        candidate_mention_scores += self.config["span_score_weight"] * flat_span_scores

      doc_ids = tf.expand_dims(inputs["doc_id"], 1)  # [num_sentences, 1]
      candidate_doc_ids = tf.boolean_mask(
          tf.reshape(tf.tile(doc_ids, [1, max_num_candidates_per_sentence]), [-1]),
          flat_candidate_mask)  # [num_candidates]
 
      k = tf.to_int32(tf.floor(tf.to_float(doc_len) * self.config["mention_ratio"]))
      top_mention_indices = srl_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
                                                  tf.expand_dims(flat_candidate_starts, 0),
                                                  tf.expand_dims(flat_candidate_ends, 0),
                                                  tf.expand_dims(k, 0), doc_len,
                                                  True, True)  # [1, k]
      top_mention_indices.set_shape([1, None])
      top_mention_indices = tf.squeeze(top_mention_indices, 0)  # [k]
      mention_starts = tf.gather(flat_candidate_starts, top_mention_indices)  # [k]
      mention_ends = tf.gather(flat_candidate_ends, top_mention_indices)  #[k]
      mention_scores = tf.gather(candidate_mention_scores, top_mention_indices)  #[k]
      mention_emb = tf.gather(candidate_span_emb, top_mention_indices)  # [k, emb]
      mention_doc_ids = tf.gather(candidate_doc_ids, top_mention_indices)  # [k]

      if head_scores is not None:
        predict_dict["coref_head_scores"] = head_scores

      # FIXME: We really shouldn't use unsorted. There must be a bug in sorting.
      max_mentions_per_doc = tf.reduce_max(
          #tf.segment_sum(data=tf.ones_like(mention_doc_ids, dtype=tf.int32),
          tf.unsorted_segment_sum(data=tf.ones_like(mention_doc_ids, dtype=tf.int32),
          segment_ids=mention_doc_ids,
          num_segments=tf.reduce_max(mention_doc_ids) + 1))  # []

      k_Print = tf.Print(k,
          [num_sentences, doc_len, k, max_mentions_per_doc],
          "Num sents, num tokens, num_mentions, max_antecedents")

      max_antecedents = tf.minimum(
          tf.minimum(self.config["max_antecedents"], k - 1), max_mentions_per_doc - 1)

      target_indices = tf.expand_dims(tf.range(k), 1)  # [k, 1]
      antecedent_offsets = tf.expand_dims(tf.range(max_antecedents) + 1, 0)  # [1, max_ant]
      raw_antecedents = target_indices - antecedent_offsets  # [k, max_ant]
      antecedents = tf.maximum(raw_antecedents, 0)  # [k, max_ant]

      target_doc_ids = tf.expand_dims(mention_doc_ids, 1)  # [k, 1]
      antecedent_doc_ids = tf.gather(mention_doc_ids, antecedents)  # [k, max_ant]
      antecedent_mask = tf.logical_and(tf.equal(target_doc_ids, antecedent_doc_ids),
                                                tf.greater_equal(raw_antecedents, 0))  # [k, max_ant]
      antecedent_log_mask = tf.log(tf.to_float(antecedent_mask))  # [k, max_ant]

      if self.config["use_metadata"]:
        print ("Using metadata")
        genre_emb = tf.gather(tf.get_variable(
            "genre_embeddings", [len(self.genres), self.config["feature_size"]]), inputs["genre"][0]) # [emb]
        flat_speaker_ids = flatten_emb_by_sentence(inputs["speaker_ids"], text_len_mask)  # [text_len]
        mention_speaker_ids = tf.gather(flat_speaker_ids, mention_starts)  # [k]
      else:
        genre_emb = None
        mention_speaker_ids = None

      # [k, max_ant], [k, max_ant, emb], [k, max_ant, emb2]
      antecedent_scores, antecedent_emb, pair_emb = get_antecedent_scores(
          mention_emb, mention_scores, antecedents, mention_speaker_ids, genre_emb, self.config, self.dropout
      )  # [k, max_ant]
      antecedent_scores = tf.concat([
          tf.zeros([k, 1]), antecedent_scores + antecedent_log_mask], 1)  # [k, max_ant+1]

    # Get labels.    
    if self.config["ner_weight"] + self.config["const_weight"] + self.config["coref_weight"] > 0:
      gold_ner_labels, gold_const_labels, gold_coref_cluster_ids = get_span_task_labels(
          candidate_starts, candidate_ends, labels, max_sentence_length)  # [num_sentences, max_num_candidates]

    # Compute SRL loss.
    if self.config["srl_weight"] > 0:
      srl_labels = get_srl_labels(
          arg_starts, arg_ends, predicates, labels, max_sentence_length
      )  # [num_sentences, max_num_args, max_num_preds]
      srl_scores = get_srl_scores(
          arg_emb, pred_emb, arg_scores, pred_scores, len(self.srl_labels), self.config, self.dropout
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
    else:
      srl_loss = 0

    # Compute Coref loss.
    if self.config["coref_weight"] > 0:
      flat_cluster_ids = tf.boolean_mask(
          tf.reshape(gold_coref_cluster_ids, [-1]), flat_candidate_mask)  # [num_candidates]
      mention_cluster_ids = tf.gather(flat_cluster_ids, top_mention_indices)  # [k]

      antecedent_cluster_ids = tf.gather(mention_cluster_ids, antecedents)  # [k, max_ant]
      antecedent_cluster_ids += tf.to_int32(antecedent_log_mask)  # [k, max_ant]
      same_cluster_indicator = tf.equal(
          antecedent_cluster_ids, tf.expand_dims(mention_cluster_ids, 1))  # [k, max_ant]
      non_dummy_indicator = tf.expand_dims(mention_cluster_ids > 0, 1)  # [k, 1]
      pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator)  # [k, max_ant]

      dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keep_dims=True))  # [k, 1]
      antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1)  # [k, max_ant+1]
      coref_loss = get_coref_softmax_loss(antecedent_scores, antecedent_labels)  # [k]
      coref_loss = tf.reduce_sum(coref_loss) # / tf.to_float(num_sentences)  # []
      predict_dict.update({
          "candidate_mention_starts": flat_candidate_starts,  # [num_candidates]
          "candidate_mention_ends": flat_candidate_ends,  # [num_candidates]
          "candidate_mention_scores": candidate_mention_scores,  # [num_candidates]
          "mention_starts": mention_starts,  # [k]
          "mention_ends": mention_ends,  # [k]
          "antecedents": antecedents,  # [k, max_ant]
          "antecedent_scores": antecedent_scores,  # [k, max_ant+1]
      })
    else:
      coref_loss = 0

    # TODO: Move to other places maybe.
    dummy_scores = tf.expand_dims(tf.zeros_like(candidate_span_ids, dtype=tf.float32), 2)
    if self.config["ner_weight"] > 0:
      ner_span_emb = candidate_span_emb
      flat_ner_scores = get_unary_scores(
          ner_span_emb, self.config, self.dropout, len(self.ner_labels) - 1,
          "ner_scores")  # [num_candidates, num_labels-1]
      if self.config["span_score_weight"] > 0:
        flat_ner_scores += self.config["span_score_weight"] * tf.expand_dims(flat_span_scores, 1)
      ner_scores = tf.gather(
          flat_ner_scores, candidate_span_ids
      ) + tf.expand_dims(spans_log_mask, 2)  # [num_sentences, max_num_candidates, num_labels-1]
      ner_scores = tf.concat([dummy_scores, ner_scores], 2)  # [num_sentences, max_num_candidates, num_labels]

      ner_loss = get_softmax_loss(ner_scores, gold_ner_labels, candidate_mask)  # [num_sentences]
      ner_loss = tf.reduce_sum(ner_loss) # / tf.to_float(num_sentences)  # []
      predict_dict["ner_scores"] = ner_scores
    else:
      ner_loss = 0
    
    if self.config["const_weight"] > 0: 
      flat_const_scores = get_unary_scores(
          candidate_span_emb, self.config, self.dropout, len(self.const_labels) - 1,
          "const_scores")  # [num_sentences, max_num_candidates, num_labels-1]
      if self.config["span_score_weight"] > 0:
        flat_const_scores += self.config["span_score_weight"] * tf.expand_dims(flat_span_scores, 1)

      const_scores = tf.gather(
        flat_const_scores, candidate_span_ids)  # [num_sentences, max_num_candidates, num_labels-1]
      const_scores = tf.concat([dummy_scores, const_scores], 2)  # [num_sentences, max_num_candidates, num_labels]

      # Do not prune constituency candidates. Because there are many of them ...
      const_loss = get_softmax_loss(const_scores, gold_const_labels, candidate_mask)  # [num_sentences]
      const_loss = tf.reduce_sum(const_loss) # / tf.to_float(num_sentences)  # []
      predict_dict["const_scores"] = const_scores
    else:
      const_loss = 0

    tf.summary.scalar("SRL_loss", srl_loss)
    tf.summary.scalar("NER_loss", ner_loss)
    tf.summary.scalar("Constituency_loss", const_loss)
    tf.summary.scalar("Coref_loss", coref_loss)
    #srl_loss_Print = tf.Print(srl_loss, [srl_loss, ner_loss, coref_loss], "Loss")
    loss = self.config["srl_weight"] * srl_loss + self.config["ner_weight"] * ner_loss + (
        self.config["const_weight"] * const_loss + self.config["coref_weight"] * coref_loss)

    return predict_dict, loss

  def load_eval_data(self):
    if self.eval_data is None:
      self.eval_data = []
      self.eval_tensors = []
      self.coref_eval_data = []
      with open(self.config["eval_path"]) as f:
        eval_examples = [json.loads(jsonline) for jsonline in f.readlines()]
      populate_sentence_offset(eval_examples)
      for doc_id, example in enumerate(eval_examples):
        doc_tensors = []
        num_mentions_in_doc = 0
        for e in self.split_document_example(example):
          # Because each batch=1 document at test time, we do not need to offset cluster ids.
          e["cluster_id_offset"] = 0
          e["doc_id"] = doc_id + 1
          doc_tensors.append(self.tensorize_example(e, is_training=False))
          num_mentions_in_doc += len(e["coref"])
        assert num_mentions_in_doc == len(util.flatten(example["clusters"]))
        self.eval_tensors.append(doc_tensors)
        self.eval_data.extend(srl_eval_utils.split_example_for_eval(example))
        self.coref_eval_data.append(example)
    print("Loaded {} eval examples.".format(len(self.eval_data)))

  def evaluate(self, session, official_stdout=False):
    self.load_eval_data()

    def _k_to_tag(k):
      if k == -3:
        return "oracle"
      elif k == -2:
        return "actual"
      elif k == -1:
        return "exact"
      elif k == 0:
        return "threshold"
      else:
        return "{}%".format(k)

    # Retrieval evaluators.
    arg_evaluators = { k:util.RetrievalEvaluator() for k in [-3, -2, -1, 30, 40, 50, 80, 100, 120, 150] }
    predicate_evaluators = { k:util.RetrievalEvaluator() for k in [-3, -2, -1, 10, 20, 30, 40, 50, 70] }
    mention_evaluators = { k:util.RetrievalEvaluator() for k in [-3, -2, -1, 10, 20, 30, 40, 50] }

    total_loss = 0
    total_num_predicates = 0
    total_gold_predicates = 0
    
    srl_comp_sents = 0

    srl_predictions = []
    ner_predictions = []
    coref_predictions = {}
    coref_evaluator = coref_metrics.CorefEvaluator()

    start_time = time.time()
    debug_printer = debug_utils.DebugPrinter()
    sent_id = 0

    # Simple analysis.
    unique_core_role_violations = 0
    continuation_role_violations = 0
    reference_role_violations = 0
    gold_u_violations = 0
    gold_c_violations = 0
    gold_r_violations = 0

    for i, doc_tensors in enumerate(self.eval_tensors):
      feed_dict = dict(zip(
          self.input_tensors,
          [pad_batch_tensors(doc_tensors, tn) for tn in self.input_names + self.label_names]))

      predict_names = []
      for tn in self.predict_names:
         if tn in self.predictions:
          predict_names.append(tn)

      predict_tensors = [self.predictions[tn] for tn in predict_names] + [self.loss]
      predict_tensors = session.run(predict_tensors, feed_dict=feed_dict)
      predict_dict = dict(zip(predict_names + ["loss"], predict_tensors))

      doc_size = len(doc_tensors)
      doc_example = self.coref_eval_data[i]
      sentences = doc_example["sentences"]
      predictions = inference_utils.mtl_decode(sentences, predict_dict, self.srl_labels_inv, self.ner_labels_inv,
                                               self.config)
      if "srl" in predictions:
        srl_predictions.extend(predictions["srl"])
        # Evaluate retrieval.
        word_offset = 0
        for j in range(len(sentences)):
          text_length = len(sentences[j])
          na = predict_dict["num_args"][j]
          np = predict_dict["num_preds"][j]
          sent_example = self.eval_data[sent_id]  # sentence, srl, ner
          gold_args = set([])
          gold_preds = set([])
          for pred, args in sent_example[1].iteritems():
            gold_preds.add((pred, pred))
            gold_args.update([(a[0], a[1]) for a in args if a[2] not in ["V", "C-V"]])
          srl_eval_utils.evaluate_retrieval(
                predict_dict["candidate_starts"][j], predict_dict["candidate_ends"][j],
                predict_dict["candidate_arg_scores"][j], predict_dict["arg_starts"][j][:na], predict_dict["arg_ends"][j][:na],
                gold_args, text_length, arg_evaluators)
          srl_eval_utils.evaluate_retrieval(
                range(text_length), range(text_length), predict_dict["candidate_pred_scores"][j],
                predict_dict["predicates"][j][:np], predict_dict["predicates"][j][:np], gold_preds, text_length,
                predicate_evaluators)
       
          # TODO: Move elsewhere.
          u_violations, c_violations, r_violations = debug_utils.srl_constraint_tracker(predictions["srl"][j])
          unique_core_role_violations += u_violations
          continuation_role_violations += c_violations
          reference_role_violations += r_violations
          total_num_predicates += len(predictions["srl"][j].keys())
          u_violations, c_violations, r_violations = debug_utils.srl_constraint_tracker(sent_example[1])
          gold_u_violations += u_violations
          gold_c_violations += c_violations
          gold_r_violations += r_violations
          total_gold_predicates += len(sent_example[1].keys())
          sent_id += 1
          word_offset += text_length

      if "ner" in predictions:
        ner_predictions.extend(predictions["ner"])

      if "predicted_clusters" in predictions:
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in doc_example["clusters"]]
        gold_mentions = set([])
        mention_to_gold = {}
        for gc in gold_clusters:
          for mention in gc:
            mention_to_gold[mention] = gc
            gold_mentions.add(mention)
        coref_evaluator.update(predictions["predicted_clusters"], gold_clusters, predictions["mention_to_predicted"],
                               mention_to_gold)
        coref_predictions[doc_example["doc_key"]] = predictions["predicted_clusters"]
        # Evaluate retrieval.
        doc_text_length = sum([len(s) for s in sentences])
        srl_eval_utils.evaluate_retrieval(
            predict_dict["candidate_mention_starts"], predict_dict["candidate_mention_ends"],
            predict_dict["candidate_mention_scores"], predict_dict["mention_starts"], predict_dict["mention_ends"],
            gold_mentions, doc_text_length, mention_evaluators)

      total_loss += predict_dict["loss"]
      if (i + 1) % 50 == 0:
        print ("Evaluated {}/{} documents.".format(i + 1, len(self.coref_eval_data)))

    debug_printer.close()
    summary_dict = {}
    task_to_f1 = {}  # From task name to F1.
    elapsed_time = time.time() - start_time

    sentences, gold_srl, gold_ner = zip(*self.eval_data)

    # Summarize results.
    if self.config["srl_weight"] > 0:
      precision, recall, f1, conll_precision, conll_recall, conll_f1, ul_prec, ul_recall, ul_f1, srl_label_mat, comp = (
          srl_eval_utils.compute_srl_f1(sentences, gold_srl, srl_predictions, self.config["srl_conll_eval_path"]))
      task_to_f1["srl"] = conll_f1
      summary_dict["PAS F1"] = f1
      summary_dict["PAS precision"] = precision
      summary_dict["PAS recall"] = recall
      summary_dict["Unlabeled PAS F1"] = ul_f1
      summary_dict["Unlabeled PAS precision"] = ul_prec
      summary_dict["Unlabeled PAS recall"] = ul_recall
      summary_dict["CoNLL F1"] = conll_f1
      summary_dict["CoNLL precision"] = conll_precision
      summary_dict["CoNLL recall"] = conll_recall
      if total_num_predicates > 0:
        summary_dict["Unique core violations/Predicate"] = 1.0 * unique_core_role_violations / total_num_predicates
        summary_dict["Continuation violations/Predicate"] = 1.0 * continuation_role_violations / total_num_predicates
        summary_dict["Reference violations/Predicate"] = 1.0 * reference_role_violations / total_num_predicates

      print "Completely correct sentences: {}/{}".format(comp, 100.0 * comp / len(srl_predictions))

      for k, evaluator in sorted(arg_evaluators.items(), key=operator.itemgetter(0)):
        tags = ["{} {} @ {}".format("Args", t, _k_to_tag(k)) for t in ("R", "P", "F")]
        results_to_print = []
        for t, v in zip(tags, evaluator.metrics()):
          results_to_print.append("{:<10}: {:.4f}".format(t, v))
          summary_dict[t] = v
        print ", ".join(results_to_print)

      for k, evaluator in sorted(predicate_evaluators.items(), key=operator.itemgetter(0)):
        tags = ["{} {} @ {}".format("Predicates", t, _k_to_tag(k)) for t in ("R", "P", "F")]
        results_to_print = []
        for t, v in zip(tags, evaluator.metrics()):
          results_to_print.append("{:<10}: {:.4f}".format(t, v))
          summary_dict[t] = v
        print ", ".join(results_to_print)

      if total_num_predicates > 0:
        print ("Constraint voilations: U: {} ({}), C: {} ({}), R: {} ({})".format(
            1.0 * unique_core_role_violations / total_num_predicates, unique_core_role_violations,
            1.0 * continuation_role_violations / total_num_predicates, continuation_role_violations,
            1.0 * reference_role_violations / total_num_predicates, reference_role_violations))
      if total_gold_predicates > 0:
        print ("Gold constraint voilations: U: {} ({}), C: {} ({}), R: {} ({})".format(
            1.0 * gold_u_violations / total_gold_predicates, gold_u_violations,
            1.0 * gold_c_violations / total_gold_predicates, gold_c_violations,
            1.0 * gold_r_violations / total_gold_predicates, gold_r_violations))
      #for label_pair, freq in srl_label_mat.most_common():
      #  if label_pair[0] != label_pair[1] and freq > 10:
      #    print ("{}\t{}\t{}".format(label_pair[0], label_pair[1], freq))

    if self.config["ner_weight"] > 0:
      ner_precision, ner_recall, ner_f1, ul_ner_prec, ul_ner_recall, ul_ner_f1, ner_label_mat = (
          srl_eval_utils.compute_span_f1(gold_ner, ner_predictions, "NER"))
      summary_dict["NER F1"] = ner_f1
      summary_dict["NER precision"] = ner_precision
      summary_dict["NER recall"] = ner_recall
      summary_dict["Unlabeled NER F1"] = ul_ner_f1
      summary_dict["Unlabeled NER precision"] = ul_ner_prec
      summary_dict["Unlabeled NER recall"] = ul_ner_recall

      # Write NER prediction to IOB format and run official eval script.
      srl_eval_utils.print_to_iob2(sentences, gold_ner, ner_predictions, self.config["ner_conll_eval_path"])
      task_to_f1["ner"] = ner_f1
      #for label_pair, freq in ner_label_mat.most_common():
      #  if label_pair[0] != label_pair[1] and freq > 10:
      #    print ("{}\t{}\t{}".format(label_pair[0], label_pair[1], freq))

    if self.config["const_weight"] > 0:
      const_precision, const_recall, const_f1, ul_const_prec, ul_const_recall, ul_const_f1, const_label_mat = (
          srl_eval_utils.compute_span_f1(constituency_gold, constituency_predictions, "Constituency"))
      summary_dict["Constituency F1"] = const_f1
      summary_dict["Constituency precision"] = const_precision
      summary_dict["Constituency recall"] = const_recall
      summary_dict["Unlabeled Constituency F1"] = ul_const_f1
      summary_dict["Unlabeled Constituency precision"] = ul_const_prec
      summary_dict["Unlabeled Constituency recall"] = ul_const_recall
      task_to_f1["constituency"] = const_f1

    if self.config["coref_weight"] > 0:
      conll_results = conll.evaluate_conll(self.config["conll_eval_path"], coref_predictions, official_stdout)
      coref_conll_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
      summary_dict["Average F1 (conll)"] = coref_conll_f1
      print "Average F1 (conll): {:.2f}%".format(coref_conll_f1)

      p,r,f = coref_evaluator.get_prf()
      summary_dict["Average Coref F1 (py)"] = f
      print "Average F1 (py): {:.2f}%".format(f * 100)
      summary_dict["Average Coref precision (py)"] = p
      print "Average precision (py): {:.2f}%".format(p * 100)
      summary_dict["Average Coref recall (py)"] = r
      print "Average recall (py): {:.2f}%".format(r * 100)

      task_to_f1["coref"] = coref_conll_f1

      for k, evaluator in sorted(mention_evaluators.items(), key=operator.itemgetter(0)):
        tags = ["{} {} @ {}".format("Mentions", t, _k_to_tag(k)) for t in ("R", "P", "F")]
        results_to_print = []
        for t, v in zip(tags, evaluator.metrics()):
          results_to_print.append("{:<10}: {:.4f}".format(t, v))
          summary_dict[t] = v
        print ", ".join(results_to_print)

    summary_dict["Dev Loss"] = total_loss / len(self.coref_eval_data)

    print "Decoding took {}.".format(str(datetime.timedelta(seconds=int(elapsed_time))))
    print "Decoding speed: {}/document, or {}/sentence.".format(
        str(datetime.timedelta(seconds=int(elapsed_time / len(self.coref_eval_data)))),
        str(datetime.timedelta(seconds=int(elapsed_time / len(self.eval_data))))
    )

    metric_names = self.config["main_metrics"].split("_")
    main_metric = sum([task_to_f1[t] for t in metric_names]) / len(metric_names)
    print "Combined metric ({}): {}".format(self.config["main_metrics"], main_metric)

    return util.make_summary(summary_dict), main_metric, task_to_f1

