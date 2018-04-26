# For debugging.

import operator
import random
import math
import json
import threading
import numpy as np
import tensorflow as tf
import h5py

import util
import coref_ops
import srl_ops
import conll
import metrics

def load_lm_embeddings(self, doc_key):
  if self.lm_file is None:
    return np.zeros([0, 0, self.lm_layers, self.lm_size])
  file_key = doc_key.replace("/", ":")
  group = self.lm_file[file_key]
  num_sentences = len(list(group.keys()))
  sentences = [group[str(i)][...] for i in range(num_sentences)]
  lm_emb = np.zeros([num_sentences, max(s.shape[1] for s in sentences), self.lm_size, self.lm_layers])
  for i, s in enumerate(sentences):
    lm_emb[i, :s.shape[1], :, :] = s.transpose(1, 2, 0)
  return lm_emb

'''
  def tensorize_mentions(self, mentions):
    if len(mentions) > 0:
      starts, ends = zip(*mentions)
    else:
      starts, ends = [], []
    return np.array(starts), np.array(ends)

  def tensorize_span_labels(self, tuples, label_dict):
    if len(tuples) > 0:
      starts, ends, labels = zip(*tuples)
    else:
      starts, ends, labels = [], [], []
    return np.array(starts), np.array(ends), np.array([label_dict[c] for c in labels])
'''

def get_candidate_labels(candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
  same_start = tf.equal(tf.expand_dims(labeled_starts, 1), tf.expand_dims(candidate_starts, 0)) # [num_labeled, num_candidates]
  same_end = tf.equal(tf.expand_dims(labeled_ends, 1), tf.expand_dims(candidate_ends, 0)) # [num_labeled, num_candidates]
  same_span = tf.logical_and(same_start, same_end) # [num_labeled, num_candidates]
  candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span)) # [1, num_candidates]
  candidate_labels = tf.squeeze(candidate_labels, 0) # [num_candidates]
  return candidate_labels


def get_span_candidates(num_sentences, max_sentence_length, text_len_mask, num_words, max_span_width):
  sentence_indices = tf.tile(tf.expand_dims(tf.range(num_sentences), 1), [1, max_sentence_length]) # [num_sentences, max_sentence_length]
  flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, text_len_mask) # [num_words]

  candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1), [1, max_span_width]) # [num_words, max_span_width]
  candidate_ends = candidate_starts + tf.expand_dims(tf.range(max_span_width), 0) # [num_words, max_span_width]
  candidate_start_sentence_indices = tf.gather(flattened_sentence_indices, candidate_starts) # [num_words, max_span_width]
  candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends, num_words - 1)) # [num_words, max_span_width]
  candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices, candidate_end_sentence_indices)) # [num_words, max_span_width]
  flattened_candidate_mask = tf.reshape(candidate_mask, [-1]) # [num_words * max_span_width]
  candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [-1]), flattened_candidate_mask) # [num_candidates]
  candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask) # [num_candidates]
  return candidate_starts, candidate_ends


def get_top_spans(candidate_starts, candidate_ends, candidate_mention_scores, num_words, top_span_ratio):
  k = tf.to_int32(tf.floor(tf.to_float(num_words) * top_span_ratio))
  top_span_indices = srl_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
                                           tf.expand_dims(candidate_starts, 0),
                                           tf.expand_dims(candidate_ends, 0),
                                           tf.expand_dims(k, 0), num_words,
                                           True, True) # [1, k]
  top_span_indices.set_shape([1, None])
  top_span_indices = tf.squeeze(top_span_indices, 0) # [k]

  top_span_starts = tf.gather(candidate_starts, top_span_indices) # [k]
  top_span_ends = tf.gather(candidate_ends, top_span_indices) # [k]
  #top_span_emb = tf.gather(candidate_span_emb, top_span_indices) # [k, emb]
  #top_span_cluster_ids = tf.gather(candidate_cluster_ids, top_span_indices) # [k]
  top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices) # [k]
  return top_span_starts, top_span_ends, top_span_mention_scores, top_span_indices


def get_antecedents(k, max_antecedents):
  max_antecedents = tf.minimum(max_antecedents, k - 1)
  target_indices = tf.expand_dims(tf.range(k), 1) # [k, 1]
  antecedent_offsets = tf.expand_dims(tf.range(max_antecedents) + 1, 0) # [1, max_ant]
  raw_antecedents = target_indices - antecedent_offsets # [k, max_ant]
  antecedents_log_mask = tf.log(tf.to_float(raw_antecedents >= 0)) # [k, max_ant]
  antecedents = tf.maximum(raw_antecedents, 0) # [k, max_ant]
  return antecedents, antecedents_log_mask


def get_scores_and_loss(top_span_emb, top_span_mention_scores, antecedents, antecedents_log_mask, top_span_cluster_ids,
                        config, dropout):
  antecedent_scores = get_antecedent_scores(top_span_emb, top_span_mention_scores, antecedents, antecedents_log_mask,
                                            config, dropout) # [k, max_ant + 1]
  antecedent_cluster_ids = tf.gather(top_span_cluster_ids, antecedents) # [k, max_ant]
  antecedent_cluster_ids += tf.to_int32(antecedents_log_mask) # [k, max_ant]
  same_cluster_indicator = tf.equal(antecedent_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1)) # [k, max_ant]
  non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1) # [k, 1]
  pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator) # [k, max_ant]
  dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keep_dims=True)) # [k, 1]
  antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1) # [k, max_ant + 1]
  loss = softmax_loss(antecedent_scores, antecedent_labels) # [k]
  loss = tf.reduce_sum(loss) # []
  return antecedent_scores, loss


def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends, config, dropout):
  span_emb_list = []
  span_start_emb = tf.gather(context_outputs, span_starts) # [k, emb]
  span_emb_list.append(span_start_emb)
  span_end_emb = tf.gather(context_outputs, span_ends) # [k, emb]
  span_emb_list.append(span_end_emb)

  span_width = 1 + span_ends - span_starts # [k]
  if config["use_features"]:
    span_width_index = span_width - 1 # [k]
    span_width_emb = tf.gather(tf.get_variable("span_width_embeddings", [config["max_span_width"], config["feature_size"]]), span_width_index) # [k, emb]
    span_width_emb = tf.nn.dropout(span_width_emb, dropout)
    span_emb_list.append(span_width_emb)

  if self.config["model_heads"]:
    span_indices = tf.expand_dims(tf.range(config["max_span_width"]), 0) + tf.expand_dims(span_starts, 1) # [k, max_span_width]
    span_indices = tf.minimum(util.shape(context_outputs, 0) - 1, span_indices) # [k, max_span_width]
    span_text_emb = tf.gather(head_emb, span_indices) # [k, max_span_width, emb]
    with tf.variable_scope("head_scores"):
      head_scores = util.projection(context_outputs, 1) # [num_words, 1]
    span_head_scores = tf.gather(head_scores, span_indices) # [k, max_span_width, 1]
    span_mask = tf.expand_dims(tf.sequence_mask(span_width, config["max_span_width"], dtype=tf.float32), 2) # [k, max_span_width, 1]
    span_head_scores += tf.log(span_mask) # [k, max_span_width, 1]
    if self.config["sva"]:
      span_attention = tf.nn.relu(span_head_scores - tf.reduce_max(span_head_scores, axis=1, keep_dims=True) + 1) # [k, max_span_width, 1]
    else:
      span_attention = tf.nn.softmax(span_head_scores, dim=1) # [k, max_span_width, 1]
    span_head_emb = tf.reduce_sum(span_attention * span_text_emb, 1) # [k, emb]
    span_emb_list.append(span_head_emb)
    span_emb = tf.concat(span_emb_list, 1) # [k, emb]
  return span_emb


def get_mention_scores(span_emb):
  with tf.variable_scope("mention_scores"):
    return util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, 1]


def softmax_loss(antecedent_scores, antecedent_labels):
  gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels)) # [k, max_ant + 1]
  marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1]) # [k]
  log_norm = tf.reduce_logsumexp(antecedent_scores, [1]) # [k]
  return log_norm - marginalized_gold_scores # [k]


def bucket_distance(distances):
  """
  Places the given values (designed for distances) into 10 semi-logscale buckets:
  [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    """
  logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances))/math.log(2))) + 3
  use_identity = tf.to_int32(distances <= 4)
  combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
  return tf.minimum(combined_idx, 9)


def get_antecedent_scores(top_span_emb, top_span_mention_scores, antecedents, antecedents_log_mask,
                          config, dropout):
  k = util.shape(top_span_emb, 0)
  max_antecedents = util.shape(antecedents, 1)

  feature_emb_list = []
  '''if self.config["use_metadata"]:
     antecedent_speaker_ids = tf.gather(top_span_speaker_ids, antecedents) # [k, max_ant]
      same_speaker = tf.equal(tf.expand_dims(top_span_speaker_ids, 1), antecedent_speaker_ids) # [k, max_ant]
      speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]]), tf.to_int32(same_speaker)) # [k, max_ant, emb]
      feature_emb_list.append(speaker_pair_emb)
      tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [k, max_antecedents, 1]) # [k, max_ant, emb]
      feature_emb_list.append(tiled_genre_emb)'''

  if config["use_features"]:
    target_indices = tf.range(k) # [k]
    antecedent_distance = tf.expand_dims(target_indices, 1) - antecedents # [k, max_ant]
    antecedent_distance_buckets = bucket_distance(antecedent_distance) # [k, max_ant]
    antecedent_distance_emb = tf.gather(tf.get_variable("antecedent_distance_emb", [10, config["feature_size"]]), antecedent_distance_buckets) # [k, max_ant]
    feature_emb_list.append(antecedent_distance_emb)

  feature_emb = tf.concat(feature_emb_list, 2) # [k, max_ant, emb]
  feature_emb = tf.nn.dropout(feature_emb, dropout) # [k, max_ant, emb]

  antecedent_emb = tf.gather(top_span_emb, antecedents) # [k, max_ant, emb]
  target_emb = tf.expand_dims(top_span_emb, 1) # [k, 1, emb]
  similarity_emb = antecedent_emb * target_emb # [k, max_ant, emb]
  target_emb = tf.tile(target_emb, [1, max_antecedents, 1]) # [k, max_ant, emb]

  pair_emb = tf.concat([target_emb, antecedent_emb, similarity_emb, feature_emb], 2) # [k, max_ant, emb]

  with tf.variable_scope("antecedent_scores"):
    antecedent_scores = util.ffnn(pair_emb, config["ffnn_depth"], config["ffnn_size"], 1, dropout) # [k, max_ant, 1]
  antecedent_scores = tf.squeeze(antecedent_scores, 2) # [k, max_ant]
  antecedent_scores += antecedents_log_mask # [k, max_ant]
  antecedent_scores += tf.expand_dims(top_span_mention_scores, 1) + tf.gather(top_span_mention_scores, antecedents) # [k, max_ant]
  dummy_scores = tf.zeros([k, 1]) # [k, 1]
  antecedent_scores = tf.concat([dummy_scores, antecedent_scores], 1) # [k, max_ant + 1]
  return antecedent_scores # [k, max_ant + 1]


def flatten_emb_by_sentence(emb, text_len_mask):
  num_sentences = tf.shape(emb)[0]
  max_sentence_length = tf.shape(emb)[1]

  emb_rank = len(emb.get_shape())
  if emb_rank  == 2:
    flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
  elif emb_rank == 3:
    flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
  else:
    raise ValueError("Unsupported rank: {}".format(emb_rank))
  return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))


def get_predicted_antecedents(self, antecedents, antecedent_scores):
  predicted_antecedents = []
  for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
    if index < 0:
      predicted_antecedents.append(-1)
    else:
      predicted_antecedents.append(antecedents[i, index])
  return predicted_antecedents

