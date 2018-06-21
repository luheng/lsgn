import math
import tensorflow as tf
import util

import srl_ops


def flatten_emb(emb):
  num_sentences = tf.shape(emb)[0]
  max_sentence_length = tf.shape(emb)[1]
  emb_rank = len(emb.get_shape())
  if emb_rank  == 2:
    flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
  elif emb_rank == 3:
    flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
  else:
    raise ValueError("Unsupported rank: {}".format(emb_rank))
  return flattened_emb


def flatten_emb_by_sentence(emb, text_len_mask):
  num_sentences = tf.shape(emb)[0]
  max_sentence_length = tf.shape(emb)[1]
  flattened_emb = flatten_emb(emb)
  return tf.boolean_mask(flattened_emb,
                         tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))


def batch_gather(emb, indices):
  # TODO: Merge with util.batch_gather.
  """
  Args:
    emb: Shape of [num_sentences, max_sentence_length, (emb)]
    indices: Shape of [num_sentences, k, (l)]
  """
  num_sentences = tf.shape(emb)[0] 
  max_sentence_length = tf.shape(emb)[1] 
  flattened_emb = flatten_emb(emb)  # [num_sentences * max_sentence_length, emb]
  offset = tf.expand_dims(tf.range(num_sentences) * max_sentence_length, 1)  # [num_sentences, 1]
  if len(indices.get_shape()) == 3:
    offset = tf.expand_dims(offset, 2)  # [num_sentences, 1, 1]
  return tf.gather(flattened_emb, indices + offset) 
  

def lstm_contextualize(text_emb, text_len, config, lstm_dropout):
  num_sentences = tf.shape(text_emb)[0]
  current_inputs = text_emb  # [num_sentences, max_sentence_length, emb]
  for layer in xrange(config["contextualization_layers"]):
    with tf.variable_scope("layer_{}".format(layer)):
      with tf.variable_scope("fw_cell"):
        cell_fw = util.CustomLSTMCell(config["contextualization_size"], num_sentences, lstm_dropout)
      with tf.variable_scope("bw_cell"):
        cell_bw = util.CustomLSTMCell(config["contextualization_size"], num_sentences, lstm_dropout)
      state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]),
                                               tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
      state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]),
                                               tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))
      (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=cell_fw,
          cell_bw=cell_bw,
          inputs=current_inputs,
          sequence_length=text_len,
          initial_state_fw=state_fw,
          initial_state_bw=state_bw)
      text_outputs = tf.concat([fw_outputs, bw_outputs], 2)  # [num_sentences, max_sentence_length, emb]
      text_outputs = tf.nn.dropout(text_outputs, lstm_dropout)
      if layer > 0:
        highway_gates = tf.sigmoid(util.projection(
            text_outputs, util.shape(text_outputs, 2)))  # [num_sentences, max_sentence_length, emb]
        text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
      current_inputs = text_outputs

  return text_outputs  # [num_sentences, max_sentence_length, emb]


def get_span_candidates(text_len, max_sentence_length, max_mention_width):
  """Get a list of candidate spans up to length W.
  Args:
    text_len: Tensor of [num_sentences,]
    max_sentence_length: Integer scalar.
    max_mention_width: Integer.
  """
  num_sentences = util.shape(text_len, 0)
  candidate_starts = tf.tile(
      tf.expand_dims(tf.expand_dims(tf.range(max_sentence_length), 0), 1),
      [num_sentences, max_mention_width, 1])  # [num_sentences, max_mention_width, max_sentence_length]
  candidate_widths = tf.expand_dims(tf.expand_dims(tf.range(max_mention_width), 0), 2)  # [1, max_mention_width, 1]
  candidate_ends = candidate_starts + candidate_widths  # [num_sentences, max_mention_width, max_sentence_length]
  
  candidate_starts = tf.reshape(candidate_starts, [num_sentences, max_mention_width * max_sentence_length])
  candidate_ends = tf.reshape(candidate_ends, [num_sentences, max_mention_width * max_sentence_length])
  candidate_mask = tf.less(
      candidate_ends,
      tf.tile(tf.expand_dims(text_len, 1), [1, max_mention_width * max_sentence_length])
  )  # [num_sentences, max_mention_width * max_sentence_length]

  # Mask to avoid indexing error.
  candidate_starts = tf.multiply(candidate_starts, tf.to_int32(candidate_mask))
  candidate_ends = tf.multiply(candidate_ends, tf.to_int32(candidate_mask))
  return candidate_starts, candidate_ends, candidate_mask  


def get_span_emb(head_emb, context_outputs, span_starts, span_ends, config, dropout):
  """Compute span representation shared across tasks.
  Args:
    head_emb: Tensor of [num_words, emb]
    context_outputs: Tensor of [num_words, emb]
    span_starts: [num_spans]
    span_ends: [num_spans]
  """
  text_length = util.shape(context_outputs, 0)
  num_spans = util.shape(span_starts, 0)

  span_start_emb = tf.gather(context_outputs, span_starts)  # [num_words, emb]
  span_end_emb = tf.gather(context_outputs, span_ends)  # [num_words, emb]
  span_emb_list = [span_start_emb, span_end_emb]

  span_width = 1 + span_ends - span_starts # [num_spans]
  max_arg_width = config["max_arg_width"]
  num_heads = config["num_attention_heads"]

  if config["use_features"]:
    span_width_index = span_width - 1  # [num_spans]
    span_width_emb = tf.gather(
        tf.get_variable("span_width_embeddings", [max_arg_width, config["feature_size"]]),
        span_width_index)  # [num_spans, emb]
    span_width_emb = tf.nn.dropout(span_width_emb, dropout)
    span_emb_list.append(span_width_emb)

  head_scores = None
  span_text_emb = None
  span_indices = None
  span_indices_log_mask = None

  if config["model_heads"]:
    span_indices = tf.minimum(
        tf.expand_dims(tf.range(max_arg_width), 0) + tf.expand_dims(span_starts, 1),
        text_length - 1)  # [num_spans, max_span_width]
    span_text_emb = tf.gather(head_emb, span_indices)  # [num_spans, max_arg_width, emb]
    span_indices_log_mask = tf.log(
        tf.sequence_mask(span_width, max_arg_width, dtype=tf.float32)) # [num_spans, max_arg_width]
    with tf.variable_scope("head_scores"):
      head_scores = util.projection(context_outputs, num_heads)  # [num_words, num_heads]
    span_attention = tf.nn.softmax(
      tf.gather(head_scores, span_indices) + tf.expand_dims(span_indices_log_mask, 2),
      dim=1)  # [num_spans, max_arg_width, num_heads]
    span_head_emb = tf.reduce_sum(span_attention * span_text_emb, 1)  # [num_spans, emb]
    span_emb_list.append(span_head_emb)

  span_emb = tf.concat(span_emb_list, 1) # [num_spans, emb]
  return span_emb, head_scores, span_text_emb, span_indices, span_indices_log_mask


def get_unary_scores(span_emb, config, dropout, num_labels = 1, name="span_scores"):
  """Compute span score with FFNN(span embedding).
  Args:
    span_emb: Tensor of [num_sentences, num_spans, emb].
  """
  with tf.variable_scope(name):
    scores = util.ffnn(span_emb, config["ffnn_depth"], config["ffnn_size"], num_labels,
                       dropout)  # [num_sentences, num_spans, num_labels] or [k, num_labels]
  if num_labels == 1:
    scores = tf.squeeze(scores, -1)  # [num_sentences, num_spans] or [k]
  return scores


def get_srl_scores(arg_emb, pred_emb, arg_scores, pred_scores, num_labels, config, dropout):
  num_sentences = util.shape(arg_emb, 0)
  num_args = util.shape(arg_emb, 1)
  num_preds = util.shape(pred_emb, 1)

  arg_emb_expanded = tf.expand_dims(arg_emb, 2)  # [num_sents, num_args, 1, emb]
  pred_emb_expanded = tf.expand_dims(pred_emb, 1)  # [num_sents, 1, num_preds, emb] 
  arg_emb_tiled = tf.tile(arg_emb_expanded, [1, 1, num_preds, 1])  # [num_sentences, num_args, num_preds, emb]
  pred_emb_tiled = tf.tile(pred_emb_expanded, [1, num_args, 1, 1])  # [num_sents, num_args, num_preds, emb]

  pair_emb_list = [arg_emb_tiled, pred_emb_tiled]
  pair_emb = tf.concat(pair_emb_list, 3)  # [num_sentences, num_args, num_preds, emb]
  pair_emb_size = util.shape(pair_emb, 3)
  flat_pair_emb = tf.reshape(pair_emb, [num_sentences * num_args * num_preds, pair_emb_size])

  flat_srl_scores = get_unary_scores(flat_pair_emb, config, dropout, num_labels - 1,
      "predicate_argument_scores")  # [num_sentences * num_args * num_predicates, 1]
  srl_scores = tf.reshape(flat_srl_scores, [num_sentences, num_args, num_preds, num_labels - 1])
  srl_scores += tf.expand_dims(tf.expand_dims(arg_scores, 2), 3) + tf.expand_dims(
      tf.expand_dims(pred_scores, 1), 3)  # [num_sentences, 1, max_num_preds, num_labels-1]
  
  dummy_scores = tf.zeros([num_sentences, num_args, num_preds, 1], tf.float32)
  srl_scores = tf.concat([dummy_scores, srl_scores], 3)  # [num_sentences, max_num_args, max_num_preds, num_labels] 
  return srl_scores  # [num_sentences, num_args, num_predicates, num_labels]


def get_batch_topk(candidate_starts, candidate_ends, candidate_scores, topk_ratio, text_len,
                   max_sentence_length, sort_spans=False, enforce_non_crossing=True):
  """
  Args:
    candidate_starts: [num_sentences, max_num_candidates]
    candidate_mask: [num_sentences, max_num_candidates]
    topk_ratio: A float number.
    text_len: [num_sentences,]
    max_sentence_length:
    enforce_non_crossing: Use regular top-k op if set to False.
 """
  num_sentences = util.shape(candidate_starts, 0)
  max_num_candidates = util.shape(candidate_starts, 1)

  topk = tf.maximum(tf.to_int32(tf.floor(tf.to_float(text_len) * topk_ratio)),
                    tf.ones([num_sentences,], dtype=tf.int32))  # [num_sentences]

  predicted_indices = srl_ops.extract_spans(
      candidate_scores, candidate_starts, candidate_ends, topk, max_sentence_length,
      sort_spans, enforce_non_crossing)  # [num_sentences, max_num_predictions]
  predicted_indices.set_shape([None, None])

  predicted_starts = batch_gather(candidate_starts, predicted_indices)  # [num_sentences, max_num_predictions]
  predicted_ends = batch_gather(candidate_ends, predicted_indices)  # [num_sentences, max_num_predictions]
  predicted_scores = batch_gather(candidate_scores, predicted_indices)  # [num_sentences, max_num_predictions]

  return predicted_starts, predicted_ends, predicted_scores, topk, predicted_indices


def get_srl_labels(arg_starts, arg_ends, predicates, labels, max_sentence_length):
  """
  Args:
    arg_starts: [num_sentences, max_num_args]
    arg_ends: [num_sentences, max_num_args]
    predicates: [num_sentences, max_num_predicates]
    labels: Dictionary of label tensors.
    max_sentence_length: An integer scalar.
  """
  num_sentences = util.shape(arg_starts, 0)
  max_num_args = util.shape(arg_starts, 1)
  max_num_preds = util.shape(predicates, 1)
  sentence_indices_2d = tf.tile(
      tf.expand_dims(tf.expand_dims(tf.range(num_sentences), 1), 2),
      [1, max_num_args, max_num_preds])  # [num_sentences, max_num_args, max_num_preds]
  tiled_arg_starts = tf.tile(
      tf.expand_dims(arg_starts, 2),
      [1, 1, max_num_preds])  # [num_sentences, max_num_args, max_num_preds]
  tiled_arg_ends = tf.tile(
      tf.expand_dims(arg_ends, 2),
      [1, 1, max_num_preds])  # [num_sentences, max_num_args, max_num_preds]
  tiled_predicates = tf.tile(
      tf.expand_dims(predicates, 1),
      [1, max_num_args, 1])  # [num_sentences, max_num_args, max_num_preds]
  pred_indices = tf.concat([
      tf.expand_dims(sentence_indices_2d, 3),
      tf.expand_dims(tiled_arg_starts, 3),
      tf.expand_dims(tiled_arg_ends, 3),
      tf.expand_dims(tiled_predicates, 3)], axis=3)  # [num_sentences, max_num_args, max_num_preds, 4]
 
  dense_srl_labels = get_dense_span_labels(
      labels["arg_starts"], labels["arg_ends"], labels["arg_labels"], labels["srl_len"], max_sentence_length,
      span_parents=labels["predicates"])  # [num_sentences, max_sent_len, max_sent_len, max_sent_len]
 
  srl_labels = tf.gather_nd(params=dense_srl_labels, indices=pred_indices)  # [num_sentences, max_num_args]
  return srl_labels


def get_dense_span_labels(span_starts, span_ends, span_labels, num_spans, max_sentence_length, span_parents=None):
  """Utility function to get dense span or span-head labels.
  Args:
    span_starts: [num_sentences, max_num_spans]
    span_ends: [num_sentences, max_num_spans]
    span_labels: [num_sentences, max_num_spans]
    num_spans: [num_sentences,]
    max_sentence_length:
    span_parents: [num_sentences, max_num_spans]. Predicates in SRL.
  """
  num_sentences = util.shape(span_starts, 0)
  max_num_spans = util.shape(span_starts, 1)
  # For padded spans, we have starts = 1, and ends = 0, so they don't collide with any existing spans.
  span_starts += (1 - tf.sequence_mask(num_spans, dtype=tf.int32))  # [num_sentences, max_num_spans]
  sentence_indices = tf.tile(
      tf.expand_dims(tf.range(num_sentences), 1),
      [1, max_num_spans])  # [num_sentences, max_num_spans]
  sparse_indices = tf.concat([
      tf.expand_dims(sentence_indices, 2),
      tf.expand_dims(span_starts, 2),
      tf.expand_dims(span_ends, 2)], axis=2)  # [num_sentences, max_num_spans, 3]
  if span_parents is not None:
    sparse_indices = tf.concat([
      sparse_indices, tf.expand_dims(span_parents, 2)], axis=2)  # [num_sentenes, max_num_spans, 4]

  rank = 3 if (span_parents is None) else 4
  # (sent_id, span_start, span_end) -> span_label
  dense_labels = tf.sparse_to_dense(
      sparse_indices = tf.reshape(sparse_indices, [num_sentences * max_num_spans, rank]),
      output_shape = [num_sentences] + [max_sentence_length] * (rank - 1),
      sparse_values = tf.reshape(span_labels, [-1]),
      default_value = 0,
      validate_indices = False)  # [num_sentences, max_sent_len, max_sent_len]
  return dense_labels
    

def get_srl_softmax_loss(srl_scores, srl_labels, num_predicted_args, num_predicted_preds):
  """Softmax loss with 2-D masking (for SRL).
  Args:
    srl_scores: [num_sentences, max_num_args, max_num_preds, num_labels]
    srl_labels: [num_sentences, max_num_args, max_num_preds]
    num_predicted_args: [num_sentences]
    num_predicted_preds: [num_sentences]
  """
  max_num_args = util.shape(srl_scores, 1)
  max_num_preds = util.shape(srl_scores, 2)
  num_labels = util.shape(srl_scores, 3)
  args_mask = tf.sequence_mask(num_predicted_args, max_num_args)  # [num_sentences, max_num_args]
  preds_mask = tf.sequence_mask(num_predicted_preds, max_num_preds)  # [num_sentences, max_num_preds]
  srl_loss_mask = tf.logical_and(
      tf.expand_dims(args_mask, 2),  # [num_sentences, max_num_args, 1]
      tf.expand_dims(preds_mask, 1)  # [num_sentences, 1, max_num_preds]
  )  # [num_sentences, max_num_args, max_num_preds]
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tf.reshape(srl_labels, [-1]),
      logits=tf.reshape(srl_scores, [-1, num_labels]),
      name="srl_softmax_loss")  # [num_sentences * max_num_args * max_num_preds]
  loss = tf.boolean_mask(loss, tf.reshape(srl_loss_mask, [-1]))
  loss.set_shape([None])
  loss = tf.reduce_sum(loss)
  return loss


def get_softmax_loss(scores, labels, candidate_mask):
  """Softmax loss with 1-D masking. (on Unary factors)
  Args:
    scores: [num_sentences, max_num_candidates, num_labels]
    labels: [num_sentences, max_num_candidates]
    candidate_mask: [num_sentences, max_num_candidates]
  """
  max_num_candidates = util.shape(scores, 1)
  num_labels = util.shape(scores, 2)
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tf.reshape(labels, [-1]), 
      logits=tf.reshape(scores, [-1, num_labels]),
      name="softmax_loss")  # [num_sentences, max_num_candidates]
  loss = tf.boolean_mask(loss, tf.reshape(candidate_mask, [-1]))
  loss.set_shape([None])
  return loss



