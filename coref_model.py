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

class CorefModel(object):
  def __init__(self, config):
    self.config = config
    self.context_embeddings = util.EmbeddingDictionary(config["context_embeddings"])
    self.head_embeddings = util.EmbeddingDictionary(config["head_embeddings"], maybe_cache=self.context_embeddings)
    self.char_embedding_size = config["char_embedding_size"]
    self.char_dict = util.load_char_dict(config["char_vocab_path"])
    self.max_span_width = config["max_span_width"]
    self.genres = { g:i for i,g in enumerate(config["genres"]) }
    if len(config["lm_path"]) > 0:
      self.lm_file = h5py.File(self.config["lm_path"], "r")
      self.lm_layers = self.config["lm_layers"]
      self.lm_size = self.config["lm_size"]
    else:
      self.lm_file = None
      self.lm_layers = 0
      self.lm_size = 0
    self.const_labels = { l:i for i,l in enumerate([""] + config["const_labels"]) }
    self.ner_labels = { l:i for i,l in enumerate([""] + config["ner_labels"]) }
    self.eval_data = None # Load eval data lazily.

    input_props = []
    input_props.append((tf.float32, [None, None, self.context_embeddings.size])) # Context embeddings.
    input_props.append((tf.float32, [None, None, self.head_embeddings.size])) # Head embeddings.
    input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers])) # LM embeddings.
    input_props.append((tf.int32, [None, None, None])) # Character indices.
    input_props.append((tf.int32, [None])) # Text lengths.
    input_props.append((tf.int32, [None])) # Speaker IDs.
    input_props.append((tf.int32, [])) # Genre.
    input_props.append((tf.bool, [])) # Is training.
    input_props.append((tf.int32, [None])) # Gold starts.
    input_props.append((tf.int32, [None])) # Gold ends.
    input_props.append((tf.int32, [None])) # Cluster ids.
    input_props.append((tf.int32, [None])) # Constituent starts.
    input_props.append((tf.int32, [None])) # Constituent ends.
    input_props.append((tf.int32, [None])) # Constituent labels.
    input_props.append((tf.int32, [None])) # NER starts.
    input_props.append((tf.int32, [None])) # NER ends.
    input_props.append((tf.int32, [None])) # NER labels.

    self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
    dtypes, shapes = zip(*input_props)
    queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
    self.enqueue_op = queue.enqueue(self.queue_input_tensors)
    self.input_tensors = queue.dequeue()

    self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)
    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    self.reset_global_step = tf.assign(self.global_step, 0)
    learning_rate = tf.train.exponential_decay(self.config["learning_rate"], self.global_step,
                                               self.config["decay_frequency"], self.config["decay_rate"], staircase=True)
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
    for var in tf.trainable_variables():
      print var


  def start_enqueue_thread(self, session):
      with open(self.config["train_path"]) as f:
        train_examples = [json.loads(jsonline) for jsonline in f.readlines()]
      def _enqueue_loop():
        while True:
          random.shuffle(train_examples)
          total_num_clusters = 0
          total_num_mentions = 0
          for example in train_examples:
            total_num_clusters += len(example["clusters"])
            total_num_mentions += len(util.flatten(example["clusters"]))
            tensorized_example = self.tensorize_example(example, is_training=True)
            feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
            session.run(self.enqueue_op, feed_dict=feed_dict)
          print "Load {} examples, including {} clusters and {} mentions in total".format(
              len(train_examples), total_num_clusters, total_num_mentions)
      enqueue_thread = threading.Thread(target=_enqueue_loop)
      enqueue_thread.daemon = True
      enqueue_thread.start()

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

  def tensorize_example(self, example, is_training):
    clusters = example["clusters"]

    gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
    gold_mention_map = {m:i for i,m in enumerate(gold_mentions)}
    cluster_ids = np.zeros(len(gold_mentions))
    for cluster_id, cluster in enumerate(clusters):
      for mention in cluster:
        cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

    sentences = example["sentences"]
    num_words = sum(len(s) for s in sentences)
    speakers = util.flatten(example["speakers"])
    const_starts, const_ends, const_labels = [], [], [] # self.tensorize_span_labels(example["constituents"], self.const_labels)
    ner_starts, ner_ends, ner_labels = [], [], [] # self.tensorize_span_labels(example["ner"], self.ner_labels)

    const_spans = zip(const_starts, const_ends)
    assert len(const_spans) == len(set(const_spans))
    ner_spans = zip(ner_starts, ner_ends)
    assert len(ner_spans) == len(set(ner_spans))

    assert num_words == len(speakers)

    max_sentence_length = max(len(s) for s in sentences)
    max_word_length = max(max(max(len(w) for w in s) for s in sentences), max(self.config["filter_widths"]))
    text_len = np.array([len(s) for s in sentences])
    context_word_emb = np.zeros([len(sentences), max_sentence_length, self.context_embeddings.size])
    head_word_emb = np.zeros([len(sentences), max_sentence_length, self.head_embeddings.size])
    char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
    for i, sentence in enumerate(sentences):
      for j, word in enumerate(sentence):
        context_word_emb[i, j] = self.context_embeddings[word]
        head_word_emb[i, j] = self.head_embeddings[word]
        char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]

    speaker_dict = { s:i for i,s in enumerate(set(speakers)) }
    speaker_ids = np.array([speaker_dict[s] for s in speakers])

    doc_key = example["doc_key"]
    genre = self.genres[doc_key[:2]]

    gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)

    lm_emb = self.load_lm_embeddings(doc_key)

    example_tensors = (context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, const_starts, const_ends, const_labels, ner_starts, ner_ends, ner_labels)

    if is_training and len(sentences) > self.config["max_training_sentences"]:
      return self.truncate_example(*example_tensors)
    else:
      return example_tensors

  def truncate_example(self, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, const_starts, const_ends, const_labels, ner_starts, ner_ends, ner_labels):
    max_training_sentences = self.config["max_training_sentences"]
    num_sentences = context_word_emb.shape[0]
    assert num_sentences > max_training_sentences

    sentence_offset = random.randint(0, num_sentences - max_training_sentences)
    word_offset = text_len[:sentence_offset].sum()
    num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
    context_word_emb = context_word_emb[sentence_offset:sentence_offset + max_training_sentences,:,:]
    head_word_emb = head_word_emb[sentence_offset:sentence_offset + max_training_sentences,:,:]
    if self.lm_file is not None:
      lm_emb = lm_emb[sentence_offset:sentence_offset + max_training_sentences,:,:,:]
    char_index = char_index[sentence_offset:sentence_offset + max_training_sentences,:,:]
    text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

    speaker_ids = speaker_ids[word_offset: word_offset + num_words]
    gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)
    gold_starts = gold_starts[gold_spans] - word_offset
    gold_ends = gold_ends[gold_spans] - word_offset
    cluster_ids = cluster_ids[gold_spans]
    '''
    const_spans = np.logical_and(const_ends >= word_offset, const_starts < word_offset + num_words)
    const_starts = const_starts[const_spans] - word_offset
    const_ends = const_ends[const_spans] - word_offset
    const_labels = const_labels[const_spans]
    ner_spans = np.logical_and(ner_ends >= word_offset, ner_starts < word_offset + num_words)
    ner_starts = ner_starts[ner_spans] - word_offset
    ner_ends = ner_ends[ner_spans] - word_offset
    ner_labels = ner_labels[ner_spans]
    '''

    return context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, const_starts, const_ends, const_labels, ner_starts, ner_ends, ner_labels

  def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
    same_start = tf.equal(tf.expand_dims(labeled_starts, 1), tf.expand_dims(candidate_starts, 0)) # [num_labeled, num_candidates]
    same_end = tf.equal(tf.expand_dims(labeled_ends, 1), tf.expand_dims(candidate_ends, 0)) # [num_labeled, num_candidates]
    same_span = tf.logical_and(same_start, same_end) # [num_labeled, num_candidates]
    candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span)) # [1, num_candidates]
    candidate_labels = tf.squeeze(candidate_labels, 0) # [num_candidates]
    return candidate_labels

  def get_predictions_and_loss(self, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, const_starts, const_ends, const_labels, ner_starts, ner_ends, ner_labels):
    self.dropout = 1 - (tf.to_float(is_training) * self.config["dropout_rate"])
    self.lexical_dropout = 1 - (tf.to_float(is_training) * self.config["lexical_dropout_rate"])
    self.lstm_dropout = 1 - (tf.to_float(is_training) * self.config["lstm_dropout_rate"])

    num_sentences = tf.shape(context_word_emb)[0]
    max_sentence_length = tf.shape(context_word_emb)[1]

    context_emb_list = [context_word_emb]
    head_emb_list = [head_word_emb]

    if self.config["char_embedding_size"] > 0:
      char_emb = tf.gather(tf.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]]), char_index) # [num_sentences, max_sentence_length, max_word_length, emb]
      flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2), util.shape(char_emb, 3)]) # [num_sentences * max_sentence_length, max_word_length, emb]
      flattened_aggregated_char_emb = util.cnn(flattened_char_emb, self.config["filter_widths"], self.config["filter_size"]) # [num_sentences * max_sentence_length, emb]
      aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length, util.shape(flattened_aggregated_char_emb, 1)]) # [num_sentences, max_sentence_length, emb]
      context_emb_list.append(aggregated_char_emb)
      head_emb_list.append(aggregated_char_emb)

    if self.lm_file is not None:
      lm_emb_size = util.shape(lm_emb, 2)
      lm_num_layers = util.shape(lm_emb, 3)
      with tf.variable_scope("lm_aggregation"):
        self.lm_weights = tf.nn.softmax(tf.get_variable("lm_scores", [self.lm_layers], initializer=tf.constant_initializer(0.0)))
        self.lm_scaling = tf.get_variable("lm_scaling", [], initializer=tf.constant_initializer(1.0))
      flattened_lm_emb = tf.reshape(lm_emb, [num_sentences * max_sentence_length * lm_emb_size, lm_num_layers])
      flattened_aggregated_lm_emb = tf.matmul(flattened_lm_emb, tf.expand_dims(self.lm_weights, 1)) # [num_sentences * max_sentence_length * emb, 1]
      aggregated_lm_emb = tf.reshape(flattened_aggregated_lm_emb, [num_sentences, max_sentence_length, lm_emb_size])
      aggregated_lm_emb *= self.lm_scaling
      context_emb_list.append(aggregated_lm_emb)

    context_emb = tf.concat(context_emb_list, 2) # [num_sentences, max_sentence_length, emb]
    head_emb = tf.concat(head_emb_list, 2) # [num_sentences, max_sentence_length, emb]
    context_emb = tf.nn.dropout(context_emb, self.lexical_dropout) # [num_sentences, max_sentence_length, emb]
    head_emb = tf.nn.dropout(head_emb, self.lexical_dropout) # [num_sentences, max_sentence_length, emb]

    text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length) # [num_sentence, max_sentence_length]

    contextualizers = {
      "lstm": self.lstm_contextualize,
      "transformer": self.transformer_contextualize
    }
    context_outputs = contextualizers[self.config["contextualizer"]](context_emb, text_len, text_len_mask) # [num_words, emb]
    num_words = util.shape(context_outputs, 0)

    # TODO: Revert later.
    #genre_emb = tf.gather(tf.get_variable("genre_embeddings", [len(self.genres), self.config["feature_size"]]), genre) # [emb]
    genre_emb = None

    sentence_indices = tf.tile(tf.expand_dims(tf.range(num_sentences), 1), [1, max_sentence_length]) # [num_sentences, max_sentence_length]
    flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, text_len_mask) # [num_words]
    flattened_head_emb = self.flatten_emb_by_sentence(head_emb, text_len_mask) # [num_words]

    candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1), [1, self.max_span_width]) # [num_words, max_span_width]
    candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width), 0) # [num_words, max_span_width]
    candidate_start_sentence_indices = tf.gather(flattened_sentence_indices, candidate_starts) # [num_words, max_span_width]
    candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends, num_words - 1)) # [num_words, max_span_width]
    candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices, candidate_end_sentence_indices)) # [num_words, max_span_width]
    flattened_candidate_mask = tf.reshape(candidate_mask, [-1]) # [num_words * max_span_width]
    candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [-1]), flattened_candidate_mask) # [num_candidates]
    candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask) # [num_candidates]

    candidate_const_labels = self.get_candidate_labels(candidate_starts, candidate_ends, const_starts, const_ends, const_labels) # [num_candidates]
    candidate_ner_labels = self.get_candidate_labels(candidate_starts, candidate_ends, ner_starts, ner_ends, ner_labels) # [num_candidates]
    candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends, cluster_ids) # [num_candidates]

    candidate_span_emb = self.get_span_emb(flattened_head_emb, context_outputs, candidate_starts, candidate_ends) # [num_candidates, emb]
    candidate_mention_scores =  self.get_mention_scores(candidate_span_emb) # [k, 1]
    candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1) # [k]

    k = tf.to_int32(tf.floor(tf.to_float(tf.shape(context_outputs)[0]) * self.config["top_span_ratio"]))
    top_span_indices = srl_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
                                             tf.expand_dims(candidate_starts, 0),
                                             tf.expand_dims(candidate_ends, 0),
                                             tf.expand_dims(k, 0),
                                             util.shape(context_outputs, 0),
                                             True, True) # [1, k]
    top_span_indices.set_shape([1, None])
    top_span_indices = tf.squeeze(top_span_indices, 0) # [k]

    top_span_starts = tf.gather(candidate_starts, top_span_indices) # [k]
    top_span_ends = tf.gather(candidate_ends, top_span_indices) # [k]
    top_span_emb = tf.gather(candidate_span_emb, top_span_indices) # [k, emb]
    top_span_cluster_ids = tf.gather(candidate_cluster_ids, top_span_indices) # [k]
    top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices) # [k]
    top_span_speaker_ids = tf.gather(speaker_ids, top_span_starts) # [k]

    max_antecedents = tf.minimum(self.config["max_antecedents"], k - 1)

    target_indices = tf.expand_dims(tf.range(k), 1) # [k, 1]
    antecedent_offsets = tf.expand_dims(tf.range(max_antecedents) + 1, 0) # [1, max_ant]
    raw_antecedents = target_indices - antecedent_offsets # [k, max_ant]
    antecedents_log_mask = tf.log(tf.to_float(raw_antecedents >= 0)) # [k, max_ant]
    antecedents = tf.maximum(raw_antecedents, 0) # [k, max_ant]

    antecedent_scores = self.get_antecedent_scores(top_span_emb, top_span_mention_scores, top_span_speaker_ids, antecedents, antecedents_log_mask, genre_emb) # [k, max_ant + 1]

    antecedent_cluster_ids = tf.gather(top_span_cluster_ids, antecedents) # [k, max_ant]
    antecedent_cluster_ids += tf.to_int32(antecedents_log_mask) # [k, max_ant]
    same_cluster_indicator = tf.equal(antecedent_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1)) # [k, max_ant]
    non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1) # [k, 1]
    pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator) # [k, max_ant]
    dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keep_dims=True)) # [k, 1]
    antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1) # [k, max_ant + 1]
    loss = self.softmax_loss(antecedent_scores, antecedent_labels) # [k]
    loss = tf.reduce_sum(loss) # []

    if self.config["const_weight"] > 0:
      with tf.variable_scope("constituency_scores"):
        candidate_const_scores = util.ffnn(candidate_span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], len(self.const_labels), self.dropout) # [num_candidates, num_labels]
      const_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=candidate_const_labels, logits=candidate_const_scores) # [num_candidates]
      const_loss = tf.reduce_sum(const_loss) # []
      loss += self.config["const_weight"] * const_loss

    if self.config["ner_weight"] > 0:
      with tf.variable_scope("ner_scores"):
        candidate_ner_scores = util.ffnn(candidate_span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], len(self.ner_labels), self.dropout) # [num_candidates, num_labels]
      ner_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=candidate_ner_labels, logits=candidate_ner_scores) # [num_candidates]
      ner_loss = tf.reduce_sum(ner_loss) # []
      loss += self.config["ner_weight"] * ner_loss

    return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, antecedents, antecedent_scores], loss

  def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends):
    span_emb_list = []

    span_start_emb = tf.gather(context_outputs, span_starts) # [k, emb]
    span_emb_list.append(span_start_emb)

    span_end_emb = tf.gather(context_outputs, span_ends) # [k, emb]
    span_emb_list.append(span_end_emb)

    span_width = 1 + span_ends - span_starts # [k]

    if self.config["use_features"]:
      span_width_index = span_width - 1 # [k]
      span_width_emb = tf.gather(tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]]), span_width_index) # [k, emb]
      span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
      span_emb_list.append(span_width_emb)

    if self.config["model_heads"]:
      span_indices = tf.expand_dims(tf.range(self.config["max_span_width"]), 0) + tf.expand_dims(span_starts, 1) # [k, max_span_width]
      span_indices = tf.minimum(util.shape(context_outputs, 0) - 1, span_indices) # [k, max_span_width]
      span_text_emb = tf.gather(head_emb, span_indices) # [k, max_span_width, emb]
      with tf.variable_scope("head_scores"):
        self.head_scores = util.projection(context_outputs, 1) # [num_words, 1]
      span_head_scores = tf.gather(self.head_scores, span_indices) # [k, max_span_width, 1]
      span_mask = tf.expand_dims(tf.sequence_mask(span_width, self.config["max_span_width"], dtype=tf.float32), 2) # [k, max_span_width, 1]
      span_head_scores += tf.log(span_mask) # [k, max_span_width, 1]
      if self.config["sva"]:
        span_attention = tf.nn.relu(span_head_scores - tf.reduce_max(span_head_scores, axis=1, keep_dims=True) + 1) # [k, max_span_width, 1]
      else:
        span_attention = tf.nn.softmax(span_head_scores, dim=1) # [k, max_span_width, 1]
      span_head_emb = tf.reduce_sum(span_attention * span_text_emb, 1) # [k, emb]
      span_emb_list.append(span_head_emb)

    span_emb = tf.concat(span_emb_list, 1) # [k, emb]
    return span_emb

  def get_mention_scores(self, span_emb):
    with tf.variable_scope("mention_scores"):
      return util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, 1]

  def softmax_loss(self, antecedent_scores, antecedent_labels):
    gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels)) # [k, max_ant + 1]
    marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1]) # [k]
    log_norm = tf.reduce_logsumexp(antecedent_scores, [1]) # [k]
    return log_norm - marginalized_gold_scores # [k]

  def bucket_distance(self, distances):
    """
    Places the given values (designed for distances) into 10 semi-logscale buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    """
    logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances))/math.log(2))) + 3
    use_identity = tf.to_int32(distances <= 4)
    combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
    return tf.minimum(combined_idx, 9)

  def get_antecedent_scores(self, top_span_emb, top_span_mention_scores, top_span_speaker_ids, antecedents, antecedents_log_mask, genre_emb):
    k = util.shape(top_span_emb, 0)
    max_antecedents = util.shape(antecedents, 1)

    feature_emb_list = []

    if self.config["use_metadata"]:
      antecedent_speaker_ids = tf.gather(top_span_speaker_ids, antecedents) # [k, max_ant]
      same_speaker = tf.equal(tf.expand_dims(top_span_speaker_ids, 1), antecedent_speaker_ids) # [k, max_ant]
      speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]]), tf.to_int32(same_speaker)) # [k, max_ant, emb]
      feature_emb_list.append(speaker_pair_emb)

      tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [k, max_antecedents, 1]) # [k, max_ant, emb]
      feature_emb_list.append(tiled_genre_emb)

    if self.config["use_features"]:
      target_indices = tf.range(k) # [k]
      antecedent_distance = tf.expand_dims(target_indices, 1) - antecedents # [k, max_ant]
      antecedent_distance_buckets = self.bucket_distance(antecedent_distance) # [k, max_ant]
      antecedent_distance_emb = tf.gather(tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]]), antecedent_distance_buckets) # [k, max_ant]
      feature_emb_list.append(antecedent_distance_emb)

    feature_emb = tf.concat(feature_emb_list, 2) # [k, max_ant, emb]
    feature_emb = tf.nn.dropout(feature_emb, self.dropout) # [k, max_ant, emb]

    antecedent_emb = tf.gather(top_span_emb, antecedents) # [k, max_ant, emb]
    target_emb = tf.expand_dims(top_span_emb, 1) # [k, 1, emb]
    similarity_emb = antecedent_emb * target_emb # [k, max_ant, emb]
    target_emb = tf.tile(target_emb, [1, max_antecedents, 1]) # [k, max_ant, emb]

    pair_emb = tf.concat([target_emb, antecedent_emb, similarity_emb, feature_emb], 2) # [k, max_ant, emb]

    with tf.variable_scope("antecedent_scores"):
      antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, max_ant, 1]
    antecedent_scores = tf.squeeze(antecedent_scores, 2) # [k, max_ant]
    antecedent_scores += antecedents_log_mask # [k, max_ant]
    antecedent_scores += tf.expand_dims(top_span_mention_scores, 1) + tf.gather(top_span_mention_scores, antecedents) # [k, max_ant]
    dummy_scores = tf.zeros([k, 1]) # [k, 1]
    antecedent_scores = tf.concat([dummy_scores, antecedent_scores], 1) # [k, max_ant + 1]
    return antecedent_scores # [k, max_ant + 1]

  def flatten_emb_by_sentence(self, emb, text_len_mask):
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

  def lstm_contextualize(self, text_emb, text_len, text_len_mask):
    num_sentences = tf.shape(text_emb)[0]

    current_inputs = text_emb # [num_sentences, max_sentence_length, emb]

    for layer in xrange(self.config["contextualization_layers"]):
      with tf.variable_scope("layer_{}".format(layer)):
        with tf.variable_scope("fw_cell"):
          cell_fw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences, self.lstm_dropout)
        with tf.variable_scope("bw_cell"):
          cell_bw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences, self.lstm_dropout)
        state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]), tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
        state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]), tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))

        (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=cell_fw,
          cell_bw=cell_bw,
          inputs=current_inputs,
          sequence_length=text_len,
          initial_state_fw=state_fw,
          initial_state_bw=state_bw)

        text_outputs = tf.concat([fw_outputs, bw_outputs], 2) # [num_sentences, max_sentence_length, emb]
        text_outputs = tf.nn.dropout(text_outputs, self.lstm_dropout)
        if layer > 0:
          highway_gates = tf.sigmoid(util.projection(text_outputs, util.shape(text_outputs, 2))) # [num_sentences, max_sentence_length, emb]
          text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
        current_inputs = text_outputs

    return self.flatten_emb_by_sentence(text_outputs, text_len_mask)

  def transformer_contextualize(self, text_emb, text_len, text_len_mask):
    num_sentences = tf.shape(text_emb)[0]
    max_sentence_length = tf.shape(text_emb)[1]

    timing_signal = util.get_timing_signal_1d(max_sentence_length, 100) # [1, max_sentence_length, emb]
    timing_signal = tf.tile(timing_signal, [num_sentences, 1, 1]) # [num_sentences, max_sentence_length, emb]

    x = tf.concat([text_emb, timing_signal], 2) # [num_sentences, max_sentence_length, emb]

    hidden_size = self.config["contextualization_size"]
    log_text_len_mask = tf.expand_dims(tf.expand_dims(tf.log(tf.to_float(text_len_mask)), 1), 1)  # [num_sentences, 1, 1, max_sentence_length]
    num_heads = self.config["num_transformer_heads"]
    qkv_size = hidden_size/num_heads
    scaling = 1/math.sqrt(qkv_size)

    with tf.variable_scope("projected_text"):
      h = util.projection(x, hidden_size) # [num_sentence, max_sentence_length, emb]

    for layer in xrange(self.config["contextualization_layers"]):
      with tf.variable_scope("layer_{}".format(layer)):
        with tf.variable_scope("qk"):
          qkv = util.projection(h, num_heads * qkv_size * 3) # [num_sentences, max_sentence_length, emb]
        qkv = tf.reshape(qkv, [num_sentences, max_sentence_length, num_heads, qkv_size * 3]) # [num_sentences, max_sentence_length, num_heads, emb]
        qkv = tf.transpose(qkv, [0, 2, 1, 3]) # [num_sentences, num_heads, max_sentence_length, emb]
        q, k, v = tf.split(qkv, 3, 3) # [num_sentences, num_heads, max_sentence_length, emb]
        attention_scores = scaling * tf.matmul(q, k, transpose_b=True) # [num_sentences, num_heads, max_sentence_length, max_sentence_length]
        attention_scores += log_text_len_mask # [num_sentences, num_heads, max_sentence_length, max_sentence_length]
        attention_weights = tf.nn.softmax(attention_scores) # [num_sentences, num_heads, max_sentence_length, max_sentence_length]
        a = tf.matmul(attention_weights, v) # [num_sentences, num_heads, max_sentence_length, emb]
        a = tf.transpose(a, [0, 2, 1, 3]) # [num_sentences, max_sentence_length, num_heads, emb]
        a = tf.reshape(a, [num_sentences, max_sentence_length, num_heads * qkv_size]) # [num_sentences, max_sentence_length, emb]
        with tf.variable_scope("concat"):
          concat = util.projection(tf.concat([x, h, a], 2), hidden_size * 2) # [num_sentences, max_sentence_length, emb]
        j, f = tf.split(concat, 2, 2) # [num_sentences, max_sentence_length, emb]
        j = tf.nn.relu(j) # [num_sentences, max_sentence_length, emb]
        f = tf.sigmoid(f) # [num_sentences, max_sentence_length, emb]
        h = f * j + (1 - f) * h  # [num_sentences, max_sentence_length, emb]
    return self.flatten_emb_by_sentence(h, text_len_mask)

  def evaluate_top_spans(self, candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, gold_starts, gold_ends, example, evaluators):
    text_length = sum(len(s) for s in example["sentences"])
    gold_spans = set(zip(gold_starts, gold_ends))

    if len(candidate_starts) > 0:
      sorted_starts, sorted_ends, sorted_scores = zip(*sorted(zip(candidate_starts, candidate_ends, candidate_mention_scores), key=operator.itemgetter(2), reverse=True))
    else:
      sorted_starts = []
      sorted_ends = []

    for k, evaluator in evaluators.items():
      if k == -3:
        predicted_spans = set(zip(candidate_starts, candidate_ends)) & gold_spans
      else:
        if k == -2:
          predicted_starts = top_span_starts
          predicted_ends = top_span_ends
          print "Predicted", zip(sorted_starts, sorted_ends, sorted_scores)[:len(gold_spans)]
          print "Gold", sorted(list(gold_spans))
        elif k == 0:
          is_predicted = candidate_mention_scores > 0
          predicted_starts = candidate_starts[is_predicted]
          predicted_ends = candidate_ends[is_predicted]
        else:
          if k == -1:
            num_predictions = len(gold_spans)
          else:
            num_predictions = (k * text_length) / 100
          predicted_starts = sorted_starts[:num_predictions]
          predicted_ends = sorted_ends[:num_predictions]
        predicted_spans = set(zip(predicted_starts, predicted_ends))
      evaluator.update(gold_set=gold_spans, predicted_set=predicted_spans)

  def get_predicted_antecedents(self, antecedents, antecedent_scores):
    predicted_antecedents = []
    for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
      if index < 0:
        predicted_antecedents.append(-1)
      else:
        predicted_antecedents.append(antecedents[i, index])
    return predicted_antecedents

  def get_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents):
    mention_to_predicted = {}
    predicted_clusters = []
    for i, predicted_index in enumerate(predicted_antecedents):
      if predicted_index < 0:
        continue
      assert i > predicted_index
      predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
      if predicted_antecedent in mention_to_predicted:
        predicted_cluster = mention_to_predicted[predicted_antecedent]
      else:
        predicted_cluster = len(predicted_clusters)
        predicted_clusters.append([predicted_antecedent])
        mention_to_predicted[predicted_antecedent] = predicted_cluster

      mention = (int(top_span_starts[i]), int(top_span_ends[i]))
      predicted_clusters[predicted_cluster].append(mention)
      mention_to_predicted[mention] = predicted_cluster

    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    mention_to_predicted = { m:predicted_clusters[i] for m,i in mention_to_predicted.items() }

    return predicted_clusters, mention_to_predicted

  def evaluate_coref(self, top_span_starts, top_span_ends, predicted_antecedents, gold_clusters, evaluator):
    gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
    mention_to_gold = {}
    for gc in gold_clusters:
      for mention in gc:
        mention_to_gold[mention] = gc

    predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
    evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
    return predicted_clusters

  def load_eval_data(self):
    if self.eval_data is None:
      with open(self.config["eval_path"]) as f:
        self.eval_data = map(lambda example: (self.tensorize_example(example, is_training=False), example), (json.loads(jsonline) for jsonline in f.readlines()))
      num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data)
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
    top_span_evaluators = { k:util.RetrievalEvaluator() for k in [-3, -2, -1, 0, 10, 15, 20, 25, 30, 40, 50] }

    coref_predictions = {}
    coref_evaluator = metrics.CorefEvaluator()

    for example_num, (tensorized_example, example) in enumerate(self.eval_data):
      _, _, _, _, _, _, _, _, gold_starts, gold_ends, _, _, _, _, _, _, _ = tensorized_example
      feed_dict = {i:t for i,t in zip(self.input_tensors, tensorized_example)}
      candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, antecedents, antecedent_scores = session.run(self.predictions, feed_dict=feed_dict)

      self.evaluate_top_spans(candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, gold_starts, gold_ends, example, top_span_evaluators)
      predicted_antecedents = self.get_predicted_antecedents(antecedents, antecedent_scores)

      coref_predictions[example["doc_key"]] = self.evaluate_coref(top_span_starts, top_span_ends, predicted_antecedents, example["clusters"], coref_evaluator)

      if example_num % 10 == 0:
        print "Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data))

    summary_dict = {}
    for k, evaluator in sorted(top_span_evaluators.items(), key=operator.itemgetter(0)):
      tags = ["{} @ {}".format(t, _k_to_tag(k)) for t in ("R", "P", "F")]
      results_to_print = []
      for t, v in zip(tags, evaluator.metrics()):
        results_to_print.append("{:<10}: {:.2f}".format(t, v))
        summary_dict[t] = v
      print ", ".join(results_to_print)

    conll_results = conll.evaluate_conll(self.config["conll_eval_path"], coref_predictions, official_stdout)
    average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
    summary_dict["Average F1 (conll)"] = average_f1
    print "Average F1 (conll): {:.2f}%".format(average_f1)

    p,r,f = coref_evaluator.get_prf()
    summary_dict["Average F1 (py)"] = f
    print "Average F1 (py): {:.2f}%".format(f * 100)
    summary_dict["Average precision (py)"] = p
    print "Average precision (py): {:.2f}%".format(p * 100)
    summary_dict["Average recall (py)"] = r
    print "Average recall (py): {:.2f}%".format(r * 100)

    return util.make_summary(summary_dict), average_f1
