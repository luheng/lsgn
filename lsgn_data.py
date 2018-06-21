import tensorflow as tf
import tensorflow_hub as hub

import h5py
import json
import numpy as np
import random
import threading

from input_utils import *
import util
import srl_eval_utils


# Names for the "given" tensors.
_input_names = [
    "tokens", "context_word_emb", "head_word_emb", "lm_emb", "char_idx", "text_len",
    "doc_id", "is_training",
    "gold_predicates", "num_gold_predicates",
]

# Names for the "gold" tensors.
_label_names = [
    "predicates", "arg_starts", "arg_ends", "arg_labels", "srl_len",
]

# Name for predicted tensors.
_predict_names = [
    "candidate_starts", "candidate_ends", "candidate_arg_scores", "candidate_pred_scores",
    "arg_starts", "arg_ends", "predicates", "num_args", "num_preds", "arg_labels", "srl_scores",
    "arg_scores", "pred_scores", "head_scores"
]


class LSGNData(object):
  def __init__(self, config):
    self.config = config
    self.context_embeddings = util.EmbeddingDictionary(config["context_embeddings"])
    self.head_embeddings = util.EmbeddingDictionary(config["head_embeddings"],
                                                    maybe_cache=self.context_embeddings)
    self.char_embedding_size = config["char_embedding_size"]
    self.char_dict = util.load_char_dict(config["char_vocab_path"])
      
    self.lm_file = None
    self.lm_hub = None
    self.lm_layers = 0  # TODO: Remove these.
    self.lm_size = 0
    if config["lm_path"]:
      if "tfhub" in config["lm_path"]:
        print "Using tensorflow hub:", config["lm_path"]
        self.lm_hub = hub.Module(config["lm_path"].encode("utf-8"), trainable=False) 
      else:
        self.lm_file = h5py.File(self.config["lm_path"], "r")
      self.lm_layers = self.config["lm_layers"]
      self.lm_size = self.config["lm_size"]

    self.adjunct_roles, self.core_roles = split_srl_labels(
        config["srl_labels"], config["include_c_v"])
    self.srl_labels_inv  = [""] + self.adjunct_roles + self.core_roles
    self.srl_labels = { l:i for i,l in enumerate(self.srl_labels_inv) }

    # IO Stuff.
    # Need to make sure they are in the same order as input_names + label_names
    self.input_props = [
        (tf.string, [None]), # String tokens.
        (tf.float32, [None, self.context_embeddings.size]), # Context embeddings.
        (tf.float32, [None, self.head_embeddings.size]), # Head embeddings.
        (tf.float32, [None, self.lm_size, self.lm_layers]), # LM embeddings.
        (tf.int32, [None, None]), # Character indices.
        (tf.int32, []),  # Text length.
        (tf.int32, []),  # Document ID.
        (tf.bool, []),  # Is training.
        (tf.int32, [None]),  # Gold predicate ids (for input).
        (tf.int32, []),  # Num gold predicates (for input).
        (tf.int32, [None]),  # Predicate ids (length=num_srl_relations).
        (tf.int32, [None]),  # Argument starts.
        (tf.int32, [None]),  # Argument ends.
        (tf.int32, [None]),  # SRL labels.
        (tf.int32, [])  # Number of SRL relations.
    ]
    self.input_names = _input_names
    self.label_names = _label_names
    self.predict_names = _predict_names
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
    self.input_dict = dict(zip(self.input_names, self.input_tensors[:num_features]))
    self.labels_dict = dict(zip(self.label_names, self.input_tensors[num_features:]))

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
            #num_mentions += len(e["coref"])
          #cluster_id_offset += len(example["clusters"])
          num_sentences += len(doc_examples[-1])
        print ("Load {} training documents with {} sentences".format(doc_id, num_sentences))
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
        # Clear out the batch buffer after each epoch to avoid the situation where the first document
        # in the next batch is the same one as the last document in the previous batch.
        if len(batch_buffer) > 0:
          batched_tensor_examples = [pad_batch_tensors(batch_buffer, tn) for tn in tensor_names]
          feed_dict = dict(zip(self.queue_input_tensors, batched_tensor_examples))
          session.run(self.enqueue_op, feed_dict=feed_dict)

    enqueue_thread = threading.Thread(target=_enqueue_loop)
    enqueue_thread.daemon = True
    enqueue_thread.start()

  def split_document_example(self, example):
    """Split document-based samples into sentence-based samples.
    """
    sentences = example["sentences"]
    split_examples = []
    word_offset = 0
    for i, sentence in enumerate(sentences):
      text_len = len(sentence)
      sent_example = {
        "sentence": sentence,
        "doc_key": example["doc_key"],
        "sent_id": i,
        "srl": example["srl"][i] if "srl" in example else [],
        "word_offset": word_offset,
        "sent_offset": example["sent_offset"]  # Sentence offset for the same doc ID.
      }
      word_offset += text_len
      split_examples.append(sent_example)
    return split_examples

  def tensorize_example(self, example, is_training):
    """Tensorize examples and caching embeddings.
    """
    sentence = example["sentence"]
    doc_key = example["doc_key"]
    sent_id = example["sent_id"]  # Number of sentence in the document.
    word_offset = example["word_offset"]
    text_len = len(sentence)

    lm_doc_key = None
    lm_sent_key = None
    # For historical reasons.
    if self.lm_file and "ontonotes" in self.config["lm_path"]:
      idx = doc_key.rfind("_")
      lm_doc_key = doc_key[:idx] + "/" + str(example["sent_offset"] + sent_id)
    elif self.lm_file and "conll05" in self.config["lm_path"]:
      lm_doc_key = doc_key[1:]  # "S1234" -> "1234"
    else:
      lm_doc_key = doc_key
      lm_sent_key = str(sent_id)
    # Load cached LM.
    lm_emb = load_lm_embeddings_for_sentence(
        self.lm_file, self.lm_layers, self.lm_size, lm_doc_key, lm_sent_key)

    max_word_length = max(max(len(w) for w in sentence), max(self.config["filter_widths"]))
    context_word_emb = np.zeros([text_len, self.context_embeddings.size])
    head_word_emb = np.zeros([text_len, self.head_embeddings.size])
    char_index = np.zeros([text_len, max_word_length])
    for j, word in enumerate(sentence):
      context_word_emb[j] = self.context_embeddings[word]
      head_word_emb[j] = self.head_embeddings[word]
      char_index[j, :len(word)] = [self.char_dict[c] for c in word]
    predicates, arg_starts, arg_ends, arg_labels = (
        tensorize_srl_relations(example["srl"], self.srl_labels,
                                filter_v_args=self.config["filter_v_args"]))
    # For gold predicate experiment.
    gold_predicates = get_all_predicates(example["srl"]) - word_offset
    example_tensor = {
      # Inputs.
      "tokens": sentence,
      "context_word_emb": context_word_emb,
      "head_word_emb": head_word_emb,
      "lm_emb": lm_emb,
      "char_idx": char_index,
      "text_len": text_len,
      "doc_id": example["doc_id"],
      "is_training": is_training,
      "gold_predicates": gold_predicates,
      "num_gold_predicates": len(gold_predicates),
      # Labels.
      "predicates": predicates - word_offset,
      "arg_starts": arg_starts - word_offset,
      "arg_ends": arg_ends - word_offset,
      "arg_labels": arg_labels,
      "srl_len": len(predicates),
    }
    return example_tensor

  def load_eval_data(self):
    eval_data = []
    eval_tensors = []
    coref_eval_data = []
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
        #num_mentions_in_doc += len(e["coref"])
      #assert num_mentions_in_doc == len(util.flatten(example["clusters"]))
      eval_tensors.append(doc_tensors)
      eval_data.extend(srl_eval_utils.split_example_for_eval(example))
      coref_eval_data.append(example)
    print("Loaded {} eval examples.".format(len(eval_data)))
    return eval_data, eval_tensors, coref_eval_data
 
