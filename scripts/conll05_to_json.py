#!/usr/bin/env python

import itertools
import re
import os
import sys
import json
import tempfile
import subprocess
import collections


def flatten(l):
  return [item for sublist in l for item in sublist]


class DocumentState(object):
  def __init__(self):
    self.doc_key = None
    self.text = []
    self.text_speakers = []
    self.speakers = []
    self.sentences = []
    self.constituents = []
    self.const_stack = []
    self.const_buffer = []
    self.ner = []
    self.ner_stack = []
    self.ner_buffer = []
    self.srl = []
    self.argument_stacks = []
    self.argument_buffers = []
    self.predicate_buffer = []
    self.clusters = collections.defaultdict(list)
    self.coref_stacks = collections.defaultdict(list)

  def assert_empty(self):
    assert self.doc_key is None
    assert len(self.text) == 0
    assert len(self.text_speakers) == 0
    assert len(self.speakers) == 0
    assert len(self.sentences) == 0
    # TODO
    assert len(self.srl) == 0
    assert len(self.predicate_buffer) == 0
    assert len(self.argument_buffers) == 0
    assert len(self.argument_stacks) == 0
    assert len(self.constituents) == 0
    assert len(self.const_stack) == 0
    assert len(self.const_buffer) == 0
    assert len(self.ner) == 0
    assert len(self.ner_stack) == 0
    assert len(self.ner_buffer) == 0
    assert len(self.coref_stacks) == 0
    assert len(self.clusters) == 0

  def assert_finalizable(self):
    assert self.doc_key is not None
    assert len(self.text) == 0
    assert len(self.text_speakers) == 0
    assert len(self.speakers) > 0
    assert len(self.sentences) > 0
    assert len(self.constituents) > 0
    assert len(self.const_stack) == 0
    assert len(self.ner_stack) == 0
    # TODO
    assert len(self.predicate_buffer) == 0
    assert all(len(s) == 0 for s in self.coref_stacks.values())

  def finalize_sentence(self):
    self.sentences.append(tuple(self.text))
    del self.text[:]
    self.speakers.append(tuple(self.text_speakers))
    del self.text_speakers[:]

    assert len(self.predicate_buffer) == len(self.argument_buffers)
    self.srl.append([])
    for pred, args in itertools.izip(self.predicate_buffer, self.argument_buffers):
      for start, end, label in args:
        self.srl[-1].append((pred, start, end, label))
    self.predicate_buffer = []
    self.argument_buffers = []
    self.argument_stacks = []
    self.constituents.append([c for c in self.const_buffer])
    self.const_buffer = []
    self.ner.append([c for c in self.ner_buffer])
    self.ner_buffer = []

  def finalize(self):
    merged_clusters = []
    for c1 in self.clusters.values():
      existing = None
      for m in c1:
        for c2 in merged_clusters:
          if m in c2:
            existing = c2
            break
        if existing is not None:
          break
      if existing is not None:
        print("Merging clusters (shouldn't happen very often.)")
        existing.update(c1)
      else:
        merged_clusters.append(set(c1))
    merged_clusters = [list(c) for c in merged_clusters]
    all_mentions = flatten(merged_clusters)
    assert len(all_mentions) == len(set(all_mentions))
    assert len(self.sentences) == len(self.srl)
    assert len(self.sentences) == len(self.constituents)
    assert len(self.sentences) == len(self.ner)
    return {
      "doc_key": self.doc_key,
      "sentences": self.sentences,
      "speakers": self.speakers,
      "srl": self.srl,
      "constituents": self.constituents,
      "ner": self.ner,
      "clusters": merged_clusters
    }


def normalize_word(word, language):
  return word


def handle_bit(word_index, bit, stack, spans, label_set):
  asterisk_idx = bit.find("*")
  if asterisk_idx >= 0:
    open_parens = bit[:asterisk_idx]
    close_parens = bit[asterisk_idx + 1:]
  else:
    open_parens = bit[:-1]
    close_parens = bit[-1]

  current_idx = open_parens.find("(")
  while current_idx >= 0:
    next_idx = open_parens.find("(", current_idx + 1)
    if next_idx >= 0:
      label = open_parens[current_idx + 1:next_idx]
    else:
      label = open_parens[current_idx + 1:]
    stack.append((word_index, label))
    label_set.add(label)
    current_idx = next_idx

  for c in close_parens:
    assert c == ")"
    open_index, label = stack.pop()
    spans.append((open_index, word_index, label))


def handle_line(line, document_state, language, num_cols, labels, stats):
  # document_state.assert_empty()
  # return None
  row = line.split()
  # Starting a new sentence.
  if len(row) == 0:
    # First finalize sentence.
    stats["max_sent_len_{}".format(language)] = max(
        len(document_state.text), stats["max_sent_len_{}".format(language)])
    stats["num_sents_{}".format(language)] += 1
    document_state.finalize_sentence()

    # Because we don't have document info for CoNLL05. Assume end the current document.
    document_state.assert_finalizable()
    finalized_state = document_state.finalize()
    stats["num_clusters"] += len(finalized_state["clusters"])
    stats["num_mentions"] += sum(len(c) for c in finalized_state["clusters"])
    return finalized_state

  assert len(row) >= num_cols + 1

  word = normalize_word(row[0], language)
  parse = "*" if num_cols < 2 else row[2]
  ner = "*" if num_cols < 2 else row[3]
  lemma = row[1] if num_cols < 2 else row[5]
  args = row[2:] if num_cols < 2 else row[6:]
  if num_cols < 2:
    predicate_sense = "-" if lemma == "-" else "0"
  else:
    predicate_sense = row[4]

  speaker = "-"
  coref = "-"

  word_index = len(document_state.text) + sum(len(s) for s in document_state.sentences)
  document_state.text.append(word)
  document_state.text_speakers.append(speaker)

  handle_bit(word_index, parse, document_state.const_stack, document_state.const_buffer,
             labels["categories"])
  handle_bit(word_index, ner, document_state.ner_stack, document_state.ner_buffer,
             labels["ner"])

  if len(document_state.argument_stacks) < len(args):
    document_state.argument_stacks = [[] for _ in args]
    document_state.argument_buffers = [[] for _ in args]

  for i, arg in enumerate(args):
    handle_bit(word_index, arg, document_state.argument_stacks[i],
               document_state.argument_buffers[i],
               labels["srl"])
  if predicate_sense != "-":
    document_state.predicate_buffer.append(word_index)
  if coref != "-":
    for segment in coref.split("|"):
      if segment[0] == "(":
        if segment[-1] == ")":
          cluster_id = int(segment[1:-1])
          document_state.clusters[cluster_id].append((word_index, word_index))
        else:
          cluster_id = int(segment[1:])
          document_state.coref_stacks[cluster_id].append(word_index)
      else:
        cluster_id = int(segment[:-1])
        start = document_state.coref_stacks[cluster_id].pop()
        document_state.clusters[cluster_id].append((start, word_index))
  return None


def minimize_partition(input_path, output_path, num_cols, labels, stats):
  count = 0
  language = "english"
  print "Minimizing {}".format(input_path)
  with open(input_path, "r") as input_file:
    with open(output_path, "w") as output_file:
      document_state = DocumentState()
      document_state.doc_key = "S{}".format(count)
      for line in input_file.readlines():
        document = handle_line(line, document_state, language, num_cols, labels, stats)
        if document is not None:
          output_file.write(json.dumps(document))
          output_file.write("\n")
          count += 1
          document_state = DocumentState()
          document_state.doc_key = "S{}".format(count)
  print "Wrote {} documents to {}".format(count, output_path)


if __name__ == "__main__":
  labels = collections.defaultdict(set)
  stats = collections.defaultdict(int)
  minimize_partition(sys.argv[1], sys.argv[2], int(sys.argv[3]), labels, stats)
  #minimize_partition("dev", "english", "05_conll", labels, stats)
  #minimize_partition("train", "english", "05_conll", labels, stats)
  #minimize_partition("test_wsj", "english", "05_conll", labels, stats)
  #minimize_partition("test_brown", "english", "05_conll", labels, stats)
  #for k, v in labels.items():
  #  print("{} = [{}]".format(k, ", ".join("\"{}\"".format(label) for label in v)))
  #for k, v in stats.items():
  #  print("{} = {}".format(k, v))

