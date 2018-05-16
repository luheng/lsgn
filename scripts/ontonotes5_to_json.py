#!/usr/bin/env python

import itertools
import re
import os
import sys
import json
import tempfile
import subprocess
import collections


BEGIN_DOCUMENT_REGEX = re.compile(r"#begin document \((.*)\); part (\d+)")


def flatten(l):
  return [item for sublist in l for item in sublist]


def get_doc_key(doc_id, part):
  return "{}_{}".format(doc_id, int(part))


class DocumentState(object):
  def __init__(self):
    self.doc_key = None
    self.text = []
    self.text_speakers = []
    self.speakers = []
    self.sentences = []
    self.constituents = []  # {}
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
  if language == "arabic":
    word = word[:word.find("#")]
  if word == "/." or word == "/?":
    return word[1:]
  else:
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
    label_set.add(label)
    stack.append((word_index, label))
    current_idx = next_idx

  for c in close_parens:
    try:
      assert c == ")"
    except AssertionError:
      print word_index, bit, spans, stack
      continue
    open_index, label = stack.pop()
    spans.append((open_index, word_index, label))
    ''' current_span = (open_index, word_index)
    if current_span in spans:
      spans[current_span] += "_" + label
    else:
      spans[current_span] = label
    spans[current_span] = label '''


def handle_line(line, document_state, language, labels, stats):
  begin_document_match = re.match(BEGIN_DOCUMENT_REGEX, line)
  if begin_document_match:
    document_state.assert_empty()
    document_state.doc_key = get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
    return None
  elif line.startswith("#end document"):
    document_state.assert_finalizable()
    finalized_state = document_state.finalize()
    stats["num_clusters"] += len(finalized_state["clusters"])
    stats["num_mentions"] += sum(len(c) for c in finalized_state["clusters"])
    # labels["{}_const_labels".format(language)].update(l for _, _, l in finalized_state["constituents"])
    # labels["ner"].update(l for _, _, l in finalized_state["ner"])
    return finalized_state
  else:
    row = line.split()
    # Starting a new sentence.
    if len(row) == 0:
      stats["max_sent_len_{}".format(language)] = max(len(document_state.text), stats["max_sent_len_{}".format(language)])
      stats["num_sents_{}".format(language)] += 1
      document_state.finalize_sentence()    
      return None
    assert len(row) >= 12

    doc_key = get_doc_key(row[0], row[1])
    word = normalize_word(row[3], language)
    parse = row[5]
    lemma = row[6]
    predicate_sense = row[7]
    speaker = row[9]
    ner = row[10]
    args = row[11:-1]
    coref = row[-1]

    word_index = len(document_state.text) + sum(len(s) for s in document_state.sentences)
    document_state.text.append(word)
    document_state.text_speakers.append(speaker)

    handle_bit(word_index, parse, document_state.const_stack, document_state.const_buffer, labels["categories"])
    handle_bit(word_index, ner, document_state.ner_stack, document_state.ner_buffer, labels["ner"])

    if len(document_state.argument_stacks) < len(args):
      document_state.argument_stacks = [[] for _ in args]
      document_state.argument_buffers = [[] for _ in args]

    for i, arg in enumerate(args):
      handle_bit(word_index, arg, document_state.argument_stacks[i], document_state.argument_buffers[i], labels["srl"])
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


def minimize_partition(input_path, output_path, language, labels, stats):
  count = 0
  print "Minimizing {}".format(input_path)
  with open(input_path, "r") as input_file:
    with open(output_path, "w") as output_file:
      document_state = DocumentState()
      for line in input_file.readlines():
        document = handle_line(line, document_state, language, labels, stats)
        if document is not None:
          output_file.write(json.dumps(document))
          output_file.write("\n")
          count += 1
          document_state = DocumentState()
  print "Wrote {} documents to {}".format(count, output_path)


if __name__ == "__main__":
  labels = collections.defaultdict(set)
  stats = collections.defaultdict(int)
  input_path = sys.argv[1]
  output_path = sys.argv[2]
  minimize_partition(input_path, output_path, "english", labels, stats)
  #minimize_language("english", labels, stats)
  #for k, v in labels.items():
  #  print("{} = [{}]".format(k, ", ".join("\"{}\"".format(label) for label in v)))
  #for k, v in stats.items():
  #  print("{} = {}".format(k, v))
