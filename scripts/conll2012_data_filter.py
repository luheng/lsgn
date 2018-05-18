"""Filter OntoNotes 5 data based on OntoNotes 4 doc ids.
"""
import codecs
import json
import sys

part = "test"
v4_input_file = "{}.english.jsonlines".format(part)
v5_input_file = "{}.english.v5.jsonlines".format(part)
output_file = "{}.english.mtl.jsonlines".format(part)


doc_count = 0
sentence_count = 0
srl_count = 0
ner_count = 0
cluster_count = 0
word_count = 0

v4_examples = []
doc_to_example = {}

with codecs.open(v4_input_file, "r", "utf8") as f:
  for jsonline in f:
    example = json.loads(jsonline)
    doc_key = example["doc_key"]
    example_id = len(v4_examples)
    doc_to_example[doc_key] = example_id
    v4_examples.append(example)
    sentences = example["sentences"]
    word_count += sum([len(s) for s in sentences])
    sentence_count += len(sentences)
    srl_count += sum([len(srl) for srl in example["srl"]])
    ner_count += sum([len(ner) for ner in example["ner"]])
    doc_count += 1
    coref = example["clusters"]
    cluster_count += len(coref)
  f.close() 

print ("Documents: {}\nSentences: {}\nWords: {}\nNER: {}, PAS: {}, Clusters: {}".format(
    doc_count, sentence_count, word_count, ner_count, srl_count, cluster_count))


doc_count = 0
sentence_count = 0
srl_count = 0
word_count = 0

with codecs.open(v5_input_file, "r", "utf8") as f:
  for jsonline in f:
    example = json.loads(jsonline)
    doc_key = example["doc_key"]
    if doc_key not in doc_to_example:
      continue
    example_id = doc_to_example[doc_key]
    v4_examples[example_id]["srl"] = example["srl"]
    sentences = example["sentences"]
    word_count += sum([len(s) for s in sentences])
    sentence_count += len(sentences)
    srl_count += sum([len(srl) for srl in example["srl"]])
    doc_count += 1
  f.close()

print ("Documents: {}\nSentences: {}\nWords: {}\nNER: {}, PAS: {}, Clusters: {}".format(
    doc_count, sentence_count, word_count, ner_count, srl_count, cluster_count))

with codecs.open(output_file, "w", "utf8") as f:
  for example in v4_examples:
    f.write(json.dumps(example))
    f.write("\n")
  f.close()


