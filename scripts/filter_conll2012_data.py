"""Filter OntoNotes5 data based on CoNLL2012 (coref) doc ids.
"""
import codecs
import json
import sys


def filter_data(v5_input_file, doc_ids_file, output_file):
  doc_count = 0
  sentence_count = 0
  srl_count = 0
  ner_count = 0
  cluster_count = 0
  word_count = 0
  doc_ids = []
  doc_ids_to_keys = {}
  filtered_examples = {}

  with open(doc_ids_file, "r") as f:
    for line in f:
      doc_id = line.strip().split("annotations/")[1]
      doc_ids.append(doc_id)
      doc_ids_to_keys[doc_id] = []
    f.close()

  with codecs.open(v5_input_file, "r", "utf8") as f:
    for jsonline in f:
      example = json.loads(jsonline)
      doc_key = example["doc_key"]
      dk_prefix = "_".join(doc_key.split("_")[:-1])
      if dk_prefix not in doc_ids_to_keys:
        continue
      doc_ids_to_keys[dk_prefix].append(doc_key)
      filtered_examples[doc_key] = example

      sentences = example["sentences"]
      word_count += sum([len(s) for s in sentences])
      sentence_count += len(sentences)
      srl_count += sum([len(srl) for srl in example["srl"]])
      ner_count += sum([len(ner) for ner in example["ner"]])
      coref = example["clusters"]
      cluster_count += len(coref)
      doc_count += 1
    f.close()

  print ("Documents: {}\nSentences: {}\nWords: {}\nNER: {}, PAS: {}, Clusters: {}".format(
      doc_count, sentence_count, word_count, ner_count, srl_count, cluster_count))

  with codecs.open(output_file, "w", "utf8") as f:
    for doc_id in doc_ids:
      for key in doc_ids_to_keys[doc_id]:
        f.write(json.dumps(filtered_examples[key]))
        f.write("\n")
    f.close()

if __name__ == "__main__":
  filter_data(sys.argv[1], sys.argv[2], sys.argv[3])

