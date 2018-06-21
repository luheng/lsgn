#!/usr/bin/env python

import os
import sys
import time
import json
import numpy as np
import tensorflow as tf

import inference_utils
import input_utils
from lsgn_data import LSGNData
from lsgn_evaluator import LSGNEvaluator
from srl_model import SRLModel
import util


if __name__ == "__main__":
  util.set_gpus()
  name = sys.argv[1]
  input_filename = sys.argv[2]
  output_filename = sys.argv[3]

  print "Running experiment: {}.".format(name)
  config = util.get_config("experiments.conf")[name]
  config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))
  print "Loading data from: {}.".format(input_filename)
  config["eval_path"] = input_filename
  config["batch_size"] = -1
  config["max_tokens_per_batch"] = -1

  # Use dev lm, if provided.
  if config["lm_path"] and "lm_path_dev" in config and config["lm_path_dev"]:
    config["lm_path"] = config["lm_path_dev"]

  util.print_config(config)
  data = LSGNData(config)
  model = SRLModel(data, config)
  evaluator = LSGNEvaluator(config)
 
  # Load data and model. 
  eval_data, eval_tensors, doc_level_eval_data = data.load_eval_data()
  variables_to_restore = []
  for var in tf.global_variables():
    #print var.name
    if "module/" not in var.name:
      variables_to_restore.append(var)
  saver = tf.train.Saver(variables_to_restore)
  log_dir = config["log_dir"]

  with tf.Session() as session:
    checkpoint_path = os.path.join(log_dir, "model.max.ckpt")
    tf.global_variables_initializer().run()
    saver.restore(session, checkpoint_path)

    with open(output_filename, "w") as f:
      #for example_num, (tensorized_example, example) in enumerate(model.eval_data):
      for i, doc_tensors in enumerate(eval_tensors):
        feed_dict = dict(zip(
            data.input_tensors,
            [input_utils.pad_batch_tensors(doc_tensors, tn) for tn in data.input_names + data.label_names]))
        predict_names = []
        for tn in data.predict_names:
           if tn in model.predictions:
            predict_names.append(tn)
        predict_tensors = [model.predictions[tn] for tn in predict_names] + [model.loss]
        predict_tensors = session.run(predict_tensors, feed_dict=feed_dict)
        predict_dict = dict(zip(predict_names + ["loss"], predict_tensors))
        doc_example = doc_level_eval_data[i]
        sentences = doc_example["sentences"]
        predictions = inference_utils.srl_decode(
            sentences, predict_dict, data.srl_labels_inv, config)
        doc_example["predicted_srl"] = []
        word_offset = 0
        for j, sentence in enumerate(sentences):
          for pred, args in predictions["srl"][j].iteritems():
            doc_example["predicted_srl"].extend([
                [int(pred + word_offset), int(a[0] + word_offset),
                 int(a[1] + word_offset), a[2]] for a in args])
          word_offset += len(sentence)

        f.write(json.dumps(doc_example))
        f.write("\n")
        if (i + 1) % 10 == 0:
          print "Decoded {} documents.".format(i + 1)

