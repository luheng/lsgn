#!/usr/bin/env python

import os
import sys
sys.path.append(os.getcwd())
import json
import time
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import coref_model as cm
import util

if __name__ == "__main__":
  if len(sys.argv) > 1:
    name = sys.argv[1]
  else:
    name = os.environ["EXP"]
  config = util.get_config("experiments.conf")[name]
  report_frequency = config["report_frequency"]
  eval_frequency = config["eval_frequency"]

  util.print_config(config)

  if "GPU" in os.environ:
    util.set_gpus(int(os.environ["GPU"]))
  else:
    util.set_gpus()

  model = cm.CorefModel(config)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())
    model.start_enqueue_thread(session)
    accumulated_loss = 0.0
    initial_time = time.time()

    profile = False
    if profile:
      options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
    else:
      run_metadata = None
      options = None

    while True:
      tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op],
                                               options=options,
                                               run_metadata=run_metadata)
      if profile:
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open("timeline_02_step_%d.json" % tf_global_step, "w") as f:
          f.write(chrome_trace)

      accumulated_loss += tf_loss

      if tf_global_step % report_frequency == 0:
        total_time = time.time() - initial_time
        steps_per_second = tf_global_step / total_time

        average_loss = accumulated_loss / report_frequency
        print "[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second)
        accumulated_loss = 0.0

      if tf_global_step % eval_frequency == 0:
        _, eval_f1 = model.evaluate(session)
        print "[{}] evaL_f1={:.2f}".format(tf_global_step, eval_f1)
