#!/usr/bin/env python

import sys
import subprocess as sp

import util

def get_hostname(server):
  return server[:server.index(":")]

def qpython(py_script, name, args, hostname):
  util.mkdirs("logs")
  command = ["qsub", "-cwd", "-S", "/usr/bin/python", "-o", "logs/{}.o".format(name), "-e", "logs/{}.e".format(name), "-N", name, "-v", args, "-l", "hostname={}".format(hostname)]
  command.append(py_script)
  print(" ".join(command))
  sp.call(command)

if __name__ == "__main__":
  exp_name = sys.argv[1]
  util.mkdirs("logs")
  cluster_config = util.get_config("experiments.conf")[exp_name]["cluster"]
  for i, server in enumerate(cluster_config["addresses"]["ps"]):
    qpython("parameter_server.py", "ps{}_{}".format(i, exp_name), "EXP={}".format(exp_name), get_hostname(server))
  for i, server in enumerate(cluster_config["addresses"]["worker"]):
    qpython("worker.py", "w{}_{}".format(i, exp_name), "EXP={},TASK={}".format(exp_name, i), get_hostname(server))
  qpython("evaluator.py", "eval_{}".format(exp_name), "EXP={}".format(exp_name), "nlp2")
