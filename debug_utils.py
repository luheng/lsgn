import codecs
import numpy as np
import os

_CORE_ARGS = { "ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5", "ARGA",
               "A0", "A1", "A2", "A3", "A4", "A5", "AA" }


def logsumexp(arr):
  maxv = np.max(arr)
  lognorm = maxv + np.log(np.sum(np.exp(arr - maxv)))
  arr2 = np.exp(arr - lognorm)
  #print maxv, lognorm, arr, arr2
  return arr2


def srl_constraint_tracker(pred_to_args):
  unique_core_role_violations = 0
  continuation_role_violations = 0
  reference_role_violations = 0
  for pred_ids, args in pred_to_args.iteritems():
    # Sort by span start, assuming they are not overlapping.
    sorted_args = sorted(args, key=lambda x: x[0], reverse=True)
    core_args = set()
    base_args = set()
    for start, end, role in sorted_args:
      if role in _CORE_ARGS:
        if role in core_args:
          unique_core_role_violations += 1
        core_args.update([role])
      elif role.startswith("C-") and not role[2:] in base_args:
        continuation_role_violations += 1
      if not role.startswith("C-") and not role.startswith("R-"):
        base_args.update(role)
    for start, end, role in sorted_args:
      if role.startswith("R-") and not role[2:] in base_args:
        reference_role_violations += 1
  return unique_core_role_violations, continuation_role_violations, reference_role_violations
    
  
def print_sentence_to_conll(fout, tokens, labels, head_scores, raw_head_scores=None):
  """token_info: Unnormalized head scores, etc.
  """
  for label_column in labels:
    assert len(label_column) == len(tokens)
  for i in range(len(tokens)):
    fout.write(tokens[i].ljust(10) + "\t")
    if raw_head_scores:
      for hs in raw_head_scores[i]:
        fout.write(str(round(hs, 3)).rjust(4) + "\t")
    for label_column, score_column in zip(labels, head_scores):
      fout.write(label_column[i].rjust(10) + "\t")
      if score_column[i] > 0:
        fout.write(str(round(score_column[i], 2)).rjust(4) + "\t")
      else:
        fout.write(" ".rjust(4) + "\t")
    fout.write("\n")
  fout.write("\n")



