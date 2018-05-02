import datetime
import time


import coref_metrics
import debug_utils
import inference_utils
from input_utils import pad_batch_tensors
import operator
import srl_eval_utils
import util


class LSGNEvaluator(object):
  def __init__(self, config):
    self.config = config
    self.eval_data = None

  # TODO: Split to multiple functions.
  def evaluate(self, session, data, predictions, loss, official_stdout=False):
    if self.eval_data is None:
      self.eval_data, self.eval_tensors, self.coref_eval_data = data.load_eval_data()

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

    # Retrieval evaluators.
    arg_evaluators = { k:util.RetrievalEvaluator() for k in [-3, -2, -1, 30, 40, 50, 80, 100, 120, 150] }
    predicate_evaluators = { k:util.RetrievalEvaluator() for k in [-3, -2, -1, 10, 20, 30, 40, 50, 70] }
    mention_evaluators = { k:util.RetrievalEvaluator() for k in [-3, -2, -1, 10, 20, 30, 40, 50] }

    total_loss = 0
    total_num_predicates = 0
    total_gold_predicates = 0

    srl_comp_sents = 0
    srl_predictions = []
    ner_predictions = []
    coref_predictions = {}
    coref_evaluator = coref_metrics.CorefEvaluator()
    all_gold_predicates = []
    all_guessed_predicates = []

    start_time = time.time()
    debug_printer = debug_utils.DebugPrinter()
    sent_id = 0

    # Simple analysis.
    unique_core_role_violations = 0
    continuation_role_violations = 0
    reference_role_violations = 0
    gold_u_violations = 0
    gold_c_violations = 0
    gold_r_violations = 0

    for i, doc_tensors in enumerate(self.eval_tensors):
      feed_dict = dict(zip(
          data.input_tensors,
          [pad_batch_tensors(doc_tensors, tn) for tn in data.input_names + data.label_names]))
      predict_names = []
      for tn in data.predict_names:
        if tn in predictions:
          predict_names.append(tn)
      predict_tensors = [predictions[tn] for tn in predict_names] + [loss]
      predict_tensors = session.run(predict_tensors, feed_dict=feed_dict)
      predict_dict = dict(zip(predict_names + ["loss"], predict_tensors))

      doc_size = len(doc_tensors)
      doc_example = self.coref_eval_data[i]
      sentences = doc_example["sentences"]
      decoded_predictions = inference_utils.mtl_decode(
          sentences, predict_dict, data.srl_labels_inv, data.ner_labels_inv, self.config)

      if "srl" in decoded_predictions:
        srl_predictions.extend(decoded_predictions["srl"])
        # Evaluate retrieval.
        word_offset = 0
        for j in range(len(sentences)):
          text_length = len(sentences[j])
          na = predict_dict["num_args"][j]
          np = predict_dict["num_preds"][j]
          sent_example = self.eval_data[sent_id]  # sentence, srl, ner
          gold_args = set([])
          gold_preds = set([])
          guessed_preds = set([])
          for pred, args in sent_example[1].iteritems():
            filtered_args = [(a[0], a[1]) for a in args if a[2] not in ["V", "C-V"]]
            if len(filtered_args) > 0:
              gold_preds.add((pred, pred))
              gold_args.update(filtered_args)
          for pred, args in decoded_predictions["srl"][j].iteritems():
            guessed_preds.add((pred, pred, "V"))
          all_gold_predicates.append([(p[0], p[1], "V") for p in gold_preds])
          all_guessed_predicates.append(guessed_preds)

          srl_eval_utils.evaluate_retrieval(
                predict_dict["candidate_starts"][j], predict_dict["candidate_ends"][j],
                predict_dict["candidate_arg_scores"][j], predict_dict["arg_starts"][j][:na], predict_dict["arg_ends"][j][:na],
                gold_args, text_length, arg_evaluators)
          srl_eval_utils.evaluate_retrieval(
                range(text_length), range(text_length), predict_dict["candidate_pred_scores"][j],
                predict_dict["predicates"][j][:np], predict_dict["predicates"][j][:np], gold_preds, text_length,
                predicate_evaluators)

          # TODO: Move elsewhere.
          u_violations, c_violations, r_violations = debug_utils.srl_constraint_tracker(decoded_predictions["srl"][j])
          unique_core_role_violations += u_violations
          continuation_role_violations += c_violations
          reference_role_violations += r_violations
          total_num_predicates += len(decoded_predictions["srl"][j].keys())
          u_violations, c_violations, r_violations = debug_utils.srl_constraint_tracker(sent_example[1])
          gold_u_violations += u_violations
          gold_c_violations += c_violations
          gold_r_violations += r_violations
          total_gold_predicates += len(sent_example[1].keys())
          sent_id += 1
          word_offset += text_length

      if "ner" in decoded_predictions:
        ner_predictions.extend(decoded_predictions["ner"])

      if "predicted_clusters" in decoded_predictions:
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in doc_example["clusters"]]
        gold_mentions = set([])
        mention_to_gold = {}
        for gc in gold_clusters:
          for mention in gc:
            mention_to_gold[mention] = gc
            gold_mentions.add(mention)
        coref_evaluator.update(decoded_predictions["predicted_clusters"], gold_clusters, decoded_predictions["mention_to_predicted"],
                               mention_to_gold)
        coref_predictions[doc_example["doc_key"]] = decoded_predictions["predicted_clusters"]
        
        # Evaluate retrieval.
        doc_text_length = sum([len(s) for s in sentences])
        srl_eval_utils.evaluate_retrieval(
            predict_dict["candidate_mention_starts"], predict_dict["candidate_mention_ends"],
            predict_dict["candidate_mention_scores"], predict_dict["mention_starts"], predict_dict["mention_ends"],
            gold_mentions, doc_text_length, mention_evaluators)

      total_loss += predict_dict["loss"]
      if (i + 1) % 50 == 0:
        print ("Evaluated {}/{} documents.".format(i + 1, len(self.coref_eval_data)))

    debug_printer.close()
    summary_dict = {}
    task_to_f1 = {}  # From task name to F1.
    elapsed_time = time.time() - start_time

    sentences, gold_srl, gold_ner = zip(*self.eval_data)

    # Summarize results.
    if self.config["srl_weight"] > 0:
      precision, recall, f1, conll_precision, conll_recall, conll_f1, ul_prec, ul_recall, ul_f1, srl_label_mat, comp = (
          srl_eval_utils.compute_srl_f1(sentences, gold_srl, srl_predictions, self.config["srl_conll_eval_path"]))
      pid_precision, pred_recall, pid_f1, _, _, _, _ = srl_eval_utils.compute_span_f1(
          all_gold_predicates, all_guessed_predicates, "Predicate ID")
      task_to_f1["srl"] = conll_f1
      summary_dict["PAS F1"] = f1
      summary_dict["PAS precision"] = precision
      summary_dict["PAS recall"] = recall
      summary_dict["Unlabeled PAS F1"] = ul_f1
      summary_dict["Unlabeled PAS precision"] = ul_prec
      summary_dict["Unlabeled PAS recall"] = ul_recall
      summary_dict["CoNLL F1"] = conll_f1
      summary_dict["CoNLL precision"] = conll_precision
      summary_dict["CoNLL recall"] = conll_recall
      if total_num_predicates > 0:
        summary_dict["Unique core violations/Predicate"] = 1.0 * unique_core_role_violations / total_num_predicates
        summary_dict["Continuation violations/Predicate"] = 1.0 * continuation_role_violations / total_num_predicates
        summary_dict["Reference violations/Predicate"] = 1.0 * reference_role_violations / total_num_predicates

      print "Completely correct sentences: {}/{}".format(comp, 100.0 * comp / len(srl_predictions))

      for k, evaluator in sorted(arg_evaluators.items(), key=operator.itemgetter(0)):
        tags = ["{} {} @ {}".format("Args", t, _k_to_tag(k)) for t in ("R", "P", "F")]
        results_to_print = []
        for t, v in zip(tags, evaluator.metrics()):
          results_to_print.append("{:<10}: {:.4f}".format(t, v))
          summary_dict[t] = v
        print ", ".join(results_to_print)

      for k, evaluator in sorted(predicate_evaluators.items(), key=operator.itemgetter(0)):
        tags = ["{} {} @ {}".format("Predicates", t, _k_to_tag(k)) for t in ("R", "P", "F")]
        results_to_print = []
        for t, v in zip(tags, evaluator.metrics()):
          results_to_print.append("{:<10}: {:.4f}".format(t, v))
          summary_dict[t] = v
        print ", ".join(results_to_print)

      if total_num_predicates > 0:
        print ("Constraint voilations: U: {} ({}), C: {} ({}), R: {} ({})".format(
            1.0 * unique_core_role_violations / total_num_predicates, unique_core_role_violations,
            1.0 * continuation_role_violations / total_num_predicates, continuation_role_violations,
            1.0 * reference_role_violations / total_num_predicates, reference_role_violations))
      if total_gold_predicates > 0:
        print ("Gold constraint voilations: U: {} ({}), C: {} ({}), R: {} ({})".format(
            1.0 * gold_u_violations / total_gold_predicates, gold_u_violations,
            1.0 * gold_c_violations / total_gold_predicates, gold_c_violations,
            1.0 * gold_r_violations / total_gold_predicates, gold_r_violations))
      #for label_pair, freq in srl_label_mat.most_common():
      #  if label_pair[0] != label_pair[1] and freq > 10:
      #    print ("{}\t{}\t{}".format(label_pair[0], label_pair[1], freq))

    if self.config["ner_weight"] > 0:
      ner_precision, ner_recall, ner_f1, ul_ner_prec, ul_ner_recall, ul_ner_f1, ner_label_mat = (
          srl_eval_utils.compute_span_f1(gold_ner, ner_predictions, "NER"))
      summary_dict["NER F1"] = ner_f1
      summary_dict["NER precision"] = ner_precision
      summary_dict["NER recall"] = ner_recall
      summary_dict["Unlabeled NER F1"] = ul_ner_f1
      summary_dict["Unlabeled NER precision"] = ul_ner_prec
      summary_dict["Unlabeled NER recall"] = ul_ner_recall

      # Write NER prediction to IOB format and run official eval script.
      srl_eval_utils.print_to_iob2(sentences, gold_ner, ner_predictions, self.config["ner_conll_eval_path"])
      task_to_f1["ner"] = ner_f1
      #for label_pair, freq in ner_label_mat.most_common():
      #  if label_pair[0] != label_pair[1] and freq > 10:
      #    print ("{}\t{}\t{}".format(label_pair[0], label_pair[1], freq))

    if self.config["const_weight"] > 0:
      const_precision, const_recall, const_f1, ul_const_prec, ul_const_recall, ul_const_f1, const_label_mat = (
          srl_eval_utils.compute_span_f1(constituency_gold, constituency_predictions, "Constituency"))
      summary_dict["Constituency F1"] = const_f1
      summary_dict["Constituency precision"] = const_precision
      summary_dict["Constituency recall"] = const_recall
      summary_dict["Unlabeled Constituency F1"] = ul_const_f1
      summary_dict["Unlabeled Constituency precision"] = ul_const_prec
      summary_dict["Unlabeled Constituency recall"] = ul_const_recall
      task_to_f1["constituency"] = const_f1

    if self.config["coref_weight"] > 0:
      conll_results = conll.evaluate_conll(self.config["conll_eval_path"], coref_predictions, official_stdout)
      coref_conll_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
      summary_dict["Average F1 (conll)"] = coref_conll_f1
      print "Average F1 (conll): {:.2f}%".format(coref_conll_f1)

      p,r,f = coref_evaluator.get_prf()
      summary_dict["Average Coref F1 (py)"] = f
      print "Average F1 (py): {:.2f}%".format(f * 100)
      summary_dict["Average Coref precision (py)"] = p
      print "Average precision (py): {:.2f}%".format(p * 100)
      summary_dict["Average Coref recall (py)"] = r
      print "Average recall (py): {:.2f}%".format(r * 100)

      task_to_f1["coref"] = coref_conll_f1

      for k, evaluator in sorted(mention_evaluators.items(), key=operator.itemgetter(0)):
        tags = ["{} {} @ {}".format("Mentions", t, _k_to_tag(k)) for t in ("R", "P", "F")]
        results_to_print = []
        for t, v in zip(tags, evaluator.metrics()):
          results_to_print.append("{:<10}: {:.4f}".format(t, v))
          summary_dict[t] = v
        print ", ".join(results_to_print)

    summary_dict["Dev Loss"] = total_loss / len(self.coref_eval_data)

    print "Decoding took {}.".format(str(datetime.timedelta(seconds=int(elapsed_time))))
    print "Decoding speed: {}/document, or {}/sentence.".format(
        str(datetime.timedelta(seconds=int(elapsed_time / len(self.coref_eval_data)))),
        str(datetime.timedelta(seconds=int(elapsed_time / len(self.eval_data))))
    )

    metric_names = self.config["main_metrics"].split("_")
    main_metric = sum([task_to_f1[t] for t in metric_names]) / len(metric_names)
    print "Combined metric ({}): {}".format(self.config["main_metrics"], main_metric)

    return util.make_summary(summary_dict), main_metric, task_to_f1

