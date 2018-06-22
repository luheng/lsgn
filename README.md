# Labeled Span Graph Network (Under Construction)

This repository contains code and models for replicating results from the following publication:
* [Jointly Predicting Predicates and Arguments in Neural Semantic Role Labeling](https://arxiv.org/abs/1805.04787)
* [Luheng He](https://homes.cs.washington.edu/~luheng), [Kenton Lee](http://kentonl.com/), [Omer Levy](https://levyomer.wordpress.com/) and [Luke Zettlemoyer](https://www.cs.washington.edu/people/faculty/lsz)
* In ACL 2018

Part of the codebase is extended from [e2e-coref](https://github.com/kentonl/e2e-coref). 

### Requirements
* Python 2.7
* TensorFlow 1.8.0
* pyhocon (for parsing the configurations)
* [tensorflow_hub](https://www.tensorflow.org/hub/) (for loading ELMo)

## Getting Started
* sudo apt-get install tcsh (Only required for processing CoNLL05 data)
* [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings and the [srlconll](http://www.lsi.upc.edu/~srlconll/soft.html) scripts:  
`./scripts/fetch_required_data.sh` 
* Build kernels: `./scripts/build_custom_kernels.sh` (Please make adjustments to the script according to your OS/gcc version)
* Download [pretrained models](https://drive.google.com/drive/u/0/folders/1TPpXx1-0TDL-hcMDa0b6fwmvn2HIp-yk) by running `./scripts/fetch_all_models.sh` 

## Setting up for ELMo (in progress)
* Some of our models are trained with the [ELMo embeddings](https://allennlp.org/elmo). We use the ELMo model loaded by [tensorflow_hub](https://www.tensorflow.org/hub/modules/google/elmo/1).
* It is recommended to cache ELMo embeddings for training and validating efficiency. Instructions will be added soon.

## Making Predictions with Pretrained Models
* Please see `data/sample.jsonlines` for input format (json). Each json object can contain multiple sentences. 
* For example, run `python decoder.py conll2012_final data/sample.jsonlines sample.out` to predict SRL structures.
* The output will also be in json format, with an additional array storing the SRL tuples. For example, for the following input sentences:

`[["John", "told", "Pat", "to", "stop", "the", "robot", "immediately", "."], ["Pat", "refused", "."]]`

The following json object 

`"predicted_srl": [[1, 0, 0, "ARG0"], [1, 2, 2, "ARG2"], [1, 3, 7, "ARG1"], [4, 2, 2, "ARG0"], [4, 5, 6, "ARG1"], [4, 7, 7, "ARGM-TMP"], [10, 9, 9, "ARG0"]]` 

contains SRL predictions for the two sentences, formatted as `[predicate_position, argument_span_start, argument_end, role_label]`. The token ids are counted starting 0 from the beginning of the document (instead of the beginning of each sentence).

## CoNLL Data
For replicating results on CoNLL-2005 and CoNLL-2012 datasets, please follow the steps below.

### CoNLL-2005
The data is provided by:
[CoNLL-2005 Shared Task](http://www.lsi.upc.edu/~srlconll/soft.html),
but the original words are from the Penn Treebank dataset, which is not publicly available.
If you have the PTB corpus, you can run:  
` ./scripts/fetch_and_make_conll05_data.sh  /path/to/ptb/`  

### CoNLL-2012
You have to follow the instructions below to get CoNLL-2012 data
[CoNLL-2012](http://cemantix.org/data/ontonotes.html), this would result in a directory called `/path/to/conll-formatted-ontonotes-5.0`.
Run:  
`./scripts/make_conll2012_data.sh /path/to/conll-formatted-ontonotes-5.0`

## Training Instructions

* Experiment configurations are found in `experiments.conf`
* Choose an experiment that you would like to run, e.g. `conll2012_best`
* For a single-machine experiment, run the following two commands:
  * `python singleton.py <experiment>`
  * `python evaluator.py <experiment>`
* Results are stored in the `logs` directory and can be viewed via TensorBoard.
* For final evaluation of the checkpoint with the maximum dev F1:
  * Run `python test_single.py <experiment>` for the single-model evaluation. For example: `python test_single.py conll2012_final`

## Other Quirks
* It does not use GPUs by default. Instead, it looks for the `GPU` environment variable, which the code treats as shorthand for `CUDA_VISIBLE_DEVICES`.
* The evaluator should not be run on GPUs, since evaluating full documents does not fit within GPU memory constraints.
* The training runs indefinitely and needs to be terminated manually. The model generally converges at about 300k steps and within 48 hours.
* At test time, the code loads the entire GloVe 300D embedding file in the beginning, which would take a while.

