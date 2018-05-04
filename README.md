# Labeled Span Graph Network

### Requirements
* Python 2.7
  * TensorFlow 1.8.0
  * pyhocon (for parsing the configurations)
  * tensorflow_hub (for ELMo)

### Setting Up

* Download pretrained word embeddings and build custom kernels by running `setup_all.sh`.
  * There are 3 platform-dependent ways to build custom TensorFlow kernels. Please comment/uncomment the appropriate lines in the script.
* Run one of the following:
  * To use the pretrained model only, run `setup_pretrained.sh`
  * To train your own models, run `setup_training.sh`
    * This assumes access to OntoNotes 5.0. Please edit the `ontonotes_path` variable.

## Training Instructions

* Experiment configurations are found in `experiments.conf`
* Choose an experiment that you would like to run, e.g. `best`
* For a single-machine experiment, run the following two commands:
  * `python singleton.py <experiment>`
  * `python evaluator.py <experiment>`
* Results are stored in the `logs` directory and can be viewed via TensorBoard.
* For final evaluation of the checkpoint with the maximum dev F1:
  * Run `python test_single.py <experiment>` for the single-model evaluation.

## Other Quirks

* It does not use GPUs by default. Instead, it looks for the `GPU` environment variable, which the code treats as shorthand for `CUDA_VISIBLE_DEVICES`.
* The evaluator should not be run on GPUs, since evaluating full documents does not fit within GPU memory constraints.
* The training runs indefinitely and needs to be terminated manually. The model generally converges at about 400k steps and within 48 hours.
