import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

coref_op_library = tf.load_op_library("./coref_kernels.so")

extract_mentions = coref_op_library.extract_mentions
tf.NotDifferentiable("ExtractMentions")

extract_mentions_cky = coref_op_library.extract_mentions_cky
tf.NotDifferentiable("ExtractMentionsCKY")
