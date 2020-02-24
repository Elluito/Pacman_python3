import distutils

import numpy as np
import tensorflow as tf

if distutils.version.LooseVersion(tf.__version__) < '1.14':
    raise Exception('This notebook is compatible with TensorFlow 1.14 or higher, for TensorFlow 1.13 or lower please use the previous version at https://github.com/tensorflow/tpu/blob/r1.13/tools/colab/shakespeare_with_tpu_and_keras.ipynb')

# This address identifies the TPU we'll use when configuring TensorFlow.
TPU_WORKER = 'grpc:// 10.240.1.2:8470'
SHAKESPEARE_TXT = '/content/shakespeare.txt'

def transform(txt):
  return np.asarray([ord(c) for c in txt if ord(c) < 255], dtype=np.int32)

def input_fn(seq_len=100, batch_size=1024):
  """Return a dataset of source and target sequences for training."""
  with tf.io.gfile.GFile(SHAKESPEARE_TXT, 'r') as f:
    txt = f.read()

  source = tf.constant(transform(txt), dtype=tf.int32)

  ds = tf.data.Dataset.from_tensor_slices(source).batch(seq_len+1, drop_remainder=True)

  def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

  BUFFER_SIZE = 10000
  ds = ds.map(split_input_target).shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True)

  return ds.repeat()