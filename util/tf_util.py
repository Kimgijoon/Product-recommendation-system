import tensorflow.compat.v1 as tf


class TFHelper(object):
  
  def __init__(self, max_seq_length):

    self.max_seq_length = max_seq_length

  def _parser(self, tfrecord):
    """
    Args:
      tfrecord
    Returns:
      example
    """
    name_to_features = {
      'input_ids': tf.io.FixedLenFeature([self.max_seq_length], tf.int64),
      'input_mask': tf.io.FixedLenFeature([self.max_seq_length], tf.int64),
      'segment_ids': tf.io.FixedLenFeature([self.max_seq_length], tf.int64)
    }

    example = tf.io.parse_single_example(tfrecord, name_to_features)

    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, dtype=tf.int32)
      example[name] = t

    return example

  def get_features_from_tfrecords(self, tfrecord_path: Union[List, str], batch_size: int, is_training: bool):
    """
    Args:
    Returns:
    """
    if type(tfrecord_path) is not list:
      tfrecord_path = [tfrecord_path]

    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(tfrecord_path))
      d = d.shuffle(buffer_size=len(tfrecord_path))

      cycle_length = min(4, len(tfrecord_path))

      d = d.apply(tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, sloppy=is_training, cycle_length=cycle_length))
      d = d.shuffle(buffer_size=1024)
    else:
      d = tf.data.TFRecordDataset(tfrecord_path)

    d = d.apply(tf.data.experimental.map_and_batch(lambda record: self._parser(record),
                                                  batch_size=batch_size,
                                                  num_parallel_batches=4,
                                                  drop_remainder=True))

    tfrecord_iterator = d.make_initializable_iterator()
    tfrecord_iterator_initializer = tfrecord_iterator.make_initializer(d, name='tfrecord_iterator_initializer')

    features = tfrecord_iterator.get_next()

    return features, tfrecord_iterator, tfrecord_iterator_initializer
