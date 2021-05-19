import os
import json
import random
import collections

import logging
import tensorflow as tf
from rich.logging import RichHandler
from typing import List, Dict, Optional
from sklearn.model_selection import train_test_split

from util.db_util import MongoController


FLAGS = tf.flags.FLAGS


class ClassifyInputExample(object):

  def __init__(self,
              guid: int,
              text_a: List[str],
              text_b: Optional[List[str]]=None,
              label: int=None):

    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class ClassifyInputFeatures(object):

  def __init__(self,
              input_ids: List[int],
              input_mask: List[int],
              segment_ids: List[int],
              label_id: int):

    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id


class CreateClassifyData(object):

  def __init__(self):

    FORMAT = "%(message)s"
    logging.basicConfig(level="NOTSET",
                        format=FORMAT,
                        datefmt="[%X]",
                        handlers=[RichHandler()])

    self.logger = logging.getLogger("rich")

  def run(self):
    """finetuning data 생성하는 함수"""
    db = MongoController(FLAGS.id, FLAGS.passwd, FLAGS.server_ip)

    data_dir = FLAGS.data_home_dir
    if tf.io.gfile.exists(data_dir) == False:
      tf.io.gfile.makedirs(data_dir)
    tfrecord_output_path = os.path.join(data_dir, FLAGS.classify_tfrecord_filename)
    val_tfrecord_output_path = os.path.join(data_dir, FLAGS.classify_val_tfrecord_filename)
    json_output_path = os.path.join(data_dir, FLAGS.classify_json_filename)
    prod2idx_output_path = os.path.join(data_dir, FLAGS.prod2idx_filename)
    labels_output_path = os.path.join(data_dir, FLAGS.label_filename)
    test_set_path = os.path.join(data_dir, FLAGS.test_filename)

    # Reading input raw data
    x, y, prod2idx, labels = db.get_train_data()

    # split data
    train_ratio, valid_ratio, test_ratio = [float(x) for x in FLAGS.split_ratio.split(',')]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio)
    x_val, x_test, y_val, y_test = train_test_split(x_test,
                                                    y_test,
                                                    test_size=test_ratio / (test_ratio + valid_ratio))
    x, y = x_train + x_val, y_train + y_val

    input_examples = []
    for i in range(len(x)):
      prod_list = x[i]
      label = labels[y[i]]

      example = ClassifyInputExample(guid=len(input_examples),
                                    text_a=prod_list,
                                    text_b=None,
                                    label=label)
      input_examples.append(example)

    self.logger.info(f"Reading input raw data completed. Total data size is [{len(input_examples)}].")

    # Shuffle
    rng = random.Random(FLAGS.random_seed)
    rng.shuffle(input_examples)
    self.logger.info("Data is shuffled.")

    # Get writer
    tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_output_path)
    val_tfrecord_writer = tf.python_io.TFRecordWriter(val_tfrecord_output_path)
    json_writer = tf.gfile.GFile(json_output_path, "w")

    # Make example to features
    written_count = 0
    val_written_count = 0
    for example in input_examples:
      feature = convert_single_example(example, FLAGS.max_seq_length, prod2idx)

      # Make TF record example with features
      features = collections.OrderedDict()
      features["input_ids"] = create_int_feature(feature.input_ids)
      features["input_mask"] = create_int_feature(feature.input_mask)
      features["segment_ids"] = create_int_feature(feature.segment_ids)
      features["label_id"] = create_int_feature(feature.label_id)
      tf_example = tf.train.Example(features=tf.train.Features(feature=features))

      # Make json example
      json_features = collections.OrderedDict()
      json_features["input_ids"] = feature.input_ids
      json_features["input_mask"] = feature.input_mask
      json_features["segment_ids"] = feature.segment_ids
      json_features["label_id"] = feature.label_id

      if val_written_count < len(x_val):
          # val Wrtier
          val_tfrecord_writer.write(tf_example.SerializeToString())
          val_written_count += 1
      else:
          # Wrtier
          tfrecord_writer.write(tf_example.SerializeToString())
          json_writer.write(json.dumps(json_features, ensure_ascii=False) + '\n')
          written_count += 1

    tfrecord_writer.close()
    json_writer.close()

    self.logger.info(f"Wrote {written_count} total instances")
    self.logger.info(f"Wrote {val_written_count} total val instances")

    with open(prod2idx_output_path, 'w') as f:
      f.write(json.dumps(prod2idx, ensure_ascii=False))

    with open(labels_output_path, 'w') as f:
      f.write(json.dumps(labels, ensure_ascii=False))

    with open(test_set_path, 'w') as f:
      f.write(json.dumps({'x': x_test, 'y': y_test}, ensure_ascii=False))


def convert_single_example(example: List[ClassifyInputExample],
                          max_seq_len: int,
                          prod2idx: Dict[str, int]) -> ClassifyInputFeatures:
  """
  Args:
    example:  raw data
    max_seq_len:  max sequence length
    prod2idx: product to index
  Return:
    result: train data
  """
  tokens_a = example.text_a

  # max_seq_len에 길이를 맞추기 위해 토큰 잘라내기
  if len(tokens_a) > max_seq_len - 2:
    tokens_a = tokens_a[:(max_seq_len - 2)]

  # Make tokens and segment_ids(=type_ids)
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)

  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  input_ids = [prod2idx[x] for x in tokens]
  input_mask = [1] * len(input_ids)

  # Padding
  while len(input_ids) < max_seq_len:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_len
  assert len(input_mask) == max_seq_len
  assert len(segment_ids) == max_seq_len

  # get label
  label_id = example.label

  result = ClassifyInputFeatures(input_ids,
                                input_mask,
                                segment_ids,
                                label_id)
  return result


def create_int_feature(values):

  if type(values) is list:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  else:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))


def create_float_feature(values):

  if type(values) is list:
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  else:
    return tf.train.Feature(float_list=tf.train.FloatList(value=[values]))


def create_bytes_feature(values):

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
