import json

import tensorflow.compat.v1 as tf
from typing import List, Dict, Tuple, Union, Any

import src.optimization as optimization
import src.electra_albert as modeling
from util.tf_util import TFHelper


class PretrainModel(object):

  def __init__(self,
              config_file: str,
              data_home_dir: str,
              workspace: str,
              is_training: bool):

    with open(config_file, 'r') as f:
      config = json.loads(f.read())

    self.data_home_dir = data_home_dir
    self.workspace = workspace
    self.is_training = is_training

    self._build_model()

  def _build_model(self):

    data_dir = os.path.join(self.data_home_dir, self.workspace)
    tfrecord_file_path = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if x.endswith('.tfrecord')]
    features, _, input_initializer = self.util.get_features_from_tfrecords(tfrecord_file_path, self.config.batch_size, self.is_training)
    self.input_initializer = input_initializer

    self.input_ids = input_ids = tf.placeholder_with_default(features['input_ids'],
                                                            shape=modeling.get_shape_list(features['input_ids']),
                                                            name='input_ids')
    self.input_mask = input_mask = tf.placeholder_with_default(features['input_mask'],
                                                            shape=modeling.get_shape_list(features['input_mask']),
                                                            name='input_mask')
    self.segment_ids = segment_ids = tf.placeholder_with_default(features['segment_ids'],
                                                            shape=modeling.get_shape_list(features['segment_ids']),
                                                            name='segment_ids')

  def fit(self):

    pass
