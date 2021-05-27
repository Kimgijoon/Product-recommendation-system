import os
import sys
import collections

import numpy as np
from typing import Union, List, Optional
import tensorflow.compat.v1 as tf


class TFHelper(object):

  def __init__(self, max_seq_length: int):

    self.max_seq_length = max_seq_length
    self.Inputs = collections.namedtuple("Inputs",
                                        ["input_ids",
                                        "input_mask",
                                        "segment_ids",
                                        "masked_lm_positions",
                                        "masked_lm_ids",
                                        "masked_lm_weights"])

  def make_session(self, target_gpu: int, gpu_usage: Optional[float]) -> tf.Session:
    """tf.Session을 return하는 함수
    Args:
      target_gpu: 점유하려는 GPU 번호
      gpu_usage:  점유할 portion
    Return:
      sess: config에 맞게 설정된 tf.session
    """
    if target_gpu is None:
      target_gpu = str(0)
    else:
      target_gpu = str(target_gpu)

    if gpu_usage == 1.0:
      config = tf.ConfigProto(allow_soft_placement=True)
    else:
      config = tf.ConfigProto(allow_soft_placement=True,
      gpu_options=tf.GPUOptions(visible_device_list=target_gpu,
                                per_process_gpu_memory_fraction=gpu_usage))
    sess = tf.Session(config=config)

    return sess

  def handle_init_checkpoint(ck_dir: str, ck_specifier: str) -> str:
    """init checkpoint path를 넘겨주는 함수
    Args:
      ck_specifier: 구분자
    Returns:
      ret:  init checkpoint path
    """
    ret = None
    if ck_specifier == 'ls':
      files = []
      for f in tf.io.gfile.listdir(ck_dir):
        if os.path.isfile(os.path.join(ck_dir, f)):
          only_name, ext = os.path.splittext(os.path.basename(f))
          if only_name not in files and '-' in only_name:
            files.append(only_name)

      files.sort(key=lambda x: int(x.split('-')[1]))
      print('[List of checkpoints]')
      for f in files:
        print(f)
      sys.exit(0)
    elif ck_specifier is None:
      pass
    else:
      init_ck_path = os.path.join(ck_dir, ck_specifier)
      if not tf.io.gfile.exists('{}.meta'.format(init_ck_path)):
        print('[ERROR] Checkpoint', init_ck_path, 'is not exist')
        sys.exit(0)
      ret = init_ck_path

    return ret

  def _pretrain_parser(self, tfrecord):
    """pretrain tfrecord 파일을 읽는 함수
    Args:
      tfrecord: tfrecord 파일
    Returns:
      example:  tfrecord에서 읽어온 데이터
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

  def _finetune_parser(self, tfrecord):
    """finetuning tfrecord 파일을 읽는 함수
    Args:
      tfrecord: tfrecord 파일
    Returns:
      example:  tfrecord에서 읽어온 데이터
    """
    label_size = 1

    name_to_features = {
      'input_ids': tf.FixedLenFeature([self.max_seq_length], tf.int64),
      'input_mask': tf.FixedLenFeature([self.max_seq_length], tf.int64),
      'segment_ids': tf.FixedLenFeature([self.max_seq_length], tf.int64),
      'label_id': tf.FixedLenFeature([label_size], tf.int64)
    }

    example = tf.io.parse_single_example(tfrecord, name_to_features)
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, dtype=tf.int32)
      example[name] = t

    return example

  def get_features_from_tfrecords(self,
                                  tfrecord_path: Union[List, str],
                                  batch_size: int,
                                  mode,
                                  is_training: bool):
    """tfrecord를 읽은 뒤 config에 맞게 데이터를 만드는 함수
    Args:
      tfrecord_path:  tfrecord가 위치한 디렉토리 경로
      batch_size: batch size
      is_training:  학습중인지에 대한 flag
    Returns:
      features: data
      tfrecord_iterator:  iterator
      tfrecord_iterator_initializer:  initializer
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

    if mode == 'pretrain':
      d = d.apply(tf.data.experimental.map_and_batch(lambda record: self._pretrain_parser(record),
                                                  batch_size=batch_size,
                                                  num_parallel_batches=4,
                                                  drop_remainder=True))
    else:
      d = d.apply(tf.data.experimental.map_and_batch(lambda record: self._finetune_parser(record),
                                                    batch_size=batch_size,
                                                    num_parallel_batches=4,
                                                    drop_remainder=True))

    tfrecord_iterator = d.make_initializable_iterator()
    tfrecord_iterator_initializer = tfrecord_iterator.make_initializer(d, name='tfrecord_iterator_initializer')

    features = tfrecord_iterator.get_next()

    return features, tfrecord_iterator, tfrecord_iterator_initializer

  def features_to_inputs(self, features):
    """모델 학습을 위해 추가적인 data를 가져오는 함수
    Args:
      features: data
    Return:
      result: 입력이 추가된 dict
    """
    result = self.Inputs(
        input_ids=features["input_ids"],
        input_mask=features["input_mask"],
        segment_ids=features["segment_ids"],
        masked_lm_positions=(features["masked_lm_positions"]
                             if "masked_lm_positions" in features else None),
        masked_lm_ids=(features["masked_lm_ids"]
                       if "masked_lm_ids" in features else None),
        masked_lm_weights=(features["masked_lm_weights"]
                           if "masked_lm_weights" in features else None),
    )
    return result

  def get_updated_inputs(self, inputs, **kwargs):
    """입력 데이터에 대해서 특정 feature가 바뀌면 입력 데이터를 update하는 함수"""
    features = inputs._asdict()
    for k, v in kwargs.items():
      features[k] = v
    return self.features_to_inputs(features)
