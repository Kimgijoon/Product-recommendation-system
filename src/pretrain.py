import os
import json
from collections import namedtuple

import logging
from rich.logging import RichHandler
import tensorflow.compat.v1 as tf
from typing import List, Dict, Tuple, Union, Any

import src.optimization as optimization
import src.electra_albert as modeling
from util.pretrain_util import PretrainHelper
from util.tf_util import TFHelper
from util.tokenizer.vocab import Vocab
from util.tokenizer.mecab_tokenizer import MeCabTokenizer
from util.tokenizer.sp_tokenizer import SentencePieceTokenizer
from util.tokenizer.mecab_sp_tokenizer import MeCabSentencePieceTokenizer


class PretrainModel(object):

  def __init__(self,
              config_file: str,
              data_home_dir: str,
              workspace: str,
              tokenizer_dir: str,
              checkpoint_dir: str,
              gpu_num: int,
              gpu_usage: float,
              is_training: bool):

    # logger
    FORMAT = "%(message)s"
    logging.basicConfig(level="NOTSET",
                        format=FORMAT,
                        datefmt="[%X]",
                        handlers=[RichHandler()])
    self.logger = logging.getLogger("rich")
    
    # load config file
    with open(config_file, 'r') as f:
      config = json.loads(f.read())

    self.config = namedtuple('config', config.keys())(*config.values())
    self.data_home_dir = data_home_dir
    self.workspace = workspace
    self.tokenizer_dir = tokenizer_dir
    self.checkpoint_dir = checkpoint_dir
    self.is_training = is_training

    # util function
    self.util = TFHelper(self.config.max_seq_length)
    self.pretrain = PretrainHelper(self.tokenizer_dir, self.config)

    # tf
    self.sess = self.util.make_session(gpu_num, gpu_usage)
    self._build_model()

  def get_loss_and_acc(self, dic: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """mlm loss, acc와 discriminator loss, acc를 계산하는 함수
    Args:
      dic:  모델 평가를 위한 항목들을 담은 dict
    Returns:
      result: loss and acc
    """
    _, masked_lm_acc = tf.metrics.accuracy(labels=tf.reshape(dic['masked_lm_ids'], [-1]),
                                          predictions=tf.reshape(dic['masked_lm_preds'], [-1]),
                                          weights=tf.reshape(dic['masked_lm_weights'], [-1]))
    _, masked_lm_loss = tf.metrics.mean(values=tf.reshape(dic['mlm_loss'], [-1]),
                                        weights=tf.reshape(dic['masked_lm_weights'], [-1]))
    _, sampled_masked_lm_acc = tf.metrics.accuracy(labels=tf.reshape(dic['masked_lm_ids'], [-1]),
                                                  predictions=tf.reshape(dic['sampled_tokids'], [-1]),
                                                  weights=tf.reshape(dic['masked_lm_weights'], [-1]))
    _, disc_acc = tf.metrics.accuracy(labels=dic['disc_labels'],
                                      predictions=dic['disc_preds'],
                                      weights=dic['input_mask'])
    _, disc_loss = tf.metrics.mean(dic['disc_loss'])

    result = {
      'masked_lm_acc': masked_lm_acc,
      'masked_lm_loss': masked_lm_loss,
      'sampled_masked_lm_acc': sampled_masked_lm_acc,
      'disc_acc': disc_acc,
      'disc_loss': disc_loss
    }
    return result

  def _build_model(self):
    """정의된 모델을 compile하는 함수"""
    data_dir = os.path.join(self.data_home_dir, self.workspace)
    tfrecord_file_path = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if x.endswith('.tfrecord')]
    features, _, input_initializer = self.util.get_features_from_tfrecords(tfrecord_file_path, self.config.batch_size, 'pretrain', self.is_training)
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
    # Mask the input
    masked_inputs = self.pretrain.mask(self.util.features_to_inputs(features), self.config.mask_prob)

    # Generator
    generator = modeling.Generator(config=self.config,
                                  is_training=self.is_training,
                                  input_ids=masked_inputs.input_ids,
                                  input_mask=masked_inputs.input_mask,
                                  token_type_ids=masked_inputs.segment_ids,
                                  use_one_hot_embeddings=False)
    mlm_output = self.pretrain.get_masked_lm_output(masked_inputs, generator)
    fake_data = self.pretrain.get_fake_data(masked_inputs, mlm_output.logits)
    self.mlm_output = mlm_output
    self.total_loss = self.config.gen_weight * mlm_output.loss

    # Discriminator
    discriminator = modeling.Discriminator(config=self.config,
                                            is_training=self.is_training,
                                            input_ids=fake_data.inputs.input_ids,
                                            input_mask=fake_data.inputs.input_mask,
                                            token_type_ids=fake_data.inputs.segment_ids,
                                            use_one_hot_embeddings=False)
    disc_output = self.pretrain.get_discriminator_output(fake_data.inputs, discriminator, fake_data.is_fake_tokens)
    self.total_loss += self.config.disc_weight * disc_output.loss
    evaluate_dic = {
      'input_ids': masked_inputs.input_ids,
      'masked_lm_preds': mlm_output.preds,
      'mlm_loss': mlm_output.per_example_loss,
      'masked_lm_ids': masked_inputs.masked_lm_ids,
      'masked_lm_weights': masked_inputs.masked_lm_weights,
      'input_mask': masked_inputs.input_mask,
      'disc_loss': disc_output.per_example_loss,
      'disc_labels': disc_output.labels,
      'disc_probs': disc_output.probs,
      'disc_preds': disc_output.preds,
      'sampled_tokids': tf.argmax(fake_data.sampled_tokens, -1, output_type=tf.int32)
    }
    self.metric_dic = self.get_loss_and_acc(evaluate_dic)

    # tfsummary
    self.summary_total_loss = tf.summary.scalar('total_loss', self.total_loss)
    self.summary_masked_lm_acc = tf.summary.scalar('masked_lm_acc', self.metric_dic['masked_lm_acc'])
    self.summary_masked_lm_loss = tf.summary.scalar('masked_lm_loss', self.metric_dic['masked_lm_loss'])
    self.summary_sampled_masked_lm_acc = tf.summary.scalar('sampled_masked_lm_acc', self.metric_dic['sampled_masked_lm_acc'])
    self.summary_disc_acc = tf.summary.scalar('disc_acc', self.metric_dic['disc_acc'])
    self.summary_disc_loss = tf.summary.scalar('disc_loss', self.metric_dic['disc_loss'])
    self.merged_summary = tf.summary.merge([self.summary_total_loss,
                                            self.summary_masked_lm_acc,
                                            self.summary_masked_lm_loss,
                                            self.summary_sampled_masked_lm_acc,
                                            self.summary_disc_acc,
                                            self.summary_disc_loss])

  def fit(self):
    """pretrain model을 학습하는 함수"""
    global_step = tf.train.get_or_create_global_step()
    train_op = optimization.create_adam_optimizer(loss=self.total_loss,
                                                  init_lr=self.config.learning_rate,
                                                  total_num_train_steps=self.config.num_train_steps,
                                                  num_warmup_steps=self.config.num_warmup_steps,
                                                  use_tpu=False,
                                                  weight_decay=self.config.weight_decay_rate)

    # make checkpoint dir
    checkpoint_dir = os.path.join(self.checkpoint_dir, self.workspace)
    epoch_ck_path = os.path.join(self.checkpoint_dir, 'model')
    summary_path = os.path.join(self.checkpoint_dir, 'summaries')
    tf.io.gfile.makedirs(checkpoint_dir)

    # load ckpt saver
    max_to_keep = max(self.config.num_epochs, (self.config.num_train_steps // 1000) + 1)
    train_summary_writer = tf.summary.FileWriter(summary_path, self.sess.graph)
    epoch_saver = tf.train.Saver(tf.global_variables(), max_to_keep = max_to_keep)
    self.logger.info(f'Saver max_to_keep: {max_to_keep}')

    # initialize model
    self.sess.run(tf.local_variables_initializer())
    self.sess.run(tf.global_variables_initializer())
    self.logger.info("Initialize models")

    _step = 0
    for cur_epoch in range(self.config.num_train_steps):
      self.logger.info(f"New epoch start : [{cur_epoch}]")

      self.sess.run(self.input_initializer)
      self.logger.info("Dataset is initialized with train tfrecord")

      while True:
        if _step >= self.config.num_train_steps:
          break

        try:
          (_train, _step, _merged, _loss, _masked_lm_acc, _masked_lm_loss,
          _sampled_masked_lm_acc, _disc_acc, _disc_loss) = self.sess.run([train_op,
                                                                          global_step,
                                                                          self.merged_summary,
                                                                          self.total_loss,
                                                                          self.metric_dic['masked_lm_acc'],
                                                                          self.metric_dic['masked_lm_loss'],
                                                                          self.metric_dic['sampled_masked_lm_acc'],
                                                                          self.metric_dic['disc_acc'],
                                                                          self.metric_dic['disc_loss']])

          self.logger.info(f"[{_step}], loss: {_loss}, lm_acc: {_masked_lm_acc}, \
                            lm_loss: {_masked_lm_loss}, sampled_acc: {_sampled_masked_lm_acc}, \
                            disc_acc: {_disc_acc}, disc_loss: {_disc_loss}")

          train_summary_writer.add_summary(_merged, _step)
          if (_step % self.config.save_period_steps) == 0:
            epoch_saver.save(self.sess, epoch_ck_path, global_step = _step)

        except tf.errors.OutOfRangeError:
          break
