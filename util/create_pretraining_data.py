import sys
import random
import traceback
import collections
import concurrent.futures
import six
from six.moves import range

import tensorflow as tf

from util.tokenizer.vocab import Vocab
from util.tokenizer.mecab_tokenizer import MeCabTokenizer
from util.tokenizer.sp_tokenizer import SentencePieceTokenizer
from util.tokenizer.mecab_sp_tokenizer import MeCabSentencePieceTokenizer


flags = tf.flags
FLAGS = flags.FLAGS


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids):
    self.tokens = tokens
    self.segment_ids = segment_ids

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
      [printable_text(x) for x in self.tokens]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def printable_text(text):
  """Returns text encoded in a way suitable for print or `tf.logging`."""

  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return six.ensure_text(text, "utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text
    elif isinstance(text, six.text_type):
      return six.ensure_binary(text, "utf-8")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return six.ensure_text(text, "utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return six.ensure_text(text, "utf-8", "ignore")
    elif isinstance(text, six.text_type):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def write_instance_to_example_files(instances,
                                    vocab,
                                    max_seq_length,
                                    max_predictions_per_seq,
                                    output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    input_ids = vocab.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if inst_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: %s" % " ".join(
        [printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
          "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):

  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):

  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def create_training_instances(input_files,
                              tokenizer,
                              max_seq_length,
                              dupe_factor,
                              short_seq_prob,
                              masked_lm_prob,
                              max_predictions_per_seq,
                              rng):
  """Create `TrainingInstance`s from raw text."""
  all_documents = [[]]

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  for input_file in input_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        line = convert_to_unicode(reader.readline())
        if not line:
          break
        line = line.strip()

        # Empty lines are used as document delimiters
        tokens = tokenizer.tokenize(line)
        if tokens:
          all_documents[-1].append(tokens)

  # Remove empty documents
  all_documents = [x for x in all_documents if x]
  rng.shuffle(all_documents)

  vocab_words = None
  instances = []
  for _ in range(dupe_factor):
    for document_index in range(len(all_documents)):
      instances.extend(
        create_instances_from_document(
          all_documents, document_index, max_seq_length, short_seq_prob,
          masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

  rng.shuffle(instances)
  return instances


def create_instances_from_document(all_documents,
                                  document_index,
                                  max_seq_length,
                                  short_seq_prob,
                                  masked_lm_prob,
                                  max_predictions_per_seq,
                                  vocab_words,
                                  rng):                                 
  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[document_index]

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  target_seq_length = max_num_tokens
  if rng.random() < short_seq_prob:
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  instances = []
  current_chunk = []
  current_length = 0
  i = 0
  while i < len(document):
    segment = document[i]
    current_chunk.append(segment)
    current_length += len(segment)
    if i == len(document) - 1 or current_length >= target_seq_length:
      if current_chunk:
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        a_end = 1
        if len(current_chunk) >= 2:
          a_end = rng.randint(1, len(current_chunk) - 1)

        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])

        tokens_b = []

        for j in range(a_end, len(current_chunk)):
          tokens_b.extend(current_chunk[j])
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

        assert len(tokens_a) >= 1

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        instance = TrainingInstance(
          tokens=tokens,
          segment_ids=segment_ids)
        instances.append(instance)
      current_chunk = []
      current_length = 0
    i += 1

  return instances


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def parallel_fn(input_file,
                output_file,
                max_seq_length,
                dupe_factor,
                short_seq_prob,
                masked_lm_prob,
                max_predictions_per_seq,
                rng):

  try:
    mecab = MeCabTokenizer(FLAGS.mecab_file)
    sp = SentencePieceTokenizer(FLAGS.spm_model)
    tokenizer = MeCabSentencePieceTokenizer(mecab, sp)
    vocab = Vocab(FLAGS.vocab_file)

    instances = create_training_instances([input_file],
                                          tokenizer,
                                          max_seq_length,
                                          dupe_factor,
                                          short_seq_prob,
                                          masked_lm_prob,
                                          max_predictions_per_seq,
                                          rng)
    write_instance_to_example_files(instances,
                                    vocab,
                                    max_seq_length,
                                    max_predictions_per_seq,
                                    [output_file])
  except Exception as e:
    tf.logging.info(f'Exception occured {e}')
    tf.logging.info(traceback.format_exc())


def run():

  tf.logging.set_verbosity(tf.logging.INFO)

  mecab = MeCabTokenizer(FLAGS.mecab_file)
  sp = SentencePieceTokenizer(FLAGS.spm_model)
  tokenizer = MeCabSentencePieceTokenizer(mecab, sp)
  vocab = Vocab(FLAGS.vocab_file)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info(f"  {input_file}")

  rng = random.Random(FLAGS.random_seed)

  if not FLAGS.output_file:
    tf.logging.info("no outputfiles were passed, using input names as output")
    output_files = [f"{in_file}.tfrecord" for in_file in FLAGS.input_file.split(",") if in_file]
  else:
    output_files = FLAGS.output_file.split(",")

  if not FLAGS.parallel:
    instances = create_training_instances(input_files,
                                          tokenizer,
                                          FLAGS.max_seq_length,
                                          FLAGS.dupe_factor,
                                          FLAGS.short_seq_prob,
                                          FLAGS.masked_lm_prob,
                                          FLAGS.max_predictions_per_seq,
                                          rng)

    tf.logging.info("number of instances: %i", len(instances))
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
      tf.logging.info(f"  {output_file}")

    write_instance_to_example_files(instances,
                                    vocab,
                                    FLAGS.max_seq_length,
                                    FLAGS.max_predictions_per_seq,
                                    output_files)
  else:
    if len(input_files) == len(output_files):
      n_files = len(input_files)
      with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(parallel_fn,
                    input_files,
                    output_files,
                    [FLAGS.max_seq_length] * n_files,
                    [FLAGS.dupe_factor] * n_files,
                    [FLAGS.short_seq_prob] * n_files,
                    [FLAGS.masked_lm_prob] * n_files,
                    [FLAGS.max_predictions_per_seq] * n_files,
                    [rng] * n_files)