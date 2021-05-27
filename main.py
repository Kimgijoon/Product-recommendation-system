import json
import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


# main operation argument
flags.DEFINE_string('op', None, '[REQUIRED] Operation code to do')
flags.mark_flag_as_required('op')

# create pretrain data 
flags.DEFINE_string("input_file", None, "Input raw text file (or comma-separated list of files).")
flags.DEFINE_string("output_file", None, "Output TF example file (or comma-separated list of files).")
flags.DEFINE_string("vocab_file", None, "The vocabulary file that the electra-albert model was trained on.")
flags.DEFINE_string("spm_model", None, "sentencepiece model file")
flags.DEFINE_string("mecab_file", None, "mecab file")
flags.DEFINE_bool("parallel",
          False,
          "Option to use multiprocess to speed up make tfrecord wokring with multiple raw files."
          "output files will be written next to input files if non are passed.")

flags.DEFINE_bool("do_lower_case",
                  True,
                  "Whether to lower case the input text. Should be True for uncased "
                  "models and False for cased models.")
flags.DEFINE_bool("do_whole_word_mask",
                  False,
                  "Whether to use whole word masking rather than per-WordPiece masking.")
flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")
flags.DEFINE_integer("max_predictions_per_seq", 20, "Maximum number of masked LM predictions per sequence.")
flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")
flags.DEFINE_integer("dupe_factor", 1, "Number of times to duplicate the input data (with different masks).")
flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")
flags.DEFINE_float("short_seq_prob",
                  0.1,
                  "Probability of creating sequences which are shorter than the maximum length.")

# create finetuning data
flags.DEFINE_string("classify_tfrecord_filename", "classify.tfrecord", "train data name")
flags.DEFINE_string("classify_val_tfrecord_filename", "classify_val.tfrecord", "validation data name")
flags.DEFINE_string("classify_json_filename", "classify.json", "train data json format")
flags.DEFINE_string("prod2idx_filename", "prod2idx.json", "product to index file")
flags.DEFINE_string("label_filename", "label.json", "label file")
flags.DEFINE_string("test_filename", "test_set.json", "test set file")
flags.DEFINE_string("split_ratio", '0.6, 0.3, 0.1', "data split ratio")

# pretrain
flags.DEFINE_string("config_file", None, "The config json specifies the model architecture.")
flags.DEFINE_string('data_home_dir', None, "Path to input directory.")
flags.DEFINE_string('checkpoint_dir', None, '')
flags.DEFINE_string('tokenizer_dir', None, '')
flags.DEFINE_integer('gpu_num', None, 'target GPU number')
flags.DEFINE_float('gpu_usage', None, 'use of GPU process memory limit')

# crawler
tf.flags.DEFINE_string('category', None, 'Name of category')
tf.flags.DEFINE_string('config_path', 'configs', 'directory of config file')
tf.flags.DEFINE_string('config_file', 'category.json', 'config file name')
tf.flags.DEFINE_string('server_ip', None, 'mongodb server address')
tf.flags.DEFINE_string('id', None, 'mongodb id')
tf.flags.DEFINE_string('passwd', None, 'mongodb password')


def main(_):

  if FLAGS.op == 'create_pretrain':
    import util.create_pretraining_data as cp
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    cp.run()

  elif FLAGS.op == 'pretrain':
    from src.pretrain import PretrainModel
    model = PretrainModel(FLAGS.config_file,
                          FLAGS.tokenizer_dir,
                          FLAGS.data_home_dir,
                          FLAGS.op,
                          FLAGS.checkpoint_dir,
                          FLAGS.gpu_num,
                          FLAGS.gpu_usage,
                          is_training=True)

  elif FLAGS.op == 'create_finetune':
    from util.create_finetuning_data import CreateClassifyData
    cc = CreateClassifyData()
    cc.run()

  elif FLAGS.op == 'finetune':
    pass

  elif FLAGS.op == 'crawler':
    from util.db_util import MongoController
    from util.crawler.naver_crawler import NaverShoppingCrawler

    category_dic_path = f'{FLAGS.config_path}/{FLAGS.config_file}'
    with open(category_dic_path, 'r') as f:
      category_dic = json.loads(f.read())

    category_name = FLAGS.category
    category_id = category_dic.get(category_name, None)

    crawler = NaverShoppingCrawler(category_name, 60)
    db = MongoController(FLAGS.id, FLAGS.passwd)

    for page_num in range(1, 50):
      url = 'https://search.shopping.naver.com/api/search/category?sort=rel&pagingIndex={}&pagingSize=40&viewType=list&productSet=total&catId={}&deliveryFee=&deliveryTypeValue=&iq=&eq=&xq=&frm=NVSHTTL&window='
      url = url.format(page_num, category_id)
      meta_list, sub_url_list, date_list = crawler.parse_main_html(url)
      if len(date_list) == 0: break
      for idx in range(len(sub_url_list)):
        comment_list = crawler.parse_sub_html(sub_url_list[idx], meta_list[idx]['name'])
        if (len(comment_list) == 0) or (comment_list == None):
          continue
        db.insert_data(comment_list, 'review')
        time.sleep(random.randint(10, 30))
      db.insert_data(meta_list, FLAGS.category)


if __name__ == '__main__':

  tf.compat.v1.app.run()
