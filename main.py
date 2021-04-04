import json
import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('op', None, '[REQUIRED] Operation code to do')
flags.mark_flag_as_required('op')

tf.flags.DEFINE_string('category', None, 'Name of category')
tf.flags.DEFINE_string('config_path', 'configs', 'directory of config file')
tf.flags.DEFINE_string('config_file', 'category.json', 'config file name')


def main():

	if FLAGS.Op == 'crawler':
		from util.crawler.naver_crawler import NaverShoppingCrawler

		category_dic_path = f'{FLAGS.config_path}/{FLAGS.config_file}'
		with open(category_dic_path, 'r') as f:
			category_dic = json.loads(f.read())

		category_name = FLAGS.category
		category_id = category_dic.get(category_name, None)

		crawler = NaverShoppingCrawler(category_name, 60)

		for page_num in range(1, 50):
			url = 'https://search.shopping.naver.com/api/search/category?sort=rel&pagingIndex={}&pagingSize=40&viewType=list&productSet=total&catId={}&deliveryFee=&deliveryTypeValue=&iq=&eq=&xq=&frm=NVSHTTL&window='
			url = url.format(page_num, category_id)
			meta_list, sub_url_list, date_list = crawler.parse_main_html(url)
			if len(date_list) == 0: break
			for idx in range(len(sub_url_list)):
				comment_list = crawler.parse_sub_html(sub_url_list[idx], meta_list[idx]['name'])
				if (len(comment_list) == 0) or (comment_list == None):
					continue
				time.sleep(random.randint(10, 30))


if __name__ == '__main__':

	main()