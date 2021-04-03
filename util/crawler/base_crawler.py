from abc import *


class BaseCrawler(metaclass=ABCMeta):

	def __init_(self):

		pass

	@abstractmethod
	def _get_html(self):

		pass

	@abstractclassmethod
	def parse_main_html(self):

		pass

	@abstractclassmethod
	def parse_sub_html(self):

		pass