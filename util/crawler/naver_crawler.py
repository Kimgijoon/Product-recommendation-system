import time
import json
import requests

from typing import List, Dict, Tuple, Optional

from util.crawler.base_crawler import BaseCrawler


class NaverShoppingCrawler(BaseCrawler):

	def __init__(self, category_name: str):

		self.category_name = category_name

	def _get_html(self, url: str) -> Tuple[Dict[str, str], List[str]]:
		"""json 받아오는 함수, 429 error 발생 시 response 받아올 때까지 recursive call
    Args:
      url:  target url
    Returns:
      dic:  response data
    """
    r = requests.get(url, timeout=self.timeout)
    current_url: str = r.url
    dic: Dict[str, str] = json.loads(r.text)
    
    if r.status_code == 429:
      time.sleep(random.randint(10, 30))
      return self._get_html(url)

    return dic, current_url

  def parse_main_html(self, url: str) -> Tuple[List[Dict[str, Optional[str]]], List[str], List[str]]:
		"""메인 url로 부터 meta 및 sub url을 긁는 함수
    Args:
      url:  target url
    Returns:
      meta_list:  list contains meta data
      sub_url_list: list contains sub url
      date_list:  list contains creation date
    """
		res, _ = self._get_html(url)
    data = res['shoppingResult']['products']

    meta_list: List[Dict[str, Optional[str]]] = []
    sub_url_list: List[str] = [] 
    date_list: List[str] = []
    for idx, dic in enumerate(data):
      if dic['lowMallList'] == None:  continue
      else:
        sub_url = dic.get('crUrl', None)
        product_name = dic.get('productName', None)
        maker = dic.get('maker', None)
        low_price = dic.get('lowPrice', None)
        mouth_review_count = dic.get('mouthReviewCount', None)
        avg_score = dic.get('scoreInfo', None)
        product_info = dic.get('characterValue', None).split('|')
        open_date = dic['openDate']

        if avg_score == '': continue

        meta_list.append({
          'name': product_name,
          'low_price': low_price,
          'avg_score': avg_score,
          'review_count': mouth_review_count,
          'maker': maker,
          'product_info': product_info,
          'open_data':  open_date
        })
        sub_url_list.append(sub_url)
        date_list.append(int(open_date[:4]))

    return meta_list, sub_url_list, date_list

	def parse_sub_html(self):

		pass