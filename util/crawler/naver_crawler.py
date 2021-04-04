import re
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
    for dic in data:
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

  def _parse_review_html(self, url: str, prod_name: str) -> List[Dict[str, Optional[str]]]:
    """sub url에서 댓글을 긁는 함수
    Args:
      url:  target url
    Returns:
      comment_list: list contains comment, user id, create date and score
    """
    res, _ = self._get_html(url)
    reviews = res.get('reviews', None)

    regex = re.compile(r'<[^>]+>')
    comment_list: List[Dict[str, Optional[str]]] = []
    for i in reviews:
      review = regex.sub('', i['content'])
      id = f"{i['mallName']}_{i['userId']}"
      date = i['registerDate']
      score = i['starScore']
      comment_list.append({
        'comment': review,
        'user_id': id,
        'create_date': date,
        'score': score,
        'prod_name': prod_name,
        'category': self.category_name
      })

    return comment_list
    
  def parse_sub_html(self):
    """sub url에서 전체 댓글을 긁는 함수
    Args:
      url:  target url
    Returns:
      comment_list: list contains multiple data
    """
    url_id: int = url.split('nvMid=')[1].split('&')[0]
    url: str = 'https://search.shopping.naver.com/api/review?nvMid=' + str(url_id) + \
              '&reviewType=ALL&sort=QUALITY&isNeedAggregation=N&isApplyFilter=N&page=1&pageSize=20'
    try:
      res, _ = self._get_html(url)
      total_count = res['totalCount']
    except Exception as e:
      print(f'err: {e}')
      return None

    if (total_count // 1000) == 0:
      max_count = (total_count // 20) + 1
    else:
      max_count = total_count // 20

    comment_list = []
    for page_num in range(1, max_count):
      review_url = 'https://search.shopping.naver.com/api/review?nvMid=' + str(url_id) + \
                  '&reviewType=ALL&sort=QUALITY&isNeedAggregation=N&isApplyFilter=N&page=' + \
                  str(page_num) + '&pageSize=1000'

      try:
        comments = self._parse_review_html(review_url, prod_name)
        comment_list.extend(comments)
      except:
        pass

    return comment_list
