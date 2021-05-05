import re

from typing import List, Dict, Any
from pymongo import MongoClient


class MongoController(object):

  def __init__(self, username: str, passwd: str):

    self.connection = MongoClient('mongodb://{}:{}@localhost'.format(username, passwd), 27017)
    self.db = self.connection['naver']

  def insert_data(self, data: List, tb_name: str):
    """db>collection에 document insert하는 함수
    Args:
      data: input data
      tb_name:  collection name
    """
    db_collection = self.db[tb_name]
    db_collection.insert_many(data)

  def select_data(self, tb_name: str, query: Dict[str, str]=None) -> Dict[str, str]:
    """db>collection에서 document를 select하는 함수
    Args:
      tb_name:  collection name
      query:  such as where i.e. select * from where 
    Return:
      result: selected output
    """
    db_collection = self.db[tb_name]

    result = []
    if query != None:
      for dic in db_collection.find():  result.append(dic)
    else:
      for dic in db_collection.find(query): result.append(dic)
    return result

  def get_prod_name_and_review(self,
                              prod_tb_list: List[str],
                              review_tb: str) -> Dict[List[str, str]]:
    """제품명과 제품명에 해당하는 리뷰, 점수를 리턴하는 함수
    Args:
      prod_tb_list: 상품 카테고리 컬렉션 리스트
      review_tb:  리뷰 컬렉션 이름
    Return:
      result: 제품명, 리뷰, 점수를 담은 딕셔너리
    """
    prod_list = []
    for prod_tb in prod_tb_list:
      prod_list.extend(self.select_data(prod_tb))  
    user_list = self.select_data(review_tb)

    result = []
    for prod in prod_list:
      prod_name = prod['name']
      for user in user_list:
        if prod_name == user['prod_name']:
          result.append({
            'name': prod_name,
            'comment': user['comment'],
            'score': user['score']})

    return result

  def _get_prod_by_user(self, dic_list: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """user id가 쇼핑몰명_id+*처리 되있음.
    그래서 전처리를 수행한 후, 동일 유저로 보고 3개 이상 상품을 구매한 유저 닉네임과 유저의 상품 리스트를 return하는 함수
    Args:
      dic_list: user id가 담겨있는 dict
    Return:
      result:  3개 이상 상품을 구매한 유저 닉네임과 유저의 상품 리스트를 담은 딕셔너리
    """
    multiple_list = []
    for i in dic_list:
      user_id, prod_name = i['user_id'], i['prod_name']
      create_date = i['create_date']
      prep_user_id = re.split('\*{2,}', user_id.split('_')[1])[0].rstrip()
      if len(prep_user_id) == 0: prep_user_id = 'Anonymous'
      multiple_list.append([prep_user_id, prod_name, create_date])

    user_occurence_dic = dict(collections.Counter([x[0] for x in multiple_list]))
    result = {}
    for user_id, prod_name, create_date in sorted(multiple_list, key=lambda x: (x[0], x[2])):
      if user_occurence_dic[user_id] > 2:
        if user_id not in result:
          result[user_id] = [prod_name]
        else:
          result[user_id].append(prod_name)

    result = {x: sorted(set(result[x]), key=result[x].index) for x in result}
    return selected_user_list

  def get_review(self, tb_name: str) -> List[Dict[str, Any]]:
    """특정 조건에 맞는 리뷰를 return하는 함수
    Args:
      tb_name:  collection name
    Return:
      selected_review_list: review list for specific conditions
    """
    dic_list: List[Dict[str, Any]] = self.select_data(tb_name)
    user_list: List[Dict[str, Any]] = self._get_prod_by_user(dic_list)

    selected_review_list: List[Dict[str, Any]] = []
    for idx, user_id in enumerate(user_list):
      if idx == 1: break
      res = [x for x in dic_list if x['user_id'] == user_id]
      selected_review_list.extend(res)

    print(selected_reivew_list[:5])