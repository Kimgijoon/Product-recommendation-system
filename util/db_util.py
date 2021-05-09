import re
import collections

from typing import List, Dict, Any, Tuple
from pymongo import MongoClient

from util.preprocess_product import PreprocessData


class MongoController(object):

  def __init__(self, username: str, passwd: str, server_ip: str):

    self.connection = MongoClient(f'mongodb://{username}:{passwd}@{server_ip}', 27017)
    self.db = self.connection['naver']

    self.prep = PreprocessData()

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
                              review_tb: str) -> List[Dict[str, str]]:
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
      user_id = i['user_id']
      if self.prep.prep_prod_name(i['category'], i['prod_name']) != None:
        prod_name = self.prep.prep_prod_name(i['category'], i['prod_name'])
      create_date = i['create_date']
      prep_user_id = re.split('\*{2,}', user_id.split('_')[1])[0].strip()
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
    return result

  def get_train_data(self, tb_name: str='review') -> Tuple[List[str], List[str]]:
    """학습을 위한 데이터를 가져오는 함수
    Args:
      tb_name:  collection name
    Returns:
      x:  sequence data e.g. [Xt-n, ... Xt-1]
      y:  Category(Xt)
      prod2idx: product to index
      labels: label information
    """
    dic_list = self.select_data(tb_name)
    data_dic = self._get_prod_by_user(dic_list)
    prod_category_dic = {self.prep.prep_prod_name(x['category'], x['prod_name']): x['category'] \
                          for x in dic_list}
    prod2idx = {x[1]: x[0] for x in enumerate(['[CLS]', '[SEP]']+list(prod_category_dic.keys()))}

    collection = self.db.collection_names(include_system_collections=False)
    labels = {x[1]: x[0] for x in enumerate(collection)}

    x, y = [], []
    for i in data_dic:
      # 중복이 발생해서 위에서 처리했더라고 상품 갯수가 3개 미만일 수 있어서
      if len(data_dic[i]) > 2:
        x.append(data_dic[i][:len(data_dic[i])-1])
        y.append(prod_category_dic[data_dic[i][-1]])

    return x, y, prod2idx, labels
