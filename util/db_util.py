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

  def _get_user_list(self, dic_list: List[Dict[str, Any]]) -> List[str]:
    """id에서 비식별화되지 않은 문자가 3개이상인 user id를 return하는 함수
    Args:
      dic_list: user id가 담겨있는 dict
    Return:
      suer_list:  selected user list
    """
    user_list: List = [x['user_id'] for x in dic_list]
    dic: Dict[str, int] = dict(collections.Counter(user_list))

    user_list: List[str] = []
    for i in dic:
      if (self.p.match(i)) and (dic[i] > 100):
        user_list.append(i)

    return user_list

  def get_review(self, tb_name: str) -> List[Dict[str, Any]]:
    """특정 조건에 맞는 리뷰를 return하는 함수
    Args:
      tb_name:  collection name
    Return:
      selected_review_list: review list for specific conditions
    """
    dic_list: List[Dict[str, Any]] = self.select_data(tb_name)
    user_list: List[Dict[str, Any]] = self._get_user_list(dic_list)

    selected_review_list: List[Dict[str, Any]] = []
    for idx, user_id in enumerate(user_list):
      if idx == 1: break
      res = [x for x in dic_list if x['user_id'] == user_id]
      selected_review_list.extend(res)

    print(selected_reivew_list[:5])