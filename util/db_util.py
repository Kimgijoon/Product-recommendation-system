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
