from abc import abstractmethod
from typing import List


class BaseTokenizer(object):
  """tokenizer meta class"""
  @abstractmethod
  def tokenize(self, text: str) -> List[str]:
    pass
