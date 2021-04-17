from collections import OrderedDict
from typing import Dict, Iterable, List, Optional


class Vocab:

  def __init__(self, vocab_path: str, pad_token: str='[PAD]', unk_token: str = '[UNK]'):
    """Vocab constructor
    Args:
      vocab_path: 로드할 vocab path
      pad_token: 사용할 padding token
      unk_token: 사용할 unknown token
    """
    self.__vocab: Dict[str, int] = self._load_vocab_file(vocab_path)
    self.__inv_vocab: Dict[int, str] = {v: k for k, v in self.__vocab.items()}

    self.pad_token = pad_token
    self.pad_token_id = self.__vocab[self.pad_token]
    self.unk_token = unk_token
    self.unk_token_id = self.__vocab[self.unk_token]

  def __contains__(self, key: str) -> bool:

    return key in self.__vocab

  def __len__(self) -> int:

    return len(self.__vocab.keys())

  def get_vocab(self) -> List[str]:

    return list(self.__vocab.keys())

  def convert_token_to_id(self, token: str, default: Optional[int]=None) -> int:
    """토큰 하나를 인덱스로 바꾸는 함수
    Args:
      token: 바꿀 토큰
      default: 만약에 해당 토큰이 vocab에 없을 경우 return할 default value
    Return:
      result: 인덱스로 변환된 토큰
    """
    if default:
      result: int = self.__vocab.get(token, default)
      return result

    result: int = self.__vocab[token]
    return result

  def convert_id_to_token(self, idx: int, default: Optional[str]=None) -> str:
    """인덱스 하나를 토큰으로 바꾸는 함수
    Args:
      idx: 바꿀 인덱스
      default: 만약 해당 인덱스가 vocab에 없을 경우 반환할 default value
    Return:
      result: 토큰으로 변환된 인덱스
    """
    if default:
      result: str = self.__inv_vocab.get(idx, default)
      return result

    result: str = self.__inv_vocab[idx]
    return result

  def convert_tokens_to_ids(self, tokens: Iterable[str]) -> List[int]:
    """토큰 여러개를 인덱스들로 바꾸는 함수
    Args:
      tokens: 바꿀 토큰들
    Return:
      result: 바뀐 인덱스들
    """
    result: List[int] = [self.__vocab.get(item, self.unk_token_id) for item in tokens]
    return result

  def convert_ids_to_tokens(self, ids: Iterable[int]) -> List[str]:
    """인덱스 여러개를 토큰들로 바꾸는 함수
    Args:
      ids: 바꿀 인덱스들
    Return:
      result: 바뀐 토큰들
    """
    result: List[str] = [self.__inv_vocab[item] for item in ids]
    return result

  def dump(self, vocab_path: str):

    with open(vocab, 'w') as f:
      f.write('\n'.join(self.__vocab.keys()))

  @staticmethod
  def _load_vocab_file(vocab_path: str) -> Dict[str, int]:

    vocab: Dict[str, int] = OrderedDict()
    with open(vocab_path, 'r') as f:
      for index, token in enumerate(f):
        token = token.strip().split('\t')[0]

        if token in vocab:
          raise ValueError('vocab에 중복된 토큰 {}이 있음'.format(token))

        vocab[token] = index

    return vocab
