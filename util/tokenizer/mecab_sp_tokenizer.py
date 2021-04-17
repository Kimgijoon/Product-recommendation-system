from typing import List

from util.tokenizer.base_tokenizer import BaseTokenizer
from util.tokenizer.mecab_tokenizer import MeCabTokenizer
from util.tokenizer.sp_tokenizer import SentencePieceTokenizer


class MeCabSentencePieceTokenizer(BaseTokenizer):

  def __init__(self, mecab: MeCabTokenizer, sp: SentencePieceTokenizer):

    self.mecab = mecab
    self.sp = sp

  def tokenize(self, text: str) -> List[str]:

    tokenized = self.mecab.tokenize(text)
    tokenized = self.sp.tokenize(' '.join(tokenized))

    output = []
    for i in range(0, len(tokenized)):
      if i + 1 < len(tokenized) and (tokenized[i] == "▁" and tokenized[i + 1] == "▃"):
        continue
      if tokenized[i] == "▁▃":
        tokenized[i] = "▃"
      output.append(tokenized[i])

    return output

  def restore(self, tokens: List[str]) -> str:

    text = "".join(tokens).replace("▁", "").replace(" ", "").replace("▃", " ").strip()
    return text
