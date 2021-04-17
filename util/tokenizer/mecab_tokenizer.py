import os
import json
from typing import List

import MeCab

from util.tokenizer.base_tokenizer import BaseTokenizer


class MeCabTokenizer(BaseTokenizer):

  def __init__(self, config_path: str):

    self.mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ko-dic')
    with open(config_path) as f:
      self.config: dict = json.load(f)

  def tokenize(self, text: str) -> List[str]:

    text = text.strip()
    text_ptr = 0
    tokenized = []
    parsed_text_by_mecab = self.mecab.parse(text).split('\n')
    for morph in parsed_text_by_mecab:
      if '\t' in morph:
        splitted = morph.split('\t')
        token = splitted[0]

        if text[text_ptr] == ' ':
          while text[text_ptr] == ' ':
            text_ptr += 1
          assert(text[text_ptr] == token[0])

          tokenized.append(self.config['space_symbol'])

        tokenized.append(token)
        text_ptr += len(token)

    return tokenized

  def restore(self, tokens: List[str]) -> str:

    text = ''.join(tokens).replace("▃", " ").strip()
    return text
