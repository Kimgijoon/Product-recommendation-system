import re


class PreprocessProduct(object):

  def __init__(self):

    self.p1 = re.compile('\(+[가-힣a-zA-Z0-9-+ ]+\)|\[+[가-힣a-zA-Z0-9-+ ]+\]|\（+[가-힣a-zA-Z0-9+ .]+\)|[0-9]+GB|[0-9]+G|자급제|[0-9]{4,}')
    self.p2 = re.compile('[a-zA-Z0-9]+\-+[a-zA-Z0-9]+')

  def _prep_cellphone(self, prod_name):
    """휴대폰 제품명 정규식으로 전처리하는 함수
    Args:
      prod_name: 제품명
    Return:
      result: 전처리된 제품명
    """
    if re.match('갤럭시|아이|LG', prod_name):
      first_prep = re.sub('Pro|pro', '프로', self.p2.sub('', self.p1.sub('', prod_name)))
      second_prep = first_prep.replace('폴더', '폴드')
      third_prep = second_prep.replace('mini', '미니')
      fourth_prep = third_prep.replace('\u200b','')
      if re.match('LG', fourth_prep):
        result = fourth_prep.strip().replace('갤럭시폴드', '갤럭시Z폴드')
      else:
        result = re.sub(' ', '', fourth_prep).rstrip().replace('갤럭시폴드', '갤럭시Z폴드')

      if len(result) > 1:
        return result
      else:
        return None

    def _prep_tablet(self, prod_name):
      """테블릿 제품명 정규식으로 전처리하는 함수
    Args:
      prod_name:  제품명
    Return:
      result: 전처리된 제품명
    """
    if re.match('갤럭시|아이|서피스', prod_name):
      result = self.p2.sub('', self.p1.sub('', prod_name))
    else:
      result = None
    return result

  def _prep_earphone(self, prod_name):
    """이어폰 제품명 정규식으로 전처리하는 함수
    Args:
      prod_name:  제품명
    Return:
      result: 전처리된 제품명
    """
    if re.match('갤럭시|에어|모멘텀', prod_name):
      result = re.sub('\(+.+\)|케이스|([가-힣]+충전|[가-힣]+\s+충전)|모노|모델', '', prod_name).strip()
    else:
      result = None
    return result

  def prep_prod_name(self, category, prod_name):
    """제품명 곂치는게 너무 sparse해서 전처리하는 함수
    Args:
      category: 제품 카테고리명
      prod_name:  제품명
    Return:
      result: 전처리된 제품명
    """
    if category == 'cellphone':
      result = self._prep_cellphone(prod_name)
    elif category == 'tablet':
      result = self._prep_tablet(prod_name)
    elif category == 'earphone':
      result = self._prep_earphone(prod_name)
    else:
      result = prod_name
    return result
  