import re
from typing import List

import numpy as np
import pandas as pd
from konlpy.tag import Okt
from sklearn.base import BaseEstimator, TransformerMixin
from functools import partial

__all__ = ['ColumnMerger', 'WordUnifier',
           'RegExReplacer', 'DuplicateRemover', 'StopWordRemover',
           'WordLower', 'MorphTokenizer', 'NounTokenizer', 'PosTokenizer',
           "CategoryOneHotEncoder"]

"""
## Transform 코드 구성

### 자연어 처리 부분 Transform
    - string을 받아 string을 내뱉는 것을 원칙으로 한다. ( 순서에 무관할 수 있도록 )
    - 멱등성을 가져야 한다. (동일 input, 동일 output)
    - 모든 input과 output은 pandas DataFrame형식을 유지한다. (except Vectorize)

    1. Basic Preprocssing
        기본적인 문장에 대한 전처리 코드가 들어가는 부분

    2. Tokenizer
        형태소 분석 부분. tokenizer 엔진으로서는 konlpy에서의 Okt(이전 Twitter)을 적용한다.
        reference : https://konlpy-ko.readthedocs.io/ko/v0.4.3/api/konlpy.tag/

    3. Vectorizer
        word를 벡터화시키는 부분

        참고 자료들
        1) BOW(Bag of Words) 방법론
        reference : https://datascienceschool.net/view-notebook/3e7aadbf88ed4f0d87a76f9ddc925d69/

        2) 임베딩 방법론
        reference : https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/11/embedding/


    4. LabelEncoder
        - CategoryOneHotEncoder
            카테고리 라벨 정보를 one-hot을 encoding함

        - CategoryMergeEncoder
            카테고리 라벨 정보를 합쳐서 encoding함
"""


############################
# 1. Basic NLP Preprocssing
#    - ColumnMerger
#    - WordUnifier
#
#    - DuplicateRemover
#    - StopWordRemover
#    - RegExReplacer
#
#    - WordLower
############################


class ColumnMerger(BaseEstimator, TransformerMixin):
    """
    주어진 데이터프레임에서 컬럼에 해당하는 string을 합침

    Example

    >>> df = pd.DataFrame(data={ "과일" : ['사과','배','딸기'],"시장" : ['명동','상정','죽도']})
    >>> cs = ColumnMerger(inputs=['과일','시장'],outputs='조합')
    >>> cs.transform(df)
       과일  시장     조합
    0  사과  명동  사과 명동
    1   배  상정   배 상정
    2  딸기  죽도  딸기 죽도
    """

    def __init__(self, inputs=[], outputs=None):
        """
        주어진 데이터프레임에서 컬럼에 해당하는 string을 합침

        :param inputs: 적용되는 column Name list
        :param outputs: 저장되는 column 이름, if None, inputs에 대체
        """
        self._inputs, self._outputs = verify_inputs_and_outputs(inputs, outputs)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X[self._inputs].apply(lambda x: " ".join(x), axis=1)
        for col_name in self._outputs:
            X[col_name] = result
        return X


class WordUnifier(BaseEstimator, TransformerMixin):
    """
    동일의미 다른 표기 통일

    Example

    >>> sample = pd.DataFrame(data={"이름":['삼성전자 노트북', "노트북 삼성", "samsung 스마트폰", 'lg 폰', "엘지전자 상거래"]})
    >>> wu = WordUnifier(inputs='이름', words_list=[["삼성","삼성전자",'samsung'], ["엘지",'엘지전자','lg']])
    >>> wu.transform(sample)
            이름
    0   삼성 노트북
    1   노트북 삼성
    2  삼성 스마트폰
    3     엘지 폰
    4   엘지 상거래

    """

    def __init__(self, inputs=[], outputs=None, words_list=[]):
        """
        주어진 데이터프레임에서 컬럼에 해당하는 string을 합침

        :param inputs:
        :param outputs:
        :param words_list:
        """
        self._inputs, self._outputs = verify_inputs_and_outputs(inputs, outputs)

        self._regex_list = []
        for words in words_list:
            unified = r'\b{}\b'.format(r'\b|\b'.join(map(re.escape, words[1:])))
            unify_regex = re.compile(unified)
            self._regex_list.append((words[0], unify_regex))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self._outputs] = X[self._inputs].applymap(self._transform)
        return X

    def _transform(self, phrase):
        for word, regex in self._regex_list:
            phrase = regex.sub(word, phrase)
        return phrase


class RegExReplacer(BaseEstimator, TransformerMixin):
    """
    정규식을 활용한 word 치환
    주어진 정규식에 만족하는 word에 대해서, 특정 word로 변경하는 코드

    Example

    >>>
    >>>
    >>>
    """

    def __init__(self, inputs=[], outputs=None, regex_list=[]):
        """

        :param inputs:
        :param outputs:
        :param regex_list:
        """
        self._inputs, self._outputs = verify_inputs_and_outputs(inputs, outputs)

        self._regex_list = regex_list

    def fit(self, X, y=None):
        return X

    def transform(self, X):
        X[self._outputs] = X[self._inputs].applymap(self._transform)
        return X

    def _transform(self, phrase):
        for word, regex in self._regex_list:
            phrase = regex.sub(word, phrase)
        return phrase


class DuplicateRemover(BaseEstimator, TransformerMixin):
    """
    중복 단어 제거

    Example

    >>> sample = pd.DataFrame(data={"이름":['청동 사과 할인 특가 사과', "삼성 컴퓨터 특가 세일 삼성", "완전 싸다 완전 초대박 싸다"]})
    >>> dr = DuplicateRemover(inputs='이름')
    >>> dr.transform(sample)
                 이름
    0   청동 사과 할인 특가
    1  삼성 컴퓨터 특가 세일
    2     완전 싸다 초대박
    """

    def __init__(self, inputs=[], outputs=None):
        self._inputs, self._outputs = verify_inputs_and_outputs(inputs, outputs)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self._outputs] = X[self._inputs].applymap(self._transform)
        return X

    @staticmethod
    def _transform(phrase):
        return " ".join(list(dict.fromkeys(phrase.split(" "))))


class StopWordRemover(BaseEstimator, TransformerMixin):
    """
    불용어를 제거

    Example
    >>> sample = pd.DataFrame(data={"이름":["노트북 할인 판매", "옷 기타 완전 세일", "비아그라 할인", "클래식기타 판매 세일", "판매왕의 판매"]})
    >>> transformer = StopWordRemover(inputs='이름',stop_words=['판매', '기타'])
    >>> transformer.transform(sample)
             이름
    0    노트북 할인
    1   옷 완전 세일
    2   비아그라 할인
    3  클래식기타 세일
    4      판매왕의
    """

    def __init__(self, inputs=[], outputs=None, stop_words=[]):
        self._inputs, self._outputs = verify_inputs_and_outputs(inputs, outputs)

        self._stop_words = stop_words
        self._sw_regex = re.compile(r'\b%s\b' %
                                    r'\b|\b'.join(map(re.escape, self._stop_words)))
        self._ds_regex = re.compile(r"\s+")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self._outputs] = X[self._inputs].applymap(self._transform)
        return X

    def _transform(self, phrase):
        _phrase = self._sw_regex.sub("", phrase)
        return self._ds_regex.sub(" ", _phrase).strip()


class WordLower(BaseEstimator, TransformerMixin):
    """
    모두 소문자화

    >>> sample = pd.DataFrame(data={"이름":['Kang', "KAM", "Kan"]})
    >>> wl = WordLower(inputs=['이름'])
    >>> wl.transform(sample)
         이름
    0  kang
    1   kam
    2   kan

    """

    def __init__(self, inputs=[], outputs=None):
        self._inputs, self._outputs = verify_inputs_and_outputs(inputs, outputs)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self._outputs] = X[self._inputs].applymap(self._transform)
        return X

    @staticmethod
    def _transform(word):
        return word.lower()


############################
# 3. Tokenizer
#   - MorphTokenizer
#   - NounTokenizer
#   - PosTokenizer
# TODO : 이 쪽은 transform 코드를 다 짠후 리팩토링 하려고 합니다.
# 고민포인트
#   konlpy를 wrapping하여 구성하려고 하는데
#   twitter를 주로 사용한다는 가정으로 설계하였습니다.
#   (좋지 못한 가정이고, 코드의 유연성을 떨어트리는 못된 행위이지요)
#   어떤 식으로 확장해야 좀 더 좋은 코드가 될 것인지
#   고민이 좀 들고 있었습니다.
############################
class MorphTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, inputs=[], outputs=None):
        self._inputs, self._outputs = verify_inputs_and_outputs(inputs, outputs)
        self._okt = Okt()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self._outputs] = X[self._inputs].applymap(self._transform)
        return X

    def _transform(self, phrase):
        return " ".join(self._okt.morphs(phrase))


class NounTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, inputs=[], outputs=None):
        self._inputs, self._outputs = verify_inputs_and_outputs(inputs, outputs)
        self._okt = Okt()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self._outputs] = X[self._inputs].applymap(self._transform)
        return X

    def _transform(self, phrase):
        return " ".join(self._okt.nouns(phrase))


class PosTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, inputs=[], outputs=None,
                 norm=False, stem=False,
                 excludes=['Punctuation', 'Number', 'Foreign']):
        self._inputs, self._outputs = verify_inputs_and_outputs(inputs, outputs)

        self._norm = norm
        self._stem = stem
        self._excludes = excludes
        self._okt = Okt()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self._outputs] = X[self._inputs].applymap(self._transform)
        return X

    def _transform(self, phrase):
        pos_list = self._okt.pos(phrase, norm=self._norm, stem=self._stem)
        pos_drop = list(filter(
            lambda pos: pos[1] not in self._excludes, pos_list))

        if len(pos_drop) == 0:
            return ""
        else:
            return " ".join(list(zip(*pos_drop))[0])

############################
# 4. Vectorizer
# TODO : vectorizer은 설계부터 고민해보고 있습니다.
# 고민포인트
# -----------------
# 오히려 임베딩 방법론은 적용하기는 간단해 보입니다.
# BOW 방법론을 어떤식으로 적용할지 고민이 되고 있습니다.
############################


############################
# 5. Category Transform
#
# 고민포인트
# ----------------
# 카카오 데이터셋에 종속적으로 one-hot encoding을 짤지,
# General한 형태로 one-hot encoding을 짤지가 고민이 되고 있습니다.
############################


class CategoryOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, inputs=[], outputs=None):
        self._inputs, self._outputs = verify_inputs_and_outputs(inputs, outputs)

        not_category_name = set(self._inputs) - set(['bcateid','dcateid', 'mcateid', 'scateid'])
        if len(not_category_name) > 0:
            raise ValueError("카테고리 이름이 아닌 것이 존재합니다 . {}".format(not_category_name))


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for src_name, dst_name in zip(self._inputs, self._outputs):
            X[dst_name] = X[src_name].map(partial(self._transform,src_name))
        return X

    @staticmethod
    def _transform(category_name, x):
        if category_name == "bcateid":
            vec_size = 57
        elif category_name == "mcateid":
            vec_size = 552
        elif category_name == "scateid":
            vec_size = 3190
        elif category_name == "dcateid":
            vec_size = 404

        result = np.zeros(shape=(vec_size,), dtype=np.bool)
        result[abs(x)-1] = 1
        return result

def verify_inputs_and_outputs(inputs, outputs):
    """
    Transform에 해당하는 inputs와 outputs를 일정한 형태로 만들어줌
    :param inputs:
    :param outputs:
    :return:
    """
    _inputs = None
    if isinstance(inputs, list) or isinstance(inputs, tuple):
        if len(inputs) == 0:
            raise ValueError("inputs에는 최소 한 개 이상의 column 이름을 지정해주어야 합니다.")
        _inputs = inputs
    elif isinstance(inputs, str):
        _inputs = [inputs]
    else:
        raise TypeError("inputs에 적절하지 못한 dataType 들어왔습니다.")

    _outputs = None
    _outputs = None
    if outputs is None:
        _outputs = _inputs
    elif isinstance(outputs, list) or isinstance(outputs, tuple):
        _outputs = outputs
    elif isinstance(outputs, str):
        _outputs = [outputs]
    else:
        raise TypeError("outputs에 적절하지 못한 DataType이 들어왔습니다.")

    return _inputs, _outputs
