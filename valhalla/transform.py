import re
from typing import List

import numpy as np
import pandas as pd
from konlpy.tag import Okt
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ['ColumnSelector', 'ColumnMerger', 'WordUnifier',
           'RegExReplacer', 'DuplicateRemover', 'StopWordRemover',
           'WordLower', 'MorphTokenizer', 'NounTokenizer', 'PosTokenizer']

"""
## Transform 코드 구성

### 자연어 처리 부분 Transform
    - string을 받아 string을 내뱉는 것을 원칙으로 한다. ( 순서에 무관할 수 있도록 )
    - 멱등성을 가져야 한다. (동일 input, 동일 output)
    - input type에 따라 output datatype이 정해지는 구조로 한다.
      np.ndarray -> np.ndarray
      pd.Series -> pd.Series
      pd.Dataframe -> pd.Dataframe


    1. DataFrame Preprocessing
        DataExtractor에서 어떤 컬럼에 전처리를 진행할지 결정하는 부분

    2. Basic Preprocssing
        기본적인 문장에 대한 전처리 코드가 들어가는 부분

    3. Tokenizer
        형태소 분석 부분. tokenizer 엔진으로서는 konlpy에서의 Okt(이전 Twitter)을 적용한다.
        reference : https://konlpy-ko.readthedocs.io/ko/v0.4.3/api/konlpy.tag/

    ----------------------------
    아래 Vectorizer는 아직 어떤 식으로 구현하는 것이 좋을지 고민이 되고 있습니다.
    여기에 대해서 같이 토의해보아요
    ----------------------------
    4. Vectorizer
        word를 벡터화시키는 부분

        참고 자료들
        1) BOW(Bag of Words) 방법론
        reference : https://datascienceschool.net/view-notebook/3e7aadbf88ed4f0d87a76f9ddc925d69/

        2) 임베딩 방법론
        reference : https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/11/embedding/


### Category 라벨링 부분 Transform
    우리는 기본적으로 분류모델. 예측 목표가 되는 라벨 정보가 있고, 카카오에서는
    labeling이 핵심이 된다.

    5.  OneHotEncoder
        - CategoryOneHotEncoder
            카테고리 라벨 정보를 one-hot을 encoding함

        - CategoryMergeEncoder
            카테고리 라벨 정보를 합쳐서 encoding함

"""


############################
# 1. DataFrame Preprocessing
#    - ColumnSelector
#    - ColumnMerger
############################


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    주어진 데이터프레임에서 Pipeline에서 적용할 컬럼을 선택

    Example

    >>> df = pd.DataFrame(data={ "과일" : ['사과','배','딸기'],"시장" : ['명동','상정','죽도']})
    >>> cs = ColumnSelector("과일")
    >>> cs.transform(df)
    0    사과
    1     배
    2    딸기
    Name: 과일, dtype: object

    """

    def __init__(self, col_name):
        self.col_name = col_name

    def fit(self, X, y=None):
        if self.col_name not in X.columns:
            raise ValueError("DataFrame 내에 {}가 없습니다.".format(self.col_name))
        return self

    def transform(self, X):
        return X[self.col_name]


class ColumnMerger(BaseEstimator, TransformerMixin):
    """
    주어진 데이터프레임에서 컬럼에 해당하는 string을 합치는

    Example

    >>> df = pd.DataFrame(data={ "과일" : ['사과','배','딸기'],"시장" : ['명동','상정','죽도']})
    >>> cs = ColumnMerger(['과일','시장'])
    >>> cs.transform(df)
    0    사과 명동
    1     배 상정
    2    딸기 죽도
    dtype: object

    """

    def __init__(self, col_names=[]):
        self.col_names = col_names

    def fit(self, X, y=None):
        for col_name in self.col_names:
            if col_name not in X.columns:
                raise ValueError("DataFrame 내에 {}가 없습니다.".format(col_name))
        return self

    def transform(self, X):
        return X[self.col_names].apply(lambda x: " ".join(x), axis=1)


############################
# 2. Basic NLP Preprocssing
#    - WordUnifier
#
#    - DuplicateRemover
#    - StopWordRemover
#    - RegExReplacer
#
#    - WordLower
############################
class WordUnifier(BaseEstimator, TransformerMixin):
    """
    동일의미 다른 표기 통일

    # TODO : 구현은 쉽지만, 잘못 구현 할 경우 속도 이슈가 날 거 같습니다.
    # 속도 이슈 없는 코드를 원합니다!

    Example

    >>> sample = np.array(['삼성전자 노트북', "노트북 삼성", "samsung 스마트폰", 'lg 폰', "엘지전자 상거래"])
    >>> wu = WordUnifier([["삼성","삼성전자",'samsung'], ["엘지",'엘지전자','lg']])
    >>> wu.transform(sample)
    array(['삼성 노트북', "노트북 삼성", "삼성 스마트폰", '엘지 폰', "엘지 상거래"], dtype=object)

    """

    def __init__(self, words_list=[]):
        self._words_list = words_list

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            x_shape = X.shape
            return np.array([self._transform(phrase)
                             for phrase in X.ravel()]).reshape(x_shape)
        elif isinstance(X, pd.Series):
            return X.map(self._transform)
        elif isinstance(X, pd.DataFrame):
            return X.applymap(self._transform)
        elif isinstance(X, list) or isinstance(X, tuple):
            return [self._transform(phrase) for phrase in X]
        else:
            raise TypeError("적절하지 못한 DataType이 들어왔습니다.")

    @staticmethod
    def _transform(phrase):
        # TODO : wordunifier 구현
        return


class RegExReplacer(BaseEstimator, TransformerMixin):
    """
    정규식을 활용한 word 치환
    주어진 정규식에 만족하는 word에 대해서, 특정 word로 변경하는 코드

    Example

    >>>
    >>>
    >>>
    """

    def __init__(self, regex_list=[]):
        self._regex_list = regex_list

    def fit(self, X, y=None):
        return X

    def transform(self, X):
        if isinstance(X, np.ndarray):
            x_shape = X.shape
            return np.array([self._transform(phrase)
                             for phrase in X.ravel()]).reshape(x_shape)
        elif isinstance(X, pd.Series):
            return X.map(self._transform)
        elif isinstance(X, pd.DataFrame):
            return X.applymap(self._transform)
        elif isinstance(X, list) or isinstance(X, tuple):
            return [self._transform(phrase) for phrase in X]
        else:
            raise TypeError("적절하지 못한 DataType이 들어왔습니다.")

    @staticmethod
    def _transform(phrase) -> List:
        if re.search(r'[0-9]+(kg|KG|Kg)', phrase) is not None:
            result = re.sub(r'[0-9]+(kg|KG|Kg)', '<단위>', phrase)
        elif re.search(r'[0-9]+.(L)', phrase) is not None:
            result = re.sub(r'[0-9]+(L)', '<부피단위>', phrase)
        else:
            result = phrase
        return result


class DuplicateRemover(BaseEstimator, TransformerMixin):
    """
    중복 단어 제거

    Example

    >>> sample = np.array(['청동 사과 할인 특가 사과', "삼성 컴퓨터 특가 세일 삼성", "완전 싸다 완전 초대박 싸다"])
    >>> dr = DuplicateRemover()
    >>> dr.transform(sample)
    array(['청동 사과 할인 특가', '삼성 컴퓨터 특가 세일', '완전 싸다 초대박'], dtype='<U12')
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            x_shape = X.shape
            return np.array([self._transform(phrase)
                             for phrase in X.ravel()]).reshape(x_shape)
        elif isinstance(X, pd.Series):
            return X.map(self._transform)
        elif isinstance(X, pd.DataFrame):
            return X.applymap(self._transform)
        elif isinstance(X, list) or isinstance(X, tuple):
            return [self._transform(phrase) for phrase in X]
        else:
            raise TypeError("적절하지 못한 DataType이 들어왔습니다.")

    @staticmethod
    def _transform(phrase):
        return " ".join(list(dict.fromkeys(phrase.split(" "))))


class StopWordRemover(BaseEstimator, TransformerMixin):
    """
    불용어를 제거

    Example
    >>> sample = ["노트북 할인 판매", "옷 기타 완전 세일", "비아그라 할인", "클래식기타 판매 세일", "판매왕의 판매"]
    >>> transformer = StopWordRemover(['판매', '기타'])
    >>> transformer.transform(sample)
    ["노트북 할인", "옷 완전 세일", "비아그라 할인", "클래식기타 세일", "판매왕의"]
        pred = transformer.transform(answer)
    """

    def __init__(self, stop_words=[]):
        self._stop_words = stop_words
        self._sw_regex = re.compile(r'\b%s\b' %
                                    r'\b|\b'.join(map(re.escape, self._stop_words)))
        self._ds_regex = re.compile(r"\s+")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            x_shape = X.shape
            return np.array([self._transform(phrase)
                             for phrase in X.ravel()]).reshape(x_shape)
        elif isinstance(X, pd.Series):
            return X.map(self._transform)
        elif isinstance(X, pd.DataFrame):
            return X.applymap(self._transform)
        elif isinstance(X, list) or isinstance(X, tuple):
            return [self._transform(phrase) for phrase in X]
        else:
            raise TypeError("적절하지 못한 DataType이 들어왔습니다.")

    def _transform(self, phrase):
        _phrase = self._sw_regex.sub("", phrase)
        return self._ds_regex.sub(" ", _phrase).strip()


class WordLower(BaseEstimator, TransformerMixin):
    """
    모두 소문자화

    >>> sample = np.array(['Kang', "KAM", "Kan"])
    >>> wl = WordLower()
    >>> wl.transform(sample)
    array(['kang', 'kam', 'kan'], dtype='<U4')

    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            x_shape = X.shape
            return np.array([self._transform(phrase)
                             for phrase in X.ravel()]).reshape(x_shape)
        elif isinstance(X, pd.Series):
            return X.map(self._transform)
        elif isinstance(X, pd.DataFrame):
            return X.applymap(self._transform)
        elif isinstance(X, list) or isinstance(X, tuple):
            return [self._transform(phrase) for phrase in X]
        else:
            raise TypeError("적절하지 못한 DataType이 들어왔습니다.")

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
    def __init__(self):
        self._okt = Okt()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            x_shape = X.shape
            return np.array([self._transform(phrase)
                             for phrase in X.ravel()]).reshape(x_shape)
        elif isinstance(X, pd.Series):
            return X.map(self._transform)
        elif isinstance(X, pd.DataFrame):
            return X.applymap(self._transform)
        elif isinstance(X, list) or isinstance(X, tuple):
            return [self._transform(phrase) for phrase in X]
        else:
            raise TypeError("적절하지 못한 DataType이 들어왔습니다.")

    def _transform(self, phrase):
        return " ".join(self._okt.morphs(phrase))


class NounTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._okt = Okt()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            x_shape = X.shape
            return np.array([self._transform(phrase)
                             for phrase in X.ravel()]).reshape(x_shape)
        elif isinstance(X, pd.Series):
            return X.map(self._transform)
        elif isinstance(X, pd.DataFrame):
            return X.applymap(self._transform)
        elif isinstance(X, list) or isinstance(X, tuple):
            return [self._transform(phrase) for phrase in X]
        else:
            raise TypeError("적절하지 못한 DataType이 들어왔습니다.")

    def _transform(self, phrase):
        return " ".join(self._okt.nouns(phrase))


class PosTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, norm=False, stem=False,
                 excludes=['Punctuation', 'Number', 'Foreign']):
        self._norm = norm
        self._stem = stem
        self._excludes = excludes
        self._okt = Okt()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            x_shape = X.shape
            return np.array([self._transform(phrase)
                             for phrase in X.ravel()]).reshape(x_shape)
        elif isinstance(X, pd.Series):
            return X.map(self._transform)
        elif isinstance(X, pd.DataFrame):
            return X.applymap(self._transform)
        elif isinstance(X, list) or isinstance(X, tuple):
            return [self._transform(phrase) for phrase in X]
        else:
            raise TypeError("적절하지 못한 DataType이 들어왔습니다.")

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
# TODO : 어떤 식으로 짤지에 대해 고민해보고 있습니다.
# 고민포인트
# ----------------
# 카카오 데이터셋에 종속적으로 one-hot encoding을 짤지,
# General한 형태로 one-hot encoding을 짤지가 고민이 되고 있습니다.
############################

