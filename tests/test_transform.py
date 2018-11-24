import sys
from unittest import TestCase, main

import re
import pandas as pd
from numpy.testing import assert_array_equal

sys.path.append("../")
from valhalla.transform import (ColumnMerger, WordUnifier,
                                DuplicateRemover, StopWordRemover,
                                WordLower, RegExReplacer,
                                MorphTokenizer, NounTokenizer,
                                PosTokenizer)

class BasicNLPPreprocessingSimpleTest(TestCase):
    """
    테스트 메소드 리스트
        - ColumnMerger : DONE
        - WordUnifier : DONE
        - DuplicateRemover : DONE
        - StopWordRemover : DONE
        - RegExReplacer : Done
        - WordLower : DONE
    """

    def test_ColumnMerger_with_pandas_input(self):
        df = pd.DataFrame(
            data={
                "과일": [
                    '사과', '배', '딸기'], "시장": [
                    '명동', '상정', '죽도']})
        answer = df.copy()
        answer['결과'] = ["사과 명동", "배 상정", "딸기 죽도"]
        cs = ColumnMerger(["과일", "시장"],outputs='결과')
        pred = cs.transform(df)
        assert_array_equal(answer, pred)

    def test_word_unifier_with_pandas_input(self):
        df = pd.DataFrame(data={"이름" : ['삼성전자 노트북', "노트북 삼성", "samsung 스마트폰", 'lg 폰', "엘지전자 상거래"]})
        answer = df.copy()
        answer['이름'] = ['삼성 노트북', "노트북 삼성", "삼성 스마트폰", "엘지 폰", "엘지 상거래"]

        transformer = WordUnifier(inputs=['이름'], words_list= [["삼성", "삼성전자", 'samsung'], ["엘지", '엘지전자', 'lg']])

        pred = transformer.transform(df)
        assert_array_equal(answer, pred)

    def test_RegExReplacer_with_pandas_input(self):
        df = pd.DataFrame({'단위':["열무김치 10kg 판매", "매실 5kg 포장", "미닛메이드 10L 병", "포도주스 30L 병",
                            "kgfi 공인 판매", "lipspace 판매"]})
        transformer = RegExReplacer(inputs='단위', regex_list=[("<단위>",re.compile("[0-9]+kg")),
                                                              ("<부피단위>", re.compile("[0-9]+L"))])
        answer = df.copy()
        answer['단위'] = ["열무김치 <단위> 판매", "매실 <단위> 포장", "미닛메이드 <부피단위> 병", "포도주스 <부피단위> 병",
                            "kgfi 공인 판매", "lipspace 판매"]

        pred = transformer.transform(df)
        assert_array_equal(answer, pred)

    def test_DuplicateRemover_with_pandas_input(self):
        df = pd.DataFrame({"이름" : ['청동 사과 할인 특가 사과', "삼성 컴퓨터 특가 세일 삼성", "완전 싸다 완전 초대박 싸다"]})

        transformer = DuplicateRemover(inputs='이름')
        answer = df.copy()
        answer['이름'] = ['청동 사과 할인 특가', '삼성 컴퓨터 특가 세일', '완전 싸다 초대박']

        pred = transformer.transform(df)
        assert_array_equal(answer, pred)

    def test_StopWordRemover_with_pandas_input(self):
        df = pd.DataFrame({"이름": ["노트북 할인 판매", "옷 기타 완전 세일",
                            "비아그라 할인", "클래식기타 판매 세일", "판매왕의 판매 전략"]})
        transformer = StopWordRemover(inputs='이름', stop_words=['판매', '기타'])
        answer = df.copy()

        answer['이름'] = ["노트북 할인", "옷 완전 세일", "비아그라 할인", "클래식기타 세일", "판매왕의 전략"]

        pred = transformer.transform(df)
        assert_array_equal(answer, pred)

    def test_WordLower_with_pandas_input(self):
        df = pd.DataFrame({"이름": ["Kang", "KAM", "Kan"]})
        transformer = WordLower(inputs='이름')
        answer = df.copy()
        answer['이름'] = ["kang", "kam", "kan"]

        pred = transformer.transform(df)
        assert_array_equal(answer, pred)


class TokenizerSimpleTest(TestCase):

    def test_MorphTokenizer_with_pandas_input(self):
        df = pd.DataFrame({"이름":["아버지가방에 들어가신다! 123", "어머니김치는 참맛있다!"]})
        transformer = MorphTokenizer(inputs='이름')
        answer = df.copy()
        answer['이름'] = ['아버지 가방 에 들어가신다 ! 123', '어머니 김치 는 참 맛있다 !']

        pred = transformer.transform(df)
        assert_array_equal(answer, pred)

    def test_NounTokenizer_with_pandas_input(self):
        df = pd.DataFrame({"이름":["아버지가방에 들어가신다! 123", "어머니김치는 참맛있다!"]})
        transformer = NounTokenizer(inputs='이름')
        answer = df.copy()
        answer['이름'] = ['아버지 가방', '어머니 김치']

        pred = transformer.transform(df)
        assert_array_equal(answer, pred)

    def test_PosTokenizer_with_pandas_input(self):
        df = pd.DataFrame({"이름":["아버지가방에 들어가신다! 123", "어머니김치는 참맛있다!"]})
        transformer = PosTokenizer(inputs='이름')
        answer = df.copy()
        answer['이름'] = ['아버지 가방 에 들어가신다', '어머니 김치 는 참 맛있다']

        pred = transformer.transform(df)
        assert_array_equal(answer, pred)


if __name__ == '__main__':
    main()
