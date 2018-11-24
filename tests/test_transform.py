import sys
from unittest import TestCase, main

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.util.testing import assert_series_equal

sys.path.append("../")
from valhalla.transform import ColumnSelector, ColumnMerger
from valhalla.transform import WordUnifier, DuplicateRemover, StopWordRemover
from valhalla.transform import WordLower, RegExReplacer
from valhalla.transform import MorphTokenizer, NounTokenizer, PosTokenizer


class ColumnPreprocessingSimpleTest(TestCase):
    """
    테스트 메소드 리스트
        - ColumnSelector : DONE
        - ColumnMerger : DONE
    """

    def test_ColumnSelector(self):
        df = pd.DataFrame(
            data={
                "과일": [
                    '사과', '배', '딸기'], "시장": [
                    '명동', '상정', '죽도']})
        answer = pd.Series(data=['사과', '배', "딸기"])
        cs = ColumnSelector("과일")
        pred = cs.transform(df)
        assert_series_equal(answer, pred, check_names=False, check_dtype=False)

    def test_ColumnMerger(self):
        df = pd.DataFrame(
            data={
                "과일": [
                    '사과', '배', '딸기'], "시장": [
                    '명동', '상정', '죽도']})
        answer = pd.Series(data=["사과 명동", "배 상정", "딸기 죽도"])
        cs = ColumnMerger(["과일", "시장"])
        pred = cs.transform(df)
        assert_series_equal(answer, pred, check_names=False, check_dtype=False)


class BasicNLPPreprocessingSimpleTest(TestCase):
    """
    테스트 메소드 리스트
        - WordUnifier : TODO

        - DuplicateRemover : DONE
        - StopWordRemover : DONE
        - RegExReplacer : TODO => Input Argument가 어떤 구성이 깔끔할지 고민이 되는 중

        - WordLower : DONE
    """

    """
    Word Unifier Test
    """

    def test_word_unifier_with_list(self):
        sample = ['삼성전자 노트북', "노트북 삼성", "samsung 스마트폰", 'lg 폰', "엘지전자 상거래"]
        transformer = WordUnifier(
            [["삼성", "삼성전자", 'samsung'], ["엘지", '엘지전자', 'lg']])
        answer = ['삼성 노트북', "노트북 삼성", "삼성 스마트폰", "엘지 폰", "엘지 상거래"]

        pred = transformer.transform(sample)
        self.assertListEqual(answer, pred)

    def test_word_unifier_with_numpy_input(self):
        sample = np.array(
            ['삼성전자 노트북', "노트북 삼성", "samsung 스마트폰", 'lg 폰', "엘지전자 상거래"])
        transformer = WordUnifier(
            [["삼성", "삼성전자", 'samsung'], ["엘지", '엘지전자', 'lg']])
        answer = np.array(['삼성 노트북', "노트북 삼성", "삼성 스마트폰", "엘지 폰", "엘지 상거래"])

        pred = transformer.transform(sample)
        assert_array_equal(answer, pred)

    def test_word_unifier_with_pandas_input(self):
        sample = pd.Series(
            ['삼성전자 노트북', "노트북 삼성", "samsung 스마트폰", 'lg 폰', "엘지전자 상거래"])
        transformer = WordUnifier(
            [["삼성", "삼성전자", 'samsung'], ["엘지", '엘지전자', 'lg']])
        answer = pd.Series(['삼성 노트북', "노트북 삼성", "삼성 스마트폰", "엘지 폰", "엘지 상거래"])

        pred = transformer.transform(sample)
        assert_series_equal(answer, pred, check_names=False, check_dtype=False)

    """
    RegExReplacer Test
    """

    def test_RegExReplacer_with_list(self):
        sample = ["열무김치 10kg 판매", "매실 5kg 포장", "미닛메이드 10L 병", "포도주스 30L 병",
                  "kgfi 공인 판매", "lipspace 판매"]
        transformer = RegExReplacer([("[0-9]+kg", "<단위>"), ("[0-9]+L", "<부피단위>")])
        answer = ["열무김치 <단위> 판매", "매실 <단위> 포장", "미닛메이드 <부피단위> 병", "포도주스 <부피단위> 병",
                  "kgfi 공인 판매", "lipspace 판매"]

        pred = transformer.transform(sample)
        self.assertListEqual(answer, pred)

    def test_RegExReplacer_with_numpy_input(self):
        sample = np.array(["열무김치 10kg 판매", "매실 5kg 포장", "미닛메이드 10L 병", "포도주스 30L 병",
                           "kgfi 공인 판매", "lipspace 판매"])
        transformer = RegExReplacer([("[0-9]+kg", "<단위>"), ("[0-9]+L", "<부피단위>")])
        answer = np.array(["열무김치 <단위> 판매", "매실 <단위> 포장", "미닛메이드 <부피단위> 병", "포도주스 <부피단위> 병",
                           "kgfi 공인 판매", "lipspace 판매"])

        pred = transformer.transform(sample)
        assert_array_equal(answer, pred)

    def test_RegExReplacer_with_pandas_input(self):
        sample = pd.Series(["열무김치 10kg 판매", "매실 5kg 포장", "미닛메이드 10L 병", "포도주스 30L 병",
                            "kgfi 공인 판매", "lipspace 판매"])
        transformer = RegExReplacer([("[0-9]+kg", "<단위>"), ("[0-9]+L", "<부피단위>")])
        answer = pd.Series(["열무김치 <단위> 판매", "매실 <단위> 포장", "미닛메이드 <부피단위> 병", "포도주스 <부피단위> 병",
                            "kgfi 공인 판매", "lipspace 판매"])

        pred = transformer.transform(sample)
        assert_series_equal(answer, pred, check_names=False, check_dtype=False)

    """
    DuplicateRemover Test
    """

    def test_DuplicateRemover_with_list(self):
        sample = ['청동 사과 할인 특가 사과', "삼성 컴퓨터 특가 세일 삼성", "완전 싸다 완전 초대박 싸다"]
        transformer = DuplicateRemover()
        answer = ['청동 사과 할인 특가', '삼성 컴퓨터 특가 세일', '완전 싸다 초대박']

        pred = transformer.transform(sample)
        self.assertListEqual(answer, pred)

    def test_DuplicateRemover_with_numpy_input(self):
        sample = np.array(
            ['청동 사과 할인 특가 사과', "삼성 컴퓨터 특가 세일 삼성", "완전 싸다 완전 초대박 싸다"])
        transformer = DuplicateRemover()
        answer = np.array(['청동 사과 할인 특가', '삼성 컴퓨터 특가 세일', '완전 싸다 초대박'])

        pred = transformer.transform(sample)
        assert_array_equal(answer, pred)

    def test_DuplicateRemover_with_pandas_input(self):
        sample = pd.Series(
            ['청동 사과 할인 특가 사과', "삼성 컴퓨터 특가 세일 삼성", "완전 싸다 완전 초대박 싸다"])
        transformer = DuplicateRemover()
        answer = pd.Series(['청동 사과 할인 특가', '삼성 컴퓨터 특가 세일', '완전 싸다 초대박'])

        pred = transformer.transform(sample)
        assert_series_equal(answer, pred, check_names=False, check_dtype=False)

    """
    StopWordRemover Test
    """

    def test_StopWordRemover_with_list(self):
        sample = [
            "노트북 할인 판매",
            "옷 기타 완전 세일",
            "비아그라 할인",
            "클래식기타 판매 세일",
            "판매왕의 판매 전략"]
        transformer = StopWordRemover(['판매', '기타'])
        answer = ["노트북 할인", "옷 완전 세일", "비아그라 할인", "클래식기타 세일", "판매왕의 전략"]

        pred = transformer.transform(sample)
        self.assertListEqual(answer, pred)

    def test_StopWordRemover_with_numpy_input(self):
        sample = np.array(["노트북 할인 판매", "옷 기타 완전 세일",
                           "비아그라 할인", "클래식기타 판매 세일", "판매왕의 판매 전략"])
        transformer = StopWordRemover(['판매', '기타'])
        answer = np.array(
            ["노트북 할인", "옷 완전 세일", "비아그라 할인", "클래식기타 세일", "판매왕의 전략"])

        pred = transformer.transform(sample)
        assert_array_equal(answer, pred)

    def test_StopWordRemover_with_pandas_input(self):
        sample = pd.Series(["노트북 할인 판매", "옷 기타 완전 세일",
                            "비아그라 할인", "클래식기타 판매 세일", "판매왕의 판매 전략"])
        transformer = StopWordRemover(['판매', '기타'])
        answer = pd.Series(
            ["노트북 할인", "옷 완전 세일", "비아그라 할인", "클래식기타 세일", "판매왕의 전략"])

        pred = transformer.transform(sample)
        assert_series_equal(answer, pred, check_names=False, check_dtype=False)

    """
    WordLower Test
    """

    def test_WordLower_with_list(self):
        sample = ["Kang", "KAM", "Kan"]
        transformer = WordLower()
        answer = ["kang", "kam", "kan"]

        pred = transformer.transform(sample)
        self.assertListEqual(answer, pred)

    def test_WordLower_with_numpy_input(self):
        sample = np.array(["Kang", "KAM", "Kan"])
        transformer = WordLower()
        answer = np.array(["kang", "kam", "kan"])

        pred = transformer.transform(sample)
        assert_array_equal(answer, pred)

    def test_WordLower_with_pandas_input(self):
        sample = pd.Series(["Kang", "KAM", "Kan"])
        transformer = WordLower()
        answer = pd.Series(["kang", "kam", "kan"])

        pred = transformer.transform(sample)
        assert_series_equal(answer, pred, check_names=False, check_dtype=False)


class TokenizerSimpleTest(TestCase):
    """
    MorphTokenizer Test
    """

    def test_MorphTokenizer_with_list(self):
        sample = ["아버지가방에 들어가신다! 123", "어머니김치는 참맛있다!"]
        transformer = MorphTokenizer()
        answer = ['아버지 가방 에 들어가신다 ! 123', '어머니 김치 는 참 맛있다 !']

        pred = transformer.transform(sample)
        self.assertListEqual(answer, pred)

    def test_MorphTokenizer_with_numpy_input(self):
        sample = np.array(["아버지가방에 들어가신다! 123", "어머니김치는 참맛있다!"])
        transformer = MorphTokenizer()
        answer = np.array(['아버지 가방 에 들어가신다 ! 123', '어머니 김치 는 참 맛있다 !'])

        pred = transformer.transform(sample)
        assert_array_equal(answer, pred)

    def test_MorphTokenizer_with_pandas_input(self):
        sample = pd.Series(["아버지가방에 들어가신다! 123", "어머니김치는 참맛있다!"])
        transformer = MorphTokenizer()
        answer = pd.Series(['아버지 가방 에 들어가신다 ! 123', '어머니 김치 는 참 맛있다 !'])

        pred = transformer.transform(sample)
        assert_series_equal(answer, pred, check_names=False, check_dtype=False)

    """
    NounTokenizer Test
    """

    def test_NounTokenizer_with_list(self):
        sample = ["아버지가방에 들어가신다! 123", "어머니김치는 참맛있다!"]
        transformer = NounTokenizer()
        answer = ['아버지 가방', '어머니 김치']

        pred = transformer.transform(sample)
        self.assertListEqual(answer, pred)

    def test_NounTokenizer_with_numpy_input(self):
        sample = np.array(["아버지가방에 들어가신다! 123", "어머니김치는 참맛있다!"])
        transformer = NounTokenizer()
        answer = np.array(['아버지 가방', '어머니 김치'])

        pred = transformer.transform(sample)
        assert_array_equal(answer, pred)

    def test_NounTokenizer_with_pandas_input(self):
        sample = pd.Series(["아버지가방에 들어가신다! 123", "어머니김치는 참맛있다!"])
        transformer = NounTokenizer()
        answer = pd.Series(['아버지 가방', '어머니 김치'])

        pred = transformer.transform(sample)
        assert_series_equal(answer, pred, check_names=False, check_dtype=False)

    """
    PosTokenizer Test
    """

    def test_PosTokenizer_with_list(self):
        sample = ["아버지가방에 들어가신다! 123", "어머니김치는 참맛있다!"]
        transformer = PosTokenizer()
        answer = ['아버지 가방 에 들어가신다', '어머니 김치 는 참 맛있다']
        pred = transformer.transform(sample)
        self.assertListEqual(answer, pred)

    def test_PosTokenizer_with_numpy_input(self):
        sample = np.array(["아버지가방에 들어가신다! 123", "어머니김치는 참맛있다!"])
        transformer = PosTokenizer()
        answer = np.array(['아버지 가방 에 들어가신다', '어머니 김치 는 참 맛있다'])

        pred = transformer.transform(sample)
        assert_array_equal(answer, pred)

    def test_PosTokenizer_with_pandas_input(self):
        sample = pd.Series(["아버지가방에 들어가신다! 123", "어머니김치는 참맛있다!"])
        transformer = PosTokenizer()
        answer = pd.Series(['아버지 가방 에 들어가신다', '어머니 김치 는 참 맛있다'])

        pred = transformer.transform(sample)
        assert_series_equal(answer, pred, check_names=False, check_dtype=False)


if __name__ == '__main__':
    main()
