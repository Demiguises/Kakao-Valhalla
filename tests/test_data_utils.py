from unittest import TestCase, main

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.util.testing import assert_series_equal, assert_frame_equal

from valhalla.data import DataLoader

"""
test.h5
내부 데이터는 총 4개의 group(bcateid, price, model, pid)로 구성되어 있고,
카카오에서 제공한 데이터와 동일한 포맷으로 구성되어 있음.
DataLoader의 동작이 제대로 되는지 테스트하기 위한 코드로, 데이터로서의 의미를 가지는 건 아니다

    bcateid	price	model	                                                    pid
0	24	    -1      		                                                    Q4081781803
1	17	    -1		                                                            W4203425504
2	24	    84750	인터파크/오피스메인/프린터/라벨/도트/바코드/기타/프린터 기타	            G4453903364
3	35	    -1	    중성펜/젤러펜 필기구 볼펜류 볼펜심제브라 Refill SK                    	U4418629259
4	40	    -1	    무형광아기세탁망원형[45cm]	                                    I4066071748
5	54	    87210	근조화환	                                                    J4586931195
6	35	    -1	    기타	                                                        F4662379886
7	34	    -1		                                                            O3764058858
8	14	    966000	인터파크/에트로/여성가방/숄더백(천연가죽)	                            J3959473240
9	3	    16620	인터파크/얀케이스/스마트폰/태블릿케이스/태블릿케이스/파우치/갤럭시용케이스/파우치	K4487826783
"""


class DataLoaderSimpleTest(TestCase):
    def setUp(self):
        self.dl = DataLoader("test.h5", 'train')

    def tearDown(self):
        del self.dl

    def test_init_dataloader(self):
        pass

    def test_length_of_dataloader(self):
        self.assertEqual(len(self.dl), 10)

    def test_columns_of_dataloader(self):
        answer = ['bcateid', 'price', 'model', 'pid']
        self.assertEqual(len(answer), len(self.dl.columns))  # 길이 같은지 확인
        self.assertListEqual(list(set(answer)), list(
            set(self.dl.columns)))  # Element 같은지 확인

    def test_get_item_by_column_name_bcateid(self):
        pred = self.dl['bcateid']
        answer = pd.Series([24, 17, 24, 35, 40, 54, 35, 34, 14, 3],
                           dtype='int32', name='bcateid')
        assert_series_equal(pred, answer)

    def test_get_item_by_coumn_name_model(self):
        pred = self.dl['model']
        answer = pd.Series(["",
                            "",
                            "인터파크/오피스메인/프린터/라벨/도트/바코드/기타/프린터 기타",
                            '중성펜/젤러펜 필기구 볼펜류 볼펜심제브라 Refill SK',
                            '무형광아기세탁망원형[45cm]',
                            '근조화환',
                            '기타',
                            '',
                            '인터파크/에트로/여성가방/숄더백(천연가죽)',
                            '인터파크/얀케이스/스마트폰/태블릿케이스/태블릿케이스/파우치/갤럭시용케이스/파우치'],
                           name='model')
        assert_series_equal(pred, answer)

    def test_get_item_by_multiple_column(self):
        pred = self.dl[['bcateid', 'price']]
        answer = pd.DataFrame([[24, -1],
                               [17, -1],
                               [24, 84750],
                               [35, -1],
                               [40, -1],
                               [54, 87210],
                               [35, -1],
                               [34, -1],
                               [14, 966000],
                               [3, 16620]],
                              columns=['bcateid', 'price'], dtype='int32')
        assert_frame_equal(pred, answer)

    def test_get_item_by_column_and_index(self):
        pred = self.dl['bcateid', 0]
        answer = pd.Series([24], name='bcateid')
        assert_series_equal(pred, answer)

    def test_get_item_by_column_and_slice(self):
        pred = self.dl['bcateid', 0:3]
        answer = pd.Series([24, 17, 24], name='bcateid', dtype='int32')
        assert_series_equal(pred, answer)

    def test_get_item_by_multiple_column_and_slice(self):
        pred = self.dl[['bcateid', 'price'], 0:3]
        answer = pd.DataFrame([[24, -1],
                               [17, -1],
                               [24, 84750]],
                              columns=['bcateid', 'price'], dtype='int32')
        assert_frame_equal(pred, answer)

    def test_get_item_by_multiple_column_and_list(self):
        pred = self.dl[['bcateid', 'price'], [0, 3]]
        answer = pd.DataFrame([[24, -1],
                               [35, -1]],
                              columns=['bcateid', 'price'], dtype='int32')
        assert_frame_equal(pred, answer)

    def test_get_item_by_multiple_column_and_list_2(self):
        pred = self.dl[['bcateid', 'price'], [5, 3, 4, 6, 0]]
        answer = pd.DataFrame([
            [54, 87210],
            [35, -1],
            [40, -1],
            [35, -1],
            [24, -1]],
            columns=['bcateid', 'price'], dtype='int32')
        assert_frame_equal(pred, answer)


class DataLoaderNumpyOutTest(TestCase):
    def setUp(self):
        self.dl = DataLoader("test.h5", 'train', df_format=False)

    def tearDown(self):
        del self.dl

    def test_init_dataloader(self):
        pass

    def test_length_of_dataloader(self):
        self.assertEqual(len(self.dl), 10)

    def test_columns_of_dataloader(self):
        answer = ['bcateid', 'price', 'model', 'pid']
        self.assertEqual(len(answer), len(self.dl.columns))  # 길이 같은지 확인
        self.assertListEqual(list(set(answer)), list(
            set(self.dl.columns)))  # Element 같은지 확인

    def test_get_item_by_column_name_bcateid(self):
        pred = self.dl['bcateid']
        answer = np.array([24, 17, 24, 35, 40, 54, 35, 34, 14, 3],
                          dtype='int32')
        assert_array_equal(pred, answer)

    def test_get_item_by_coumn_name_model(self):
        pred = self.dl['model']
        answer = np.array(["",
                           "",
                           "인터파크/오피스메인/프린터/라벨/도트/바코드/기타/프린터 기타",
                           '중성펜/젤러펜 필기구 볼펜류 볼펜심제브라 Refill SK',
                           '무형광아기세탁망원형[45cm]',
                           '근조화환',
                           '기타',
                           '',
                           '인터파크/에트로/여성가방/숄더백(천연가죽)',
                           '인터파크/얀케이스/스마트폰/태블릿케이스/태블릿케이스/파우치/갤럭시용케이스/파우치'])
        assert_array_equal(pred, answer)

    def test_get_item_by_multiple_column(self):
        pred = self.dl[['bcateid', 'price']]
        answer = np.array([[24, -1],
                           [17, -1],
                           [24, 84750],
                           [35, -1],
                           [40, -1],
                           [54, 87210],
                           [35, -1],
                           [34, -1],
                           [14, 966000],
                           [3, 16620]],
                          dtype='int32')
        assert_array_equal(pred, answer)

    def test_get_item_by_column_and_index(self):
        pred = self.dl['bcateid', 0]
        answer = np.array([24])
        assert_array_equal(pred, answer)

    def test_get_item_by_column_and_slice(self):
        pred = self.dl['bcateid', 0:3]
        answer = np.array([24, 17, 24], dtype='int32')
        assert_array_equal(pred, answer)

    def test_get_item_by_multiple_column_and_slice(self):
        pred = self.dl[['bcateid', 'price'], 0:3]
        answer = np.array([[24, -1],
                           [17, -1],
                           [24, 84750]],
                          dtype='int32')
        assert_array_equal(pred, answer)

    def test_get_item_by_multiple_column_and_list(self):
        pred = self.dl[['bcateid', 'price'], [0, 3]]
        answer = np.array([[24, -1],
                           [35, -1]],
                          dtype='int32')
        assert_array_equal(pred, answer)

    def test_get_item_by_multiple_column_and_list_2(self):
        pred = self.dl[['bcateid', 'price'], [5, 3, 4, 6, 0]]
        answer = np.array([
            [54, 87210],
            [35, -1],
            [40, -1],
            [35, -1],
            [24, -1]],
            dtype='int32')
        assert_array_equal(pred, answer)



if __name__ == '__main__':
    main()
