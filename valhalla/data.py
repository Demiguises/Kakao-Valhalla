import json
import os
import warnings
from operator import itemgetter

import fire
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


class DataLoader(object):
    """
    h5파일에서 데이터를 불러오는 DataLoader. H5파일의 모든 데이터를 램에 한번에 올리지 않고
    pandas DataFrame와 같이 지정해줄 때, 부분만 가져올 수 있도록 코드를 수정함

    example
    >>> dl = DataLoader("./data/prep/textOnly.h5",'train')
    >>> dl['pid',0]
                pid
    0	Q4081781803
    >>> dl[['pid','brand'], 0]
                pid	brand
    0	W4100190891	NT커터
    >>> dl[['pid','brand'],8:10]
                pid	brand
    0	J3959473240	에트로
    1	K4487826783	얀케이스
    >>> dl[:,0]
        bcateid	dcateid	mcateid	price	scateid	brand	maker	model	pid	product	updttm
    0	54	-1	467	87210	2082	꽃배달늑대와여우	꽃배달늑대와여우	근조화환	J4586931195	031-893-8020 평택안중백병원장례식장 화환추천BZ	20180421102112
    >>> dl['pid',[0,1,4]]
                pid
    0	Q4081781803
    1	W4203425504
    2	I4066071748

    """

    def __init__(self, file_path, subset_name='train'):
        """

        :param file_path: 읽어들일 .h5 path
        :param subset_name: train / dev / test 중 하나
        """
        if not os.path.exists(file_path):
            raise ValueError("file_path{}에 파일이 존재하지 않습니다.".format(file_path))

        self._f = h5py.File(file_path)
        if subset_name not in self._f:
            raise KeyError("{}이 {}에 없습니다".format(subset_name, file_path))
        self._g = self._f[subset_name]

        # column type 나누기
        self._int_cols = []
        self._str_cols = []
        for col, value in self._g.items():
            if np.issubdtype(value.dtype, np.string_):
                self._str_cols.append(col)
            elif np.issubdtype(value.dtype, np.integer):
                self._int_cols.append(col)
            elif np.issubdtype(value.dtype, np.floating):
                warnings.warn("아직 img_feat를 Data Loader로 호출할 수 없습니다. img_feat는 직접 h5파일에서 호출해 주세요.")
                # TODO : 아직 img_feat를 읽어들이는 기능은 추가하지 않았음
                pass
            else:
                raise ValueError("integer도 아니고 string도 아닌 미친놈이 있어요")

        self.columns = self._int_cols + self._str_cols
        self._len = len(self._g['pid'][:])

    def __len__(self):
        return self._len

    def __getitem__(self, key):
        if isinstance(key, str):
            cols = key
            slc = slice(None, None, None)
        elif isinstance(key, tuple):
            cols, slc = key
        elif isinstance(key, list):
            cols = key
            slc = slice(None, None, None)
        else:
            raise ValueError("뭔가 지정을 이상하게 했는데...")

        # 인자 유효성 검사
        if isinstance(cols, str):
            cols = [cols]
        elif isinstance(cols, list):
            cols = cols
        elif isinstance(cols, slice):
            cols = self.columns[cols]
        else:
            raise KeyError("column 지정 좀 똑바로 하지?")

        if not (isinstance(slc, slice)
                or isinstance(slc, int)
                or isinstance(slc, list)
                or isinstance(slc, np.ndarray)):
            raise KeyError("범위 값으로 좀 똑바로 넣지?")

        # H5에서 실제로 파일을 가져오는 부분
        if isinstance(slc, list) or isinstance(slc, np.ndarray):
            # order가 보장이 되지 않는다.
            order_flag = True
            sorted_list = sorted(enumerate(slc), key=itemgetter(1))
            idx, slc = list(zip(*sorted_list))
            idx, slc = list(idx), list(slc)
        else:
            # order가 보장된다.
            order_flag = False

        items = []
        for col in cols:
            item = self._get_item(col, slc)
            if order_flag:
                item.index = idx
                item.sort_index(inplace=True)
            items.append(item)

        if len(items) > 1:
            return pd.concat(items, axis=1)
        else:
            return items[0]

    def __del__(self):
        # 이 인스턴스가 제거될 때 file을 닫아주어야 함
        self._f.close()
        del self._f

    def _get_item(self, col, slc):
        if col in self.columns:
            item = self._g[col][slc]
        else:
            raise KeyError("DataLoader Column에 {}키는 없습니다.".format(col))
        series = pd.Series(item, name=col)

        if col in self._str_cols:
            return series.map(lambda x: x.decode('utf-8'))
        else:
            return series


def get_category_map(json_path):
    """
    카카오에서 제공된 json 파일을 파싱하여 code2name map return
    :param json_path:
    :return: dictionary

    example
    >>> cate_map = get_category_map(json_path="../data/raw/cate1.json")
    """
    json_data = open(json_path).read()
    data = json.loads(json_data)
    s_cate_map = {value: key for key, value in data['s'].items()}
    b_cate_map = {value: key for key, value in data['b'].items()}
    m_cate_map = {value: key for key, value in data['m'].items()}
    d_cate_map = {value: key for key, value in data['d'].items()}
    return {
        "scateid": s_cate_map,
        "bcateid": b_cate_map,
        "mcateid": m_cate_map,
        "dcateid": d_cate_map
    }


def merge_h5_files(merge_dir, save_path, excludes=['img_feat']):
    """
    나뉘어진 h5 파일을 합쳐주는 함수

    :param merge_dir: 함께 합칠 .h5 파일을 모은 디렉토리의 위치
    :param save_path: 모은 것을 저장할 위치
    :param excludes: 제외할 Column 이름
    :return:
    """
    # merge 하고자 하는 데이터 셋의 파일 path
    file_paths = [os.path.join(merge_dir, file_name)
                  for file_name in os.listdir(merge_dir)
                  if "chunk" in file_name]

    # Merge하고자 하는 attribute
    with h5py.File(file_paths[0], 'r') as f:
        group_name = list(f.keys())[0]
        attrs = list(f[group_name].keys())
    attrs = list(set(attrs) - set(excludes))

    # Main Merge
    with h5py.File(save_path, 'w') as write_file:
        for attr in tqdm(attrs):
            values = {}
            for file_path in file_paths:
                # 데이터를 읽어들임
                with h5py.File(file_path, 'r') as read_file:
                    group_name = list(read_file.keys())[0]
                    value = read_file[group_name][attr][:]

                # subset 그룹별로 저장
                if group_name not in values:
                    values[group_name] = [value]
                else:
                    values[group_name].append(value)

            # 나뉘어진 데이터들을 merge함
            for subset_name in values.keys():
                merge_value = np.concatenate(values[subset_name])

                if subset_name not in write_file:
                    subset = write_file.create_group(subset_name)
                else:
                    subset = write_file[subset_name]

                subset.create_dataset(attr, data=merge_value)

    file_size = os.stat(save_path).st_size // (1024 ** 2)
    print("{}에 저장되었습니다. (size : {}mb)".format(save_path, file_size))


if __name__ == "__main__":
    fire.Fire({
        "merge": merge_h5_files
    })
