# Phase1.  Extract

```markdown
ver.0.01
TEAM DEMIGUISE
2018.11.17 (sat)
Programmer : rocket
Author : Best10
Purpose : Sharing, Specification, Learning, User's manual
```

h5 Data를 정말 활용하기 좋게 만들어줄 Class를 만들었다. 

**Valhallian**은 이 DataExtractor를 생성하여 dataframe처럼 쓰기만 하면 된다.

그밖에도 chunk로 나누어져있는 h5를 통합하는 함수, category structure를 dictionary로 얻는 함수를 구현해 두었다.

데이터를 활용할 준비는 모두 되었으니, 여러분은 이제 즐기라.



## 1. Overview

**1.Code :** [~/valhalla/extract.py](#3-dataextractor)

**2.Script :**

​	1) ~/script/2) 데이터를\ 한데\ 모으자.ipynb  

​	2) ~/script/3)데이터\ 로더\ 만들기.ipynb  

**3.Input** : 1. textOnly.h5, 2. cate1.json, 3. train.chunk.01.h5 ~ train.chunk.09.h5

**4.Output** : 1. DataExtractor Class,  2. cate_map dictionary, 3. textOnly.h5



## 2. Motivation



1. hdf5 type의 data에 대한 이해가 부족한 User들에게 이를  pandas의 dataframe 처럼 활용할 수 있는 환경을 제공하자.
2. hdf5파일의 특성 상, (1)  여러 Session 에서 동시 open 시 file에 lock  (2) instance를 delete하지 않고 Session 이 종료될경우 file에 lock이 걸려 더 이상 file을 사용할 수 없는 문제 해결하자.  
3. 누구나 손쉽게 hdf5 format file을 open하고, load하고 save할 수 있는 환경을 만들자.



## 3. DataExtractor

**1. category :** Data Structure

**2. Spec :** 

 	1. h5 type의 file open 및 read, data 통합 후 save 기능 제공
 	2. hdf5 type의 data를 pandas의 문법과 동일하게 Column Selecting 및 Row Slicing
 	3. 다수의 Session 에서 파일에 lock이 걸리는 문제 해결



## 4. Usage

#### 1. DataExtractor class

```python
1. DataExtractor Instance 생성
dl = DataExtractor( H5_SRC_FILE_PATH, subset_name, df_format) 

2. User가 원하는 Data를 호출 
dl[[column_name, ... ], start_row : end_row ] 
# pandas dataframe이 return
```

#### 2. get_category_map()

```python
1. cate1.json 파일을 read
cate_map = get_category_map(json_path="SRC_FILE_PATH") 
# cate_map : dictionary -> {cateid : {name : code, ... }}
```

#### 3. merge_h5_files()

```python
1. 나누어진 h5 file을 합친 후 저장
merge_ht_files(merge_dir="SRC_DIR_PATH", save_path = "SAVE_FILE_PATH", excludes = [column_name, ... ])
# save_path에 통합된 h5파일 저장, excludes column 제외
```



## 5. Code Structure

```python
class DataExtractor(object):
    
    # @Magic Method
	def __init__(self, file_path, subset_name='train', df_format=True):
		# hdf5 파일 호출, column의 type 별로 다른 변수에 저장
	def __len__(self):
        return # 데이터의 길이
    def __getitem__(self, key):
        return # pandas dataframe
		# * h5 type의 data를 pandas dataframe처럼 호출하도록 처리
        # h5형식의 전체를 호출하는 것이 아닌, User가 pandas처럼 호출한 부분만 call

    # @internal Method
    def _get_item(self, col, slc):
        return item # 사용자가 지정한 부분의 item
    	#__getitem__ 에서 호출
    
    # @staticmethod
    def _decode_utf8(byte):
        return # utf-8로 decoding 된 byte 반환
    	# 문자열에 대한 encoding 문제 해결
    def _divide_col_and_slice(self, key):
        return # column, row-slice data
    	#__getitem__ 에서 호출
    	# 사용자가 submit한 key value를 통해 col을 selecting, row를 slicing

def get_category_map(json_path):
    return # { cateid : category_name }
	# kakao에서 제공한 cate1.json ( category structure info.)을 dictionary로
    # 이때, kakao의 데이터는 name:code 구조로 제공 -> code:name으로 변환

def merge_h5_files(merge_dir, save_path, excludes=['img_feat']):
    save # h5로 통합된 file
    # chunk로 나뉘어진 h5파일이 저장된 경로를 input
    # excludes parameter로 저장하지 않을 column 선택 가능.
    # img_feat의 경우, data의 크기가 너무 커서 excluding
    
```



## 6. Examples

```python
>>> dl = DataExtractor("./data/prep/textOnly.h5",'train')
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
```



## 7. Issues

pass
