* [Kakao-Valhalla 오픈카톡방](https://open.kakao.com/o/go6Idb4)
* [Kakao-Arena 대회Link](https://arena.kakao.com/c/1)

# 카카오 Arena -> Valhalla

> Kakao-Arena를 Arena 그 이상의 피터지는 Valhalla로. (모두를 위한 코드 전쟁터)

__우리는 Kakao-arena에 참여하며 Demiguises팀의 모든 코드와 아이디어를 공개합니다.__

### Introduction
대회는 **Competition**이지만, 우리는 **Contribution**으로 이 대회를 깨보고 싶다. 

마치 넷플릭스에서 진행된 추천 알고리즘 대회처럼, 
연구자 개발자들끼리 의견을 공유하면서 서로 성장할 수 있는 그런 대회를 만들고 싶다. 

이 세계에서 우린 초보자에 불과하지만, 이 프로젝트를 함께 하는 우리의 작은 생각들이 쌓이면서 
2달간 미친듯이 크면 어떨까.

단순히 이 대회의 우승이 목표가 아닌, 대회를 진행하면서 실력을 미친듯이 키우고 싶다.
그래서 우리가 진행하는 모든 코드는 공개하면서 진행하고자 한다.

이 github을 읽는 모든 도전자와 사람들은 피드백해주시면 감사하다. 여기의 모든 내용들이 누군가에게
도움이 되었으면 좋겠다.  

3가지 목표를 가지고 시작한다. 
```    
1. 수단과 방법을 가리지 않는, 시도해 볼 수 있는 모든 방법을 총동원하기
    
2. 다른 카카오 아레나 도전자들도 우리 꺼 읽고 디스할 수 있는 코드를 짜기
    
3. 이 프로젝트를 진행하면서, 나중에 까먹지 않도록 생각의 흐름을 정리하기
```

이 글을 보고 혹시 함께 공부해보고 싶고, 이 대회에 도전하고 싶은 사람들은 위의 오픈카톡방으로 연락주시면 감사하다.
종종 연락하고 안부인사 좀 묻고 그런 사이가 좀 되어 보자.

### 설치

위의 코드를 모두 돌리기 위해서는 카카오에서 받은 모든 데이터를 data/raw/에 위치시켜야 한다.
```shell
pip install -r requirements.txt

# 전처리 파일 생성
python valhalla/data.py merge ../data/raw/ ../data/prep/textOnly.h5
```


### 순서

이 때까지 진행하고 있던 프로젝트의 흐름이다.

#### 1. 데이터 까보기 
> 카카오가 준 데이터가 도대체 뭐야? 어떻게 열어 볼까? 

#### 2. 데이터 모으기
> 카카오가 준 데이터 너무 용량 커, 버릴 거 버려서 내 노트북 위에서 기본적으로 돌릴 수 있게 줄여보자.

#### 3. 데이터 로더 만들기
> h5을 다루기가 귀찮다. 나는 pandas를 사랑한다. pandas 스타일로 바꾸어 버리자 

#### 4. 간단한 EDA 하기
> 이제 드디어 데이터를 좀 볼 수 있게 되었다. 한번 까서 보자

#### 5. 간단한 데이터 전처리 하기 (1) product & Model (TODO)
> 이야아. 데이터들이 굉장히 더럽다. 비정형 데이터인줄은 알았는데 극심하게 비정형이다. 어떻게 조지지 생각해보자

#### 6. 간단한 데이터 전처리 하기 (2) brand & Maker (TODO)
> Brand와 Maker의 데이터도 더럽기 매한가지이다. 어떻게 조져야 할 지 감도 안온다. 그래도 조지자. 

#### 7. 좀 더 구체적으로 데이터 EDA 하기 (TODO)
> 데이터가 좀 정제되니 분석이 가능해졌다. 뜯어보자. 

#### 8. 가장 간단한 모델로 돌려보기 (TODO)
> 우리만의 BaseLine 모델을 만들어 보자. (사실 카카오 코드 아직 안읽은 상태, 보고 나면 생각이 굳을까봐..)



### 기여자

1. rocketgrowthsj : rocketgrowthsj@gmail.com
2. Best10 : best10.csy@gmail.com
