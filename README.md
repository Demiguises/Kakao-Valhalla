* [Kakao-Valhalla 오픈카톡방](https://open.kakao.com/o/go6Idb4)
* [Kakao-Arena 대회Link](https://arena.kakao.com/c/1)

# Kakao-Arena -> Valhalla

> Kakao-Arena를 대회(Arena) 그 이상의 전쟁터(Valhalla)로. ( 누구나 참여 가능한 전장의 설계 )

__우리는 Kakao-arena(1. 쇼핑몰 카테고리 분류)에 대한 Demiguises팀의 모든 코드와 아이디어를 공개합니다.__



### Introduction

우리는 **Competition**의 **Contestant**이자 **Contributor**로서 이 대회를 더욱 피터지는 전장으로 만들고자 한다.



__AS-IS__

1. 함께 싸우고 싶은 많은 개발자들이 우수한 모델링 기술과 지식을 가지고 있음에도 기술적 제약이나 딥러닝 이외의 지식 (Data handling 등)이 부족으로 대회 출전에 주저하고 있다.
2. 뿐만 아니라, 2달이라는 대회 기간동안 딥러닝 모델링에 집중을 하지 못하고 다른 문제에 부딫혀 빛을 발하지 못하는 안타까운 팀도 많이 있다.
3. 스스로 실력이 미숙하다고 생각하거나, 문화적 차이를 느낀다거나, 용기가 나질 않아 커뮤니티 활동(대회, 해커톤, 네트워킹 등)에 등져 자신감을 잃은 개발자들을 많이 목격했다.

__TO-BE__

1. 우리는 그러한 제약을 최소화하고, 대회 참가자 간 지식을 자유롭게 공유하는 커뮤니티를 형성하여 **대회의 Contributor**가 되고자 한다.<br>

2. 말그대로, 대회를 늘상 열리는 단순한 개발 대회가 아닌 **딥러닝 모델링에 대한 지식을 가지고 있는 누구나 키보드만 들고 달려들어 싸울 수 있는 전장로 만드는 데 뜻이 있다.**

3. 아울러, A2V 커뮤니티가 거대화 된다면 오픈소스 개발자 생태계에서 필수적인 **커뮤니티 활동(세미나, 스터디, 네트워킹, 해커톤 등)을 계획할 예정**이다. 이로써, Valhalla라는 전장에서 개발자들이 겪는 모든 장벽을 찾아 모조리 허물고자 한다.

<br>

### Motivation

*Team Demiguise는 대회를 우승하는 것 이상의, 모두가 함께 미친듯이 성장하는 가치를 만들어가고 싶다.*

> 마치 넷플릭스에서 진행된 추천 알고리즘 대회처럼, 연구자와 개발자가 모두 함께 의견을 공유하며 서로 성장할 수 있는 대회로 만들고 싶다.

우린 아직 초보자에 불과하지만,  많은 개발자들의 작은 생각들이 쌓여 함께 성장한다면 우승을 통해 성장하는 것 보다도 훨신 빠르고 정교하게 성장할 수 있지 않을까.<br>

우리는 그러한 생각을 바탕으로 최대한 가독성을 가진 코드를 작성하여 모든 코드를 공개하고, 설명하면서 진행하고자 한다.



그러니, **이 github을 읽는 모든 도전자와 개발자들에게 어떠한 방향으로든 피드백을 부탁드린다.** 모든 내용이 누군가에게 도움이 되었으면 좋겠다. 

<br>

### Objective

Team Demiguise는 3가지 목표와 지향점을 가지고 시작한다.

```    
1. 모두가 함께함으로써, 수단과 방법을 가리지 않고 최고의 성능을 낼 수 있는 가능한 모든 방법을 총동원
    
2. 누구라도 활용가능 하고, 누구라도 디스 할 수 있는 코드의 설계
    
3. 프로젝트의 진행과 생각의 흐름을 정리하여 재사용성 높은 지식 체계의 구축
```

이 글을 보고 혹시 함께 하고 싶은 개발자는 누구나 위의 오픈카톡방 링크를 통해 함께하길 바란다.(걱정 마라. 익명이다.)<br>

해당 github과 오픈톡방을 중심으로 소통하고, A2V 프로젝트를 계속 이어나가 하나의 지성집단을 만들고자 한다.<br>

__우리는 이 지성집단 내에서 세미나와 스터디, 해커톤, 네트워킹 등 다양한 커뮤니티 활동을 진행할 예정이다.__

종종 연락하고 안부인사 좀 묻고 그런 사이가 좀 되어 보자. <br>

__(아직 프로젝트가 홍보단계에 진입하지 않아, 오픈카톡방에 비밀번호를 걸었습니다. __<br>

__곧 이 코드베이스가 안정되는 순간, 오픈카톡방도 해동할 예정입니다.)__

<br>

### *NOTE

1.  [pep 8style](https://www.python.org/dev/peps/pep-0008/) : 우리의 코드는 기본적으로 pep8 style을 지향한다.<br>
2. 코드 골격을 잡아나가고 있는 단계이기 때문에, 이 리파짓토리의 코드들은 계속 바뀌어 나갈 것이다.<br>
3. 우리는 대회 참가하는 모두가 쉽게 사용할 수 있는 **Data Pipeline**을 구축하는 것을 목표로 하고 있다.<br>
4. 조속히 코드 구조를 잡아나가, 모두가 Demiguise의 코드를 들고 달려들 수 있는 **Valhalla**를 구축하겠다.

<br>

### USAGE

```
# 1. Install package
pip install -r requirements.txt

# 2. Setting Data Structure
root
|- data
	|- raw
		|- init.py
		|- 여기 위치에 카카오의 모든 데이터를 넣어주세요.
		
# 3. Making preprocessed Data
python valhalla/extract.py merge ../data/raw/ ../data/prep/textOnly.h5
```

<br>

### Phase flow

누구라도 **Valhalla**에 뛰어들고, 전장터를 이용하기 위한 Flow manual

이 전장 설계의 흐름만 따라온다면 지금 이를 읽고 있는 자네도 **Valhalla**의 전사가 된다.



#### 1. 데이터 까보기 

>  Kakao-arena의 데이터는 hdf5 포멧으로 train, dev, test 데이터셋으로 제공되었다.
>
>  어떻게 열어 볼까? 대체 어떤 데이터가 주어졌을까? raw data의 확인.



/ PHASE 1. EXTRACT

#### 2. 데이터 모으기 ( Making textOnly.h5 )

Input : 9개로 나누어진 대빵 큰 raw data => output : 12기가짜리 통합 data

> 카카오측에서 제공한 데이터의 크기 : 84.98 GB. -> 데이터가 너무 크다. 
>
> image feature 부분은 제외하고 Local 로 작업 가능하게 줄이고 하나의 파일(textOnly.h5)로 통합해보자.



#### 3.  데이터 로더 만들기 ( DataExtractor )

Input : 제약이 많은 h5 변수

> pandas, numpy는 알아도 h5는 처음보는 모든 이들을 위해.
>
> DataExtractor -> h5를 Extract하긴 하는데, 마치 pandas처럼 쓸 수 있게 해준다.



#### 4. 데이터를 간단히 탐색해보기 ( EDA )

> EDA (Exploratory data analysis)
>
> 이제 데이터를 pandas처럼 편하게 까볼 수 있게 됐다. 한번 보자?
>
> *대회 시작 전, 데이터 내부가 궁금하다면 누구라도 와서 이 script를 먼저 보라.



#### 5. 간단한 데이터 전처리 하기 (1) product & Model (TODO)

>  1. Product column
>  2. Model column
>
>  찐득한 비정형데이터. 이를 어떻게 우리가 써먹을 수 있도록 바꿔볼 수 있을까?



#### 6. 간단한 데이터 전처리 하기 (2) brand & Maker (TODO)
>  1. Brand
>  2. Maker
>
>  여기도 찐득한 비정형데이터다. 더 본격적인 데이터 EDA를 위해 간단히 전처리 해보자



/ PHASE 2. TRANSFORM

#### 7. 본격적인 데이터 EDA 하기 (TODO)

> EDA는 소중하다. 가장 꼼꼼해지는 순간.

/ PHASE 3. LOADER



/ PHASE 4. MODEL

#### 8. 가장 간단한 모델로 돌려보기 (TODO)

> 우리만의 BaseLine 모델을 만들어 보자. 
>
> 첫번째 모델을 구축하기 전까지, 카카오의 Basecode는 읽지 않기로 했다. 생각의 자유를 최대로 존중하기 위해.

<br>

/ PHASE 5. EVALUATE



### REFERENCES

1. pep8 : 



### 기여자

1. rocketgrowthsj : rocketgrowthsj@gmail.com
2. Best10 : best10.csy@gmail.com / +82 10 7242 0548



## [License](https://github.com/Demiguises/Kakao-Valhalla/blob/master/LICENSE)

Apache License 2.0

Copyright 2018 (c) Team Demiguise