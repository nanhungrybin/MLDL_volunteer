# passionpay_volunteer_
2023 청원고등학교 ML/DL 과정 봉사 <3일차 과정>을 위해 직접 만든 교안
* * *

#### :fire:담당 교육 제목 : 
머신러닝 방법론에 대한 소개와 전체적인 데이터 분석 프로세스에 대한 실습

#### :house:실습 주제 : 
주택 가격 예측하기와 주택 가격의 고가 여부 분류하기

![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/794f368e-389e-4864-8009-d6f64c21f891)

학생들이 머신러닝의 방법론들과 전체적인 데이터 분석 프로세스에 대해 경험해보고자 하도록 수업을 구성하고 자료를 제작했다. 하나의 housing 데이터 셋을 가공하여 머신러닝의 주요 모델인 회귀모델과 분류모델을 모두 사용하고자 하도록 실습을 진행했다. 이때 학생들이 이해하기 쉽도록 ppt에는 세분화된 설명을 작성했고 코드는 google colab 환경에서의 학생들이 직접 실행할 수 있도록 구축했다. 
* * *

### [학생들이 실습을 진행할 COLOAB 파일 예시] 
<br/> ![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/f1a27653-8b5c-40b8-a5c6-8b894888335d)

### [이해하기 쉬운 설명 방식] 
<br/> ![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/52bde2fe-f58d-402f-b16b-00694bca4b34) ![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/54b804b9-ebf9-482f-b009-9f2a560d918b)   
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/83e127e8-3ace-4a64-8db9-08ce979b7611)
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/b32a471c-29b1-404b-b12f-2b0b08d8da17)   
학생들이 실습을 쉽게 이해하도록 그림을 통해 태스크를 도식화하고 질문들을 만들어 학생들의 주의를 상기시키도록 노력했다. 특히 학생들은 머신러닝과 데이터 분석에 대해 처음 접하기 때문에 최대한 간단하게 프로세스에 대해 이해시키려고 했으며 전처리와 모델링 그리고 출력값 이렇게 크게 3가지 단계로 나눠서 설명했다.

### [전처리 파트 - 데이터확인, 이상치 처리, 결측치 처리, 변수변환]
<br/> ![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/75567b00-2aae-47a4-b318-210c0a42a9ca) ![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/58af1b77-f872-428e-a750-b6666abe4a4f) ![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/d514031d-7c4e-445b-9e46-103d7a229d12)
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/31ed5606-e43c-4c8a-bc8c-51b39f431e40)
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/a7bcb788-35ee-49c8-a1b3-0ce7c9e46af5)
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/45d0374a-c87d-4441-8523-603240d7927a)   
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/26ca7342-cb33-4760-a079-d253c66c3eb3)
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/2780d9f0-8980-4132-a927-17117876c27a)
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/0d75941a-c6a6-4538-9161-72fdf7a4077a)

세부적으로 전처리 단계에서 필요한 데이터 확인, 결측치 제거, 이상치 제거, 변수 변환 기법에 대해 설명한다. 중요한 부분에 대해 적절한 질문을 통해 학생들의 주의를 환기시키고자 했다. 결측치 확인 과정에서는 결측치 처리 기법으로 크게 제거와 중앙값으로 대체라는 방법을 소개했고, 이상치 확인 과정에서는 boxplot 그래프를 그리는 방법을 간단하게 소개했다. 변수 변환 기법에서는 변수변환이 필요한 이유들을 쉬운 예시를 들어서 설명을 하고 그래프를 그려 독립변수들의 분포를 확인하는 과정을 진행했다. 그리고 왜 변수변환기법에서 로그변환이 쓰이는지도 학생들의 눈높이에 맞춰 설명을 진행했다.   

### [회귀 모델링 파트]
### - 데이터탐색 (데이터 생성, 상관관계 확인)
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/864a9f9c-7438-491b-9f00-6b4e560f083c)










