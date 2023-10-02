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

### :pencil2:[학생들이 실습을 진행할 COLOAB 파일 예시] 
<br/> ![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/f1a27653-8b5c-40b8-a5c6-8b894888335d)

### :pencil2:[이해하기 쉬운 설명 방식] 
<br/> ![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/52bde2fe-f58d-402f-b16b-00694bca4b34) ![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/54b804b9-ebf9-482f-b009-9f2a560d918b)   
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/83e127e8-3ace-4a64-8db9-08ce979b7611)
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/b32a471c-29b1-404b-b12f-2b0b08d8da17)   
학생들이 실습을 쉽게 이해하도록 그림을 통해 태스크를 도식화하고 질문들을 만들어 학생들의 주의를 상기시키도록 노력했다. 특히 학생들은 머신러닝과 데이터 분석에 대해 처음 접하기 때문에 최대한 간단하게 프로세스에 대해 이해시키려고 했으며 전처리와 모델링 그리고 출력값 이렇게 크게 3가지 단계로 나눠서 설명했다.

### :pencil2:[전처리 파트 - 데이터확인, 이상치 처리, 결측치 처리, 변수변환]
<br/> ![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/75567b00-2aae-47a4-b318-210c0a42a9ca) ![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/58af1b77-f872-428e-a750-b6666abe4a4f) ![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/d514031d-7c4e-445b-9e46-103d7a229d12)
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/31ed5606-e43c-4c8a-bc8c-51b39f431e40)
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/a7bcb788-35ee-49c8-a1b3-0ce7c9e46af5)
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/45d0374a-c87d-4441-8523-603240d7927a)   
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/26ca7342-cb33-4760-a079-d253c66c3eb3)
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/2780d9f0-8980-4132-a927-17117876c27a)
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/0d75941a-c6a6-4538-9161-72fdf7a4077a)

세부적으로 전처리 단계에서 필요한 데이터 확인, 결측치 제거, 이상치 제거, 변수 변환 기법에 대해 설명한다. 중요한 부분에 대해 적절한 질문을 통해 학생들의 주의를 환기시키고자 했다. 결측치 확인 과정에서는 결측치 처리 기법으로 크게 제거와 중앙값으로 대체라는 방법을 소개했고, 이상치 확인 과정에서는 boxplot 그래프를 그리는 방법을 간단하게 소개했다. 변수 변환 기법에서는 변수변환이 필요한 이유들을 쉬운 예시를 들어서 설명을 하고 그래프를 그려 독립변수들의 분포를 확인하는 과정을 진행했다. 그리고 왜 변수변환기법에서 로그변환이 쓰이는지도 학생들의 눈높이에 맞춰 설명을 진행했다.   

### :pencil2:[회귀 모델링 파트]
###:pencil:데이터탐색 (데이터 생성, 상관관계 확인)
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/864a9f9c-7438-491b-9f00-6b4e560f083c) ![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/f316db49-c593-4000-af44-b2d5fea96808)   
우선적으로 주택 가격을 예측하기 위해서는 회귀 모델링을 해야한다는 것을 학생들에게 다시한번 상기시켰다. 그리고 회귀모델링 과정인 데이터 탐색 부분에서 데이터를 생성하고 상관관계를 확인하는 파트에 대해 설명하며 각 과정의 중요성과 목적에 대해 설명을 진행했다.

###:pencil:분석 모형 구축 (데이터 분할, 데이터 스케일링)
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/58e9401b-7681-4d9f-a986-989bbe144c20) ![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/1be5cc9d-ac67-452d-9d97-a4001c9cb912)   
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/bd969fa4-e69c-4cd2-a2bb-970302b3c8a9)![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/03577f22-e2b8-40ba-9f91-832969546c41)   
분석모형을 구축할 때 왜 데이터를 분할해야하는지에 대한 내용을 설명하고, 이때 사용하는 라이브러리에 대한 설명을 진행했다. 더불어서 데이터 스케일링을 해야하는 이유와 이때 사용하게 되는 라이브러리에 대한 설명도 진행했다.   

###:pencil:분석 모형 구축 (모델 구축)   
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/20810d3b-8100-4c84-a8b3-5423584ccb8f)   
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/1dfbbcb2-85bd-4d14-8906-43aa670141d4) ![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/83b2a097-6695-41ac-a194-fb90d38ac5ed)   
주택가격 예측에 대한 분석모형을 만드는데 사용한 선형회귀와 svm 그리고 랜덤포레스트 알고리즘에 대해 설명을 진행했다. 그리고 위 알고리즘을 통해 처음 생성시킨 모델은 박스와 다름 없기 때문에 학습을 진행해야 한다고 설명했다. 이에 전에 구축한 트레인 데이터를 통해 모델을 학습하는 과정에 대해 설명했다. 더불어 해당 분석 모델들은 모두 지도 할습 알고리즘이기 때문에 비지도 학습 알고리즘 또한 존재한다는 점과 어떠한 것들이 있는지 설명했다. 
그리고 실무에서는 각 모델별 하이퍼 파라미터 튜닝을 통해 성능을 올리는 작업을 하게 됨다는 것을 설명하고 하이퍼파라미터란 모델의 학습 과정을 제어하고 조절하는 매개변수로, 사용자가 직접 설정해야 하는 값이라고 예시를 들며 설명한다. 이때 학생들에게 우리는 전체적인 분석 과정에 대해 익히는 것이 중요하기 때문에 실무에서는 가장 중요한 것으로 인지되는 하이퍼파라미터 튜닝을 하지 않고 default모델을 사용한다고 말한다.   

###:pencil:분석 모형 평가   
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/9d0199ce-ccaf-4b61-908b-29c731c76140)   
앞에서 만들었던 모델에 test 데이터를 넣어서 예측값을 예측할 수 있다는 점을 설명하고 해당 예측값이 우리의 최종 목표의 주택 가격이라는 점을 설명했다. 해당 예측값과 실제값을 MSE라는 평가지표를 이용해서 어떻게 성능평가를 진행할 수 있는지 보여줬다. 그리고 우리가 최종적으로 선택하게 될 모델은 오차가 제일 적게 나온 랜덤포레스트 모델이라는 것을 설명했다.

### :pencil2: [분류 모델링 파트]
###:pencil:데이터탐색, 분석모형구축   
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/c4004257-4776-485b-b8aa-8f03b2525a5b)![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/78c06350-a47a-46ad-bbf4-59002e560457) ![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/c14c9d99-2ff3-44df-8bcf-e9f59ae10d22)   
회귀 모델링에서 설명했던 부분과 유사하다고 하며 설명을 진행했다. 회귀 모델링에서는 왜 이 과정을 진행하는지에 초점을 맞춰서 설명했다면, 이번에는 사용하는 라이브러리와 코드에 대한 설명을 위주로 진행했다. 이때 학생들이 자신이 어떤 테스크를 이행하고 있는지 헷갈릴 수 있기 때문에 다시한번 도식화된 그림으로 분류모델링을 통해 주택의 고가여부를 판단한다는 사실을 상기시켜줬다.

###:pencil:분석 모형 구축 (모델 구축)   
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/d1de8a00-1c55-4aeb-8b2f-2abd716e59f7)   
주택가격 분류에 대한 분석모형을 만드는데 사용한 로지스틱회귀와 SVM 그리고 랜덤포레스트 알고리즘에 대해 설명을 진행했다. 그리고 위 알고리즘을 통해 처음 생성시킨 모델은 박스와 다름 없기 때문에 학습을 진행해야 한다고 설명했다. 이에 전에 구축한 트레인 데이터를 통해 모델을 학습하는 과정에 대해 다시한번 설명했다. 
이때 주의할 점으로 회귀모델링과 분류모델링에서 동시에 설명한 랜덤포레스트 이름의 차이를 설명했다. 예를 들어 회귀모델링은 RandomForestRegressor/ 분류모델링 RandomForestClassifier 그리고SVM 알고리즘은 회귀모델링 SVR / 분류모델링 SVC 라고 설명했다.   

### :pencil:분석 모형 평가   
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/cdefdef6-1c86-4de0-a423-21cf87cb7acf)   
아까 만든 모델에 스케일링을 끝낸 테스트 데이터를 넣어 예측값을 구하게 구라고 예측 값이 0.5 이상이면 1, 아니면 0으로 즉 주택을 고가, 저가로 분류하게 된다는 것을 설명했다. 모델의 성능을 측정하기 위해서는 sklearn 패키지의 classification_report 함수를 통해 평가 지표인 accuracy, precision, recall등이 이용한다는 것을 설명했다. 이때 F1 score이 제일 중요하고 이는 모델의 정밀도(Precision)와 재현율(Recall)을 조합으로 제일 보편적인 성능지표이며 1에 가까울수록 성능이 좋다는 것을 설명했다.   
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/baa29d7f-a651-472e-b39c-b08182c48391)

### :pencil2: [총정리]   
![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/27b3c7b5-b789-471e-862e-f0469088eb2e) ![image](https://github.com/nanhungrybin/passionpay_volunteer_/assets/97181397/da375659-25eb-4f05-bc4d-156041357498)   
전체적인 데이터 분석 프로세스는 데이터 전처리(확인, 결측치 및 이상치 처리, 변수변환) → 모델링(탐색, 분석모형 구축, 분석모형 평가)라는 점을 다시 한번 설명해주고 오늘 우리가 한 테스크가는 주택 가격을 예측하고, 고가 인지 저가인지 판단했다는 것을 설명해줬다. 더불어 오늘 우리가 진행한 회귀 모델링, 분류 모델링에 따른 방법론 들과 이에 대한 특징들에 대해서도 표로 정리해서 다시 한번 상기시켜 줬다.

















