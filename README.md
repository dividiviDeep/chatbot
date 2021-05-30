# Emotion chatbot
> **DiviDiviDeep Learning**    팀의 감정인식 기반 챗봇 웹


### 📘 Members  

| 김민재  | 김영은 | 박가연 | 이승미 | 이화경|
***

### ✨ 감정 대응형 챗봇

<br>
    ✔ 대화를 자연어 처리하고 의도 및 감정 분석을 통하여 이에 대한 적절한 반응을 할 수 있는 감정 대응형 챗봇 시스템을 설계 및 개발  
   
   <세부목표>  
   - 주어진 대화 자연어처리 가능
   - 대화를 6가지 감정(**기쁨,슬픔,공포,분노,놀람,중립**) 중 하나의 감정으로 분류
   - 분류된 감정에 알맞은 답변 생성
   - 감정 인식률 70% 이상의 감정 엔진 개발  

<br>

***

### ✨ 시스템 개요도
![image](https://user-images.githubusercontent.com/55472510/118766324-3c973300-b8b7-11eb-999a-c30a47127ef9.png)

   ✔ 감정 대응형 챗봇 시스템은 감정 엔진과 답변 생성기로 이루어져 있다. 사용자로부터 웹을 통해 텍스트를 입력받으면 각 감정 분류 모델과 답변 생성 모델에 대화가 전달되도록 한다. 감정 분류 모델에서는 KoBERT를 통해 해당 대화에 대한 감성 분류와 분류된 감성 범주에 해당하는 감정 분류를 수행한다. 답변 생성 모델에서는 감정에 공감하는 답변을 출력하고 KoGPT-2모델을 통해 생성된 적절한 답변을 다시 사용자에게 전달한다.
<br>  

***

### ✨ 시스템 설계도
![image](https://user-images.githubusercontent.com/55472510/118766936-0dcd8c80-b8b8-11eb-9c66-27b6e15f785f.png)  

   ✔ 시스템 설계도는 크게 DATA, AI, INFRA 플랫폼으로 구성된다.  
   ##### ⓵ DATA 플랫폼  
   	- 모델을 학습시키기 위한 데이터셋 전처리 및 정제  
	- 학습에 이용하는 데이터셋: 생성 모델 학습에 사용되는 Dialog 셋, 감정분류 모델 학습에 사용되는 Sentence&Emotion 셋   
   ##### ⓶ AI 플랫폼  
   	- 감정 분류 모델 : KoBERT → 감정 분류  
	- 답변 생성 모델 : KoGPT2 → 답변 생성  
	- 위의 2가지 모델을 사용하여 웹애플리케이션을 통한 생성된 답변 출력  
   ##### ⓷ INFRA 플랫폼
   	- 각 모델은 Flask에서 동작  
	- 소켓 통신을 통해 감정 분류 결과 전달 및 클라이언트에 전달
***
### ✨ 감정 분석에 사용한 모델 정확도
- 데이터셋을 이용하여 학습시킨 감성분석 모델을 이용한 경우: 68.05
- 감성분석을 수행하지 않은 경우: 67.12
<br>

***
### ✨ 웹 구현 화면
![image](https://user-images.githubusercontent.com/55472510/118767282-7e74a900-b8b8-11eb-9843-7c13a4d254d4.png)



***
### ✨ 실행 방법
	1. server Ip 주소를 변경하고 serv.py파일을 실행시킨다.
	2. client Ip 주소를 변경하고 cli.py파일을 실행시킨다.
	3. url에 client Ip 주소:5000를 입력하면 웹 화면이 실행된다. 
	- server측에 `감정 인식 모델`의 결과 checkpoint와 model이 있어야함
	- client측에 `답변 생성 모델`의 결과 checkpoint와 model이 있어야함


### 📙 사용된 라이브러리 및 언어

* [Flask]
* [Socket]
* [Html,Css,Javascript]
* [Python]

### 📘 사용 학습 데이터 셋
- [Isear](https://www.unige.ch/cisa/research/materials-and-online-research/research-material/)
- [DailyDialog](http://yanran.li/dailydialog)
- [Chatbot](https://github.com/songys/Chatbot_data)
- [한국어 감정 정보가 포함된 단발성 대화 데이터셋](https://aihub.or.kr/keti_data_board/language_intelligence)
<br>

### 📕 실행 동영상 링크
- [시연 영상](https://drive.google.com/file/d/1kuppVDUxszylBvjo82p-11BVlAfUj743/view?usp=sharing) 

- [발표 영상](https://drive.google.com/file/d/1Vuq2B5zdn3ZyobG19naZX088aSy71bS9/view?usp=sharing)

<br>

### 📘 참고 Github 주소
- [답변 생성 모델](https://github.com/nawnoes/WellnessConversation-LanguageModel)
- [감정 인식 모델](https://github.com/SKTBrain/KoBERT)

### 📙 개발 기간
2020/08 ~ 2021/05
	 

