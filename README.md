## 강화학습(Actor-Critic Network)을 기반으로 하는 추천 시스템 


## Team 

- 최재희
- 김강민
- 오창환
- 이휘준
- 이은성

## AgileSoDa 제안
- user의 과거 기록들을 이용해 하나의 카테고리(Book)에서의 아이템들 Rangking을 주어 Top K개를 추천, user의 구매를 유도하는 알고리즘

- State: User가 과거에 관심을 보였던 n개의 items 및 item에 대한 정보
- Action: K개의 items으로 이루어진 item list(user에게 추천)
- Reward: Action에 대해 user가 남긴 rating
- Input > Output: 어떤 state를 보고 적절한 추천 list action을 결정

- Data: 각 user별로 남긴 평점을 시간순으로 정렬해 state, action, reward로 이루어진 train data 생성해서 사용

### 참고 논문
- Deep Neural Network for YouTube recommendation 
- [Deep reinforcement learning based recommendation with Explicit User-item Interaction Modeling](https://arxiv.org/pdf/1810.12027.pdf)
- [Top-K off-Policy Correction for a REINFORCE Recommender System](https://arxiv.org/pdf/1812.02353.pdf)
- [Deep Reinforcement Learning for List-wise Recommendations](https://arxiv.org/abs/1801.00209)

### 참고 코드
- https://github.com/luozachary/drl-rec
- https://github.com/egipcy/LIRD
- https://github.com/shashist/recsys-rl/blob/274341bc867ee81eeb14177ed79a14fe578464cd/recsys_rl.ipynb


## MileStone

### 이론
- T-Academy
  - [10.23](https://tacademy.skplanet.com/live/player/onlineLectureDetail.action?seq=163) 
  - [10.30](https://tacademy.skplanet.com/live/player/onlineLectureDetail.action?seq=170#sec3)

- 참고 도서
  - 파이썬과 케라스로 배우는 강화학습
  - 바닥부터 배우는 강화학습(torch)
  - 수학으로 풀어보는 강화학습 원리와 알고리즘

- 참고 사이트
  - [Wide & Deep Learning for Recommender System](https://soobarkbar.tistory.com/131)
  - [Deep Neural Networks for YouTube Recommendations 요약](http://keunwoochoi.blogspot.com/2016/09/deep-neural-networks-for-youtube.html)
  - [강화학습 알아보기 - Actor-Critic, A2C, A3C](https://greentec.github.io/reinforcement-learning-fourth/)
  - [논문 모음](https://github.com/guyulongcs/Deep-Reinforcement-Learning-for-Recommender-Systems)
 
 - 논문
 

### 코드리뷰
1. LIRD : https://github.com/egipcy/LIRD/blob/master/LIRD.ipynb
2. DRR : 
  1)https://github.com/shashist/recsys-rl/tree/274341bc867ee81eeb14177ed79a14fe578464cd
  2)https://github.com/bcsrn/RL_DDPG_Recommendation

### 구현
- embedding by MF or Auto-Encoder with Aamazon book data
- recsys-rl 정확도 개선

### 평가

## Dataset

- [Amazon: Book.csv](https://nijianmo.github.io/amazon/index.html#subsets)
  - ratings only(51,311,621): item, user, rating, timestamp
  - matadata: asin, title, feature, description, price, image, related, salesRank, rand, categories, tech1, tech2, similar
  
### 원본 데이터
- 총 개수: 51311621
- 유저수: 15,362,619
- 아이템수: 2,930,451

### rating 20 이상 데이터
- 총 개수: 15,731,887
- 유저수: 301,567
- 아이템수: 1,615,039
  
## Branch 관리
