---
layout: page
title: "밑바닥부터 시작하는 딥러닝 56단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 56단계"
categories: deeplearnig
comments: true
published: true
---
# CNN 메커니즘 2    

3차원 텐서  
![그림 56-1](/assets/images/img/그림 56-1.png)    
* 주의할 점은 입력 데이터와 필터의 채널 수를 같이 맞춰줘야 한다.

![그림 56-2](/assets/images/img/그림 56-2.png)    
(채널, 높이, 너비), (C, H, W)   
출력은 특징 맵(feature map)라고 불리며 특징 맵이 한장만 출력

![그림 56-3](/assets/images/img/그림 56-3.png)    
![그림 56-4](/assets/images/img/그림 56-4.png)    



미니배치 처리   
![그림 56-5](/assets/images/img/그림 56-5.png)    


풀링층(가로, 세로 공간을 작게...)      
2x2 최대 풀링(Max polling) : 2x2는 대상 영역의 크기를 나타냄.  (일반적으로 풀링 크기와 스트라이드 크기는 값은 값으로 설정)      
![그림 56-6](/assets/images/img/그림 56-6.png)    
Max 풀링 외에 평균 풀링(Average 풀링)이 있지만 주리 Max Polling 사용

풀링층의 주요 특징      
    - 학습하는 매개변수가 없다.     
    - 채널 수가 변하지 않는다. ![그림 56-7](/assets/images/img/그림 56-7.png)     
    - 미세한 위치 변화에 영향을 덜 받는다. ![그림 56-8](/assets/images/img/그림 56-8.png)    


