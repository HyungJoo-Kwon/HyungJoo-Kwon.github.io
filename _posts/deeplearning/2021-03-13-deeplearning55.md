---
layout: page
title: "밑바닥부터 시작하는 딥러닝 55단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 55단계"
categories: deeplearnig
comments: true
published: true
---
# CNN 메커니즘 1    


CNN (Convolutional Nerual Network) 합성곱 신경망의 약자로 이미지 인식, 음성 인식, 자연어 처리 등 다양한 분야에서 사용됨. 이미지 인식용 딥러닝은 대부분 CNN 기반.    
![그림 55-1](/assets/images/img/그림 55-1.png)    
Conv -> ReLU -> (Pool) 순서로 연결됨/ Pool 계층은 생략가능  
출력이 가까워지면 Linear -> ReLU 조합이 사용

![그림 55-2](/assets/images/img/그림 55-2.png)        
 필터를 문헌에 따라 커널이라고도 사용함     
![그림 55-3](/assets/images/img/그림 55-3.png)    
필터가 가로 세로 두 방향으로 이동한다. 이처럼 필터가 두 개의 차원으로 움직여 2차원 합성곱층이라고 함, 세 방향으로 움직이면 3차원 합성곱층.  
![그림 55-4](/assets/images/img/그림 55-4.png)

패딩    
합성곱층의 주요 처리 전에 입력 데이터 주위에 고정값을 채울 수 있으며 이러한 처리를 패딩.    
아래 그림은 (4,4) 입력 데이터에 폭 1짜리 패딩을 적용한 모습     
(세로 방향 패딩과 가로 방향 패딩을 다르게 설정 가능)     
![그림 55-5](/assets/images/img/그림 55-5.png)    
(6, 6) * (3, 3) -> (4, 4)    
패딩을 사용하는 주된 이유는 출력 크기를 조정하기 위함.

스트라이드  
필터를 작용하는 위치의 간격을 스트라이드(보폭).
(세로 방향과 가로 방향 스트라이드 값을 다르게 설정 가능)
![그림 55-6](/assets/images/img/그림 55-6.png)


```python
# 출력 크기 계산 방법
# 패딩 크기를 늘리면 출력 데이터의 크기가 커지고, 스트라이드를 크게하면 출력 데이터의 크기가 작아진다.
def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1

H, W = 4, 4  # Input size
KH, KW = 3, 3  # Kernel size
SH, SW = 1, 1  # Kernel stride
PH, PW = 1, 1  # Padding size

OH = get_conv_outsize(H, KH, SH, PH)
OW = get_conv_outsize(W, KW, SW, PW)
print(OH, OW)
```

    4 4
    


