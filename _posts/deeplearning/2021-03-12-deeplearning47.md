---
layout: page
title: "밑바닥부터 시작하는 딥러닝 47단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 47단계"
categories: deeplearnig
comments: true
published: true
---
# 소프트맥스 함수와 교차 엔트로피 오차      

```python
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
```


```python
import numpy as np 
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.get_item(x, 1)
y.backward()
print(y)
print(x.grad)

indices = np.array([0, 0, 1])
y = F.get_item(x, indices)
print(y)
```

    variable([4 5 6])
    variable([[0 0 0]
              [1 1 1]])
    variable([[1 2 3]
              [1 2 3]
              [4 5 6]])
    

get_item()은 Variable의 다차원 배열 중에서 일부를 슬라이스하여 뽑아준다.        
![그림 47-1](/assets/images/img/그림 47-1.png)


```python
Variable.__getitem__ = F.get_item       # Variable의 메서드로 설정

y = x[1]
print(y)

y = x[:,2]
print(y)
```

    variable([4 5 6])
    variable([3 6])
    

소프트맥스 함수     
![식 47.1](/assets/images/img/식 47.1.png)
![그림 47-2](/assets/images/img/그림 47-2.png)    
각 행의 원소들은 0이상 1이하이고, 총합은 1이 된다.

선형 회귀에서는 손실 함수로 평균 제곱 오차를 이용했지만 다중 클래스 분류에는 교차 엔트로피 오차를 손실 함수로 사용.     
![식 47.2](/assets/images/img/식 47.2.png)      
정답데이터의 각 원소는 정답에 해당하는 클래스면 1로, 그렇지 않으면 0으로 기록됨. = (**원핫 벡터**)



```python
# 소프트맥스 함수와 교차 엔트로피 오차를 한꺼번에 수행하는 함수
def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]
    p = softmax(x)
    p = clip(p, 1e-15, 1.0)  # To avoid log(0), 0을 1e-15라는 작은 값으로 대체
                              # clip(x, x_min, x_max) : x가 x_min 이하면 x_min으로 변환, x_max 이상이면 x_max로 변환
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]  # log_p[0, t.data[0]], log_p[1, t.data[1]], ... 
    y = -1 * sum(tlog_p) / N
    return y
```


```python
import numpy as np
np.random.seed(0)
from dezero import Variable, as_variable
import dezero.functions as F
from dezero.models import MLP


def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y


model = MLP((10, 3))

x = Variable(np.array([[0.2, -0.4]]))
y = model(x)
print(y, '\n')
p = softmax1d(y)
print(y, '\n')
print(p, '\n')


# 입력데이터 x와 정답 데이터 t
x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])

y = model(x)
p = F.softmax_simple(y)
print(y, '\n')
print(p, '\n')

loss = F.softmax_cross_entropy_simple(y, t)
loss.backward()
print(loss, '\n')
```

    variable([[-0.61505778 -0.42790161  0.31733289]]) 
    
    variable([[-0.61505778 -0.42790161  0.31733289]]) 
    
    variable([[0.21068639 0.25404893 0.53526469]]) 
    
    variable([[-0.61505778 -0.42790161  0.31733289]
              [-0.76395313 -0.2497645   0.18591382]
              [-0.52006391 -0.96254612  0.57818938]
              [-0.94252164 -0.50307479  0.17576323]]) 
    
    variable([[0.21068639 0.25404893 0.53526469]
              [0.19019916 0.31806646 0.49173438]
              [0.21545395 0.13841619 0.64612986]
              [0.17820704 0.27655034 0.54524263]]) 
    
    variable(1.496744252405306) 
    
