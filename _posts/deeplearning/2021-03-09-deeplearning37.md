---
layout: page
title: "밑바닥부터 시작하는 딥러닝 37단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 37단계"
categories: deeplearnig
comments: true
published: true
---
# 텐서      

```python
import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array(1.0))     # 스칼라 (0차원 ndarray)
y = F.sin(x)
print(y)
```

    variable(0.8414709848078965)
    


```python
x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.sin(x)
print(y)                    # x의 원소 각각에 적용
```

    variable([[ 0.84147098  0.90929743  0.14112001]
              [-0.7568025  -0.95892427 -0.2794155 ]])
    


```python
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
c = Variable(np.array([[10, 20, 30], [40, 50, 60]]))
t = x + c
y = F.sum(t)

y.backward(retain_grad=True)
print(t)
print(y)
print(y.grad)
print(t.grad)
print(x.grad)
print(c.grad)
# x, c, t의 데이터와 기울기(미분값)의 shape이 같음.
```

    variable([[11 22 33]
              [44 55 66]])
    variable(231)
    variable(1)
    variable([[1 1 1]
              [1 1 1]])
    variable([[1 1 1]
              [1 1 1]])
    variable([[1 1 1]
              [1 1 1]])
    

![그림 37-1](/assets/images/img/그림 37-1.png)

브로드캐스트 - 만약 x와 c의 형상이 다르면 자동으로 데이터를 복사하여 같은 형상의 텐서로 변환해주는 기능

텐서 사용 시의 역전파 - 데이터가 텐서일 경우에는 전처리로 **벡터화 처리(원소를 1열로 정렬하는 형상 변환 처리)**     

y = F(x)에서 y와 x 모두 벡터이므로 미분은 행렬의 형태가 됨. 이 행렬은 **야코비 행렬**이라고 함. 

![그림 37-2](/assets/images/img/그림 37-2.png)
$\frac{\partial b}{\partial a} \frac{\partial a}{\partial x}$ -> n x n 행렬
![그림 37-3](/assets/images/img/그림 37-3.png)
$\frac{\partial y}{\partial b} \frac{\partial b}{\partial a}$ -> 1 x n 행렬     

역전파 수행 시 야코비 행렬을 구하면 행렬의 곱이 사용 되지 않는다. $\frac{\partial a}{\partial x}$는 대각 행렬이기 때문
즉, 원소별 연산에서는 역전파도 미분을 원소별로 곱하야 구함.

