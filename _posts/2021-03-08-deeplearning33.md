---
layout: page
title: "밑바닥부터 시작하는 딥러닝 33단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 33단계"
categories: deeplearnig
comments: true
published: true
---
# 2차 미분 자동 계산    
```python

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)   # 첫번째 역전파 진행

    gx = x.grad                     # y의 x에 대한 미분값을 꺼냄
    x.cleargrad()
    gx.backward()                   #  gx를 한 번 더 역전파
    gx2 = x.grad

    x.data -= gx.data / gx2.data  # 아래 식
```

    0 variable(2.0)
    1 variable(1.4545454545454546)
    2 variable(1.1510467893775467)
    3 variable(1.0253259289766978)
    4 variable(1.0009084519430513)
    5 variable(1.0000012353089454)
    6 variable(1.000000000002289)
    7 variable(1.0)
    8 variable(1.0)
    9 variable(1.0)
    

![식 33.1](/assets/images/img/식 33.1.png)


