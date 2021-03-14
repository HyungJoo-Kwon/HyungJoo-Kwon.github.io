---
layout: page
title: "밑바닥부터 시작하는 딥러닝 24단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 24단계"
categories: deeplearnig
comments: true
published: true
---
# 복잡한 함수의 미분    

```python
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable


def sphere(x, y):
    z = x ** 2 + y ** 2
    return z


def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z


def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z


x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = sphere(x, y)  
z.backward()
print(x.grad, y.grad)

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = matyas(x, y)  
z.backward()
print(x.grad, y.grad)

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = goldstein(x, y)  
z.backward()
print(x.grad, y.grad)
```

    variable(2.0) variable(2.0)
    variable(0.040000000000000036) variable(0.040000000000000036)
    variable(-5376.0) variable(8064.0)
    

![24-1](/assets/images/img/그림 24-1.png)   
최적화 문제의 테스트 함수란 다양한 최적화 기법이 얼마나 좋은가를 평가하는 데에 사용되는 함수.(벤치마크용 함수)

![표 B-1](/assets/images/img/표 B-1.png)


