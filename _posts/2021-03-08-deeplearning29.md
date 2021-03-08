---
layout: page
title: "밑바닥부터 시작하는 딥러닝 29단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 29단계"
categories: deeplearnig
comments: true
published: true
---
# 뉴턴 방법으로 푸는 최적화     

```python
# 이전 단계에서 경사하강법으로 5만 번 가까이 반복해야 목적지에 도달. 수렴이 빠른 뉴턴 방법 이용 시 단 6번만에 도달.
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
import numpy as np
from dezero import Variable
# import dezero's simple_core explicitly
import dezero
if not dezero.is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import setup_variable
    setup_variable()
```

뉴턴 방법은 추가된 2차 미분 정보로 효율적인 탐색을 기대할 수 있으며 목적지에 더 빨리 도달   
![그림 29-1](/assets/images/img/그림 29-1.png)


```python
# 2차 미분을 수동 코딩 함.
def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


def gx2(x):
    return 12 * x ** 2 - 4


x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward()

    x.data -= x.grad / gx2(x.data)
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
    

![그림 29-5](/assets/images/img/그림 29-5.png)


