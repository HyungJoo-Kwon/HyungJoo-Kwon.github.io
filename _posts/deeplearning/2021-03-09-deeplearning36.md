---
layout: page
title: "밑바닥부터 시작하는 딥러닝 36단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 36단계"
categories: deeplearnig
comments: true
published: true
---
# 고차 미분 이외의 용도     

두 식이 주어졌을 때 x = 2.0에서 x에 대한 z의 미분을 구하자.     
![식 36.1](/assets/images/img/식 36.1.png)  
y미분 : 2x  
![식 36.2](/assets/images/img/식 36.2.png)  
z = (2x)^3 + y = 8x^3 + x^2     
z미분 : 24x^2 + 2x


```python
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

x = Variable(np.array(2.0))
y = x ** 2
y.backward(create_graph=True)
gx = x.grad
x.cleargrad()

z = gx ** 3 + y
z.backward()
print(x.grad)
```

    variable(100.0)
    
