---
layout: page
title: "밑바닥부터 시작하는 딥러닝 35단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 35단계"
categories: deeplearnig
comments: true
published: true
---
# 고차 미분 계산 그래프     
```python
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F
```

![식 35.1](/assets/images/img/식 35.1.png)
![그림 35-1](/assets/images/img/그림 35-1.png)

tanh 함수 미분  
![식 35.3](/assets/images/img/식 35.3.png)


```python
class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)
```


```python
x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

iters = 1

for i in range(iters):              # 2차 미분
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

gx = x.grad
gx.name = 'gx' + str(iters + 1)
plot_dot_graph(gx, verbose=False, to_file='tanh.png')
```




    
![png](/assets/images/output_35_0.png)
    



![그림 35-3](/assets/images/img/그림 35-3.png)
