---
layout: page
title: "밑바닥부터 시작하는 딥러닝 41단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 41단계"
categories: deeplearnig
comments: true
published: true
---
# 행렬의 곱     

```python
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


```

벡터의 내적     
a = $(a_{1}, \cdots, a_{n})$, b = $(b_{1}, \cdots, b_{n})$  
![식 41.1](/assets/images/img/그림 41.1.png)      

행렬의 곱   
![그림 41-1](/assets/images/img/그림 41-1.png)


```python
import numpy as np 
a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.dot(a,b)
print(c)

a = np.array([[1,2], [3,4]])
b = np.array([[5,6], [7,8]])
c = np.dot(a,b)
print(c)
```

    32
    [[19 22]
     [43 50]]
    

![그림 41-2](/assets/images/img/그림 41-2.png)

# 행렬 곱
y = xW (x = 1 x D, W = D x H, y = 1 x H)    
![그림 41-3](/assets/images/img/그림 41-3.png)    
y = xW (x = N x D, W = D x H, y = N x H)   
![그림 41-5](/assets/images/img/그림 41-5.png)


```python
class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)            # W.T와 x.T에는 transpose 함수가 호출(step38)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    return MatMul()(x, W)

```


```python
import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.random.randn(2, 3))
w = Variable(np.random.randn(3, 4))
y = F.matmul(x, w)
print(f'x : {x} {x.shape}')
print(f'w : {w} {w.shape}')
print(f'y : {y} {y.shape}')
y.backward()

print(f'x.grad.shape {x.grad.shape}')
print(f'w.grad.shape {w.grad.shape}')
```

    x : variable([[ 1.03456353  0.27193871  0.86023841]
              [ 1.06768801 -0.83540155  1.35419697]]) (2, 3)
    w : variable([[-0.77110712  0.31744908  1.4434869   0.81484399]
              [-0.59206421 -0.47752476  0.42763204  0.84477188]
              [-0.82779724  1.08705633  0.90385875  2.0810289 ]]) (3, 4)
    y : variable([[-1.67086747  1.13369138  2.38720262  2.86291505]
              [-1.44969098  2.20994988  2.40795197  2.98239845]]) (2, 4)
    x.grad.shape (2, 3)
    w.grad.shape (3, 4)