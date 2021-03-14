---
layout: page
title: "밑바닥부터 시작하는 딥러닝 39단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 39단계"
categories: deeplearnig
comments: true
published: true
---
# 합계 함수     

```python
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
```

![그림 39-1](/assets/images/img/그림 39-1.png)    
![그림 39-2](/assets/images/img/그림 39-2.png)
![그림 39-3](/assets/images/img/그림 39-3.png)    
기울기를 입력 변수의 형상과 같아지도록 복사


```python
class Sum(Function):
    
    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum()
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x):
    return Sum()(x)
```


```python
import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([1, 2, 3, 4, 5, 6]))
y = F.sum(x)
y.backward()

print(y)
print(x.grad)
```

    variable(21)
    variable([1 1 1 1 1 1])
    


```python
x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad)
```

    variable(21)
    variable([[1 1 1]
              [1 1 1]])
    


```python
x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.sum(x, axis=0)
y.backward()
print(y)
print(x.grad)
print(x.shape, '->', y.shape)

y = F.sum(x, axis=(0,1))
print(y)
```

    variable([5 7 9])
    variable([[1 1 1]
              [1 1 1]])
    (2, 3) -> (3,)
    variable(21)
    

![그림 39-4](/assets/images/img/그림 39-4.png)
![그림 39-5](/assets/images/img/그림 39-5.png)    
axis=None : 모든 원소를 다 더한 값 하나(스칼라)를 출력(default임)   
axis=(0,2) : 튜플로 지정하면 해당 튜플에서 지정한 0번과 2번 축 모두 대해 합계


```python
x = np.array([[1,2,3], [4,5,6]])
y = np.sum(x, keepdims=True)        # keepdims는 입력과 출력의 차원 수를 똑같게 유지할지 정하는 플래그
print(y)                            # keepdims=True : 축의 수를 유지
print(y.shape)    
y = np.sum(x, keepdims=False)        # keepdims=False : 형상은 (), 스칼라
print(y)                              
print(y.shape)    
```

    [[21]]
    (1, 1)
    21
    ()
    


```python
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis,
                                        self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)
```


```python
x = Variable(np.random.randn(2, 3, 4, 5))
y = x.sum(keepdims=True)
print(x.shape)
print(y)
print(y.shape)
```

    (2, 3, 4, 5)
    variable([[[[14.0856125]]]])
    (1, 1, 1, 1)