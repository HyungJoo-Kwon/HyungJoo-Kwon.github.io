---
layout: page
title: "밑바닥부터 시작하는 딥러닝 40단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 40단계"
categories: deeplearnig
comments: true
published: true
---
# 브로드캐스트 함수     

```python
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
```

[브로드캐스트 github](https://github.com/numpy/numpy/blob/v1.19.0/numpy/lib/stride_tricks.py#L141-L180)


```python
# 넘파이 np.broadcast_to(x, shape) : ndarray 인스턴스인 x의 원소를 복제하여 shape 인수로 지정한 형상이 되도록.
x = np.array([1, 2, 3])
y = np.broadcast_to(x, (2,3))
print(y)
print(f'{x.shape} -> {y.shape}')
```

    [[1 2 3]
     [1 2 3]]
    (3,) -> (2, 3)
    

![그림 40-1](/assets/images/img/그림 40-1.png)    
브로드캐스트(원소 복사)가 일어날 경우 역전파에서는 x에 기울기를 두 번 흘려 보내게 되어 기울기가 더해지게 됨.    
브로드캐스트(원소 복사)가 일어날 경우 기울기를 합하면 된다.


```python
# sum_to : x의 원소의 합을 구해 shape 형상을 만들어주는 함수

def sum_to(x, shape):
    """Sum elements along axes to output an array of a given shape.

    Args:
        x (ndarray): Input array.
        shape:

    Returns:
        ndarray: Output array of the shape.
    """
    ndim = len(shape)
    print(f'ndim {ndim}')
    lead = x.ndim - ndim
    print(f'lead {lead}')
    lead_axis = tuple(range(lead))
    print(f'lead_axis {lead_axis}')

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    print(f'axis {axis}')
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)            # 1차원인 축을 제로
    return y

x = np.array([[1, 2, 3], [4, 5, 6]])
y = sum_to(x, (1,3))
print(y)
print(f'{x.shape} -> {y.shape} \n')

y = sum_to(x, (2,1))
print(y)
print(f'{x.shape} -> {y.shape} \n')

# x.sum((0,), keepdims=True) # axis=0 세로
# x.sum((1,), keepdims=True) # axis=1 가로
```

    ndim 2
    lead 0
    lead_axis ()
    axis (0,)
    [[5 7 9]]
    (2, 3) -> (1, 3) 
    
    ndim 2
    lead 0
    lead_axis ()
    axis (1,)
    [[ 6]
     [15]]
    (2, 3) -> (2, 1) 
    
    

![그림 40-2](/assets/images/img/그림 40-2.png)    
x의 형상과 같아지도록 기울기의 원소를 복제


```python
# Dezero version의 sun_to()와 broadcast_to()
# sun_to()와 broadcast_to()는 상호의존적
class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        xp = dezero.cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)
```


```python
class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:  # for broadcaset        #--------------------------------------------
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)       # gx0은 x0, gx1은 x1의 형상이 되도록 sum_to
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)       #--------------------------------------------
        return gx0, gx1
# 순전파는 ndarray 인스턴스를 사용해 구현했기 때문에 variable인스턴스 에서 브로드캐스트가 일어남.
```


```python
import numpy as np
from dezero import Variable

x0 = Variable(np.array([1, 2, 3]))
x1 = Variable(np.array([10]))
y = x0 + x1
print(y)

y.backward()
print(x1.grad)
print(x0.grad)
```

    variable([11 12 13])
    variable([3])
    variable([1 1 1])
    

![그림 40-3](/assets/images/img/그림 40-3.png)


