---
layout: page
title: "밑바닥부터 시작하는 딥러닝 38단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 38단계"
categories: deeplearnig
comments: true
published: true
---
# 형상 변환 함수 (reshape, transpose)       

```python
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[0, 1, 2], [3, 4, 5]]))
y = F.reshape(x, (6,))  # y = x.reshape(6)
print(y)
y.backward(retain_grad=True)
print(x.grad)


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.transpose(x)  # y = x.T
y.backward()
print(x.grad)
```

    variable([0 1 2 3 4 5])
    variable([[1 1 1]
              [1 1 1]])
    variable([[1 1 1]
              [1 1 1]])
    

![그림 38-1](/assets/images/img/그림 38-1.png)


```python
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape          # 역전파에 사용할 shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):             # gy는 Variable 인스턴스
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)
```

![그림 38-2](/assets/images/img/그림 38-2.png)    
data와 grad의 형상이 일치


```python
x = np.random.rand(1,2,3)
print(x)
y = x.reshape((2,3))    # 튜플로
print(f'{y} {type(y)}')
y = x.reshape([2,3])    # 리스트로
print(f'{y} {type(y)}')
y = x.reshape(2,3)      # 인수 그대로
print(f'{y} {type(y)}')
```

    [[[0.20790506 0.70694382 0.825933  ]
      [0.05052655 0.06452241 0.60763724]]]
    [[0.20790506 0.70694382 0.825933  ]
     [0.05052655 0.06452241 0.60763724]] <class 'numpy.ndarray'>
    [[0.20790506 0.70694382 0.825933  ]
     [0.05052655 0.06452241 0.60763724]] <class 'numpy.ndarray'>
    [[0.20790506 0.70694382 0.825933  ]
     [0.05052655 0.06452241 0.60763724]] <class 'numpy.ndarray'>
    


```python
class Variable:
    ...

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)
```


```python
x = Variable(np.random.rand(1,2,3))
print(x)
y = x.reshape((2,3))    # 튜플로
print(f'{y} {type(y)}')
y = x.reshape([2,3])    # 리스트로
print(f'{y} {type(y)}')
y = x.reshape(2,3)      # 인수 그대로
print(f'{y} {type(y)}')
```

    variable([[[0.15225959 0.82392413 0.25257209]
               [0.97473538 0.27579144 0.91532736]]])
    variable([[0.15225959 0.82392413 0.25257209]
              [0.97473538 0.27579144 0.91532736]]) <class 'dezero.core.Variable'>
    variable([[0.15225959 0.82392413 0.25257209]
              [0.97473538 0.27579144 0.91532736]]) <class 'dezero.core.Variable'>
    variable([[0.15225959 0.82392413 0.25257209]
              [0.97473538 0.27579144 0.91532736]]) <class 'dezero.core.Variable'>
    

행렬의 전치     
![그림 38-3](/assets/images/img/그림 38-3.png)


```python
class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)
```


```python
x = Variable(np.array([[1,2,3], [4,5,6]]))
y = F.transpose(x)
y.backward()
print(y)
print(x.grad)
```

    variable([[1 4]
              [2 5]
              [3 6]])
    variable([[1 1 1]
              [1 1 1]])
    


```python
class Variable:
    ...

    def transpose(self):
            return dezero.functions.transpose(self, axes)

        @property
        def T(self):
            return dezero.functions.transpose(self)
```


```python
x = Variable(np.random.rand(2,3))
y = x.transpose()
print(y)
y = x.T
print(y)
```

    variable([[0.5041187  0.41034285]
              [0.06302946 0.8664606 ]
              [0.68756705 0.63477581]])
    variable([[0.5041187  0.41034285]
              [0.06302946 0.8664606 ]
              [0.68756705 0.63477581]])
    


```python
# transpose 함수
A, B, C, D = 1, 2, 3, 4
x = np.random.rand(A, B, C, D)
print(f'{x} {x.shape} \n')
print('------------------------------------------------------- \n')
y = x.transpose(1, 0, 3, 2)
print(f'{y} {y.shape} \n')
print('------------------------------------------------------- \n')
y = x.transpose()
print(f'{y} {y.shape} \n')
```

    [[[[0.75147587 0.30682387 0.85112738 0.63454265]
       [0.34424171 0.90582832 0.32751845 0.06420491]
       [0.67027894 0.17196008 0.31259227 0.30201902]]
    
      [[0.6280575  0.7044106  0.61482446 0.55201586]
       [0.12795702 0.68524623 0.37494585 0.07248666]
       [0.68783251 0.94559924 0.03004996 0.53189678]]]] (1, 2, 3, 4) 
    
    ------------------------------------------------------- 
    
    [[[[0.75147587 0.34424171 0.67027894]
       [0.30682387 0.90582832 0.17196008]
       [0.85112738 0.32751845 0.31259227]
       [0.63454265 0.06420491 0.30201902]]]
    
    
     [[[0.6280575  0.12795702 0.68783251]
       [0.7044106  0.68524623 0.94559924]
       [0.61482446 0.37494585 0.03004996]
       [0.55201586 0.07248666 0.53189678]]]] (2, 1, 4, 3) 
    
    ------------------------------------------------------- 
    
    [[[[0.75147587]
       [0.6280575 ]]
    
      [[0.34424171]
       [0.12795702]]
    
      [[0.67027894]
       [0.68783251]]]
    
    
     [[[0.30682387]
       [0.7044106 ]]
    
      [[0.90582832]
       [0.68524623]]
    
      [[0.17196008]
       [0.94559924]]]
    
    
     [[[0.85112738]
       [0.61482446]]
    
      [[0.32751845]
       [0.37494585]]
    
      [[0.31259227]
       [0.03004996]]]
    
    
     [[[0.63454265]
       [0.55201586]]
    
      [[0.06420491]
       [0.07248666]]
    
      [[0.30201902]
       [0.53189678]]]] (4, 3, 2, 1) 
    
    

![그림 38-4](/assets/images/img/그림 38-4.png)    
x.transpose() : 역순으로 정렬
