---
layout: page
title: "밑바닥부터 시작하는 딥러닝 43단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 43단계"
categories: deeplearnig
comments: true
published: true
---
# 신경망        

```python
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F
```

![그림 43-1](/assets/images/img/그림 43-1.png)    
linear처럼 사용 시 메모리를 더 효율적으로


```python
from dezero.core import Function, Variable, as_variable, as_array

class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb

def linear(x, W, b=None):
    return Linear()(x, W, b)

def linear_simple(x, W, b=None):
    t = matmul(x, W)
    print(f't {type(t)}')
    if b is None:
        return t

    y = t + b
    t.data = None  # Release t.data (ndarray) for memory efficiency
    return y

# chainer는 Aggressive Buffer Release 사용
# https://docs.google.com/document/d/1CxNS2xg2bLT9LoUe6rPSMuIuqt8Gbkz156LTPJPT3BE/edit#
```


```python
np.random.seed(0)
x = np.random.rand(100, 1)                          
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)          # 비선형 데이터셋
```

신경망은 선형 변환의 출력에 비선형 변환을 수행. (비선형 변환 = 활성화 함수)     
![식 43.1](/assets/images/img/식 43.1.png)  
![그림 43-3](/assets/images/img/그림 43-3.png)


```python
# 시그모이드 함수
def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y

# 시그모이드 개선
class Sigmoid(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        # y = 1 / (1 + xp.exp(-x))
        y = xp.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)
```


```python
I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H)) # (1, 10), 가중치는 랜덤으로
b1 = Variable(np.zeros(H))                  # (10,)
W2 = Variable(0.01 * np.random.randn(H, O)) # (10, 1), 가중치는 랜덤으로
b2 = Variable(np.zeros(O))                  # (1,)


def predict(x):                 # 2층 신경망
    y = F.linear(x, W1, b1)     
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y


lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    if i % 1000 == 0:
        print(loss)


# Plot
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, .01)[:, np.newaxis]             
print(np.arange(0, 1, .01)[:, np.newaxis].shape)   # (100,1) 열 부분에 np.newaxis를 입력시, 차원을 분해한 후 한 단계 추가합니다.
print(np.arange(0, 1, .01)[np.newaxis, :].shape)  # (1,100) 행 부분에 np.newaxis를 입력시, 차원을 한 단계 추가합니다.

y_pred = predict(t)
plt.plot(t, y_pred.data, color='r')
plt.show()
```

    variable(0.8561756639400779)
    variable(0.2521551854421045)
    variable(0.2489420133902076)
    variable(0.24035819448025872)
    variable(0.21732506692039028)
    variable(0.17844212478459937)
    variable(0.1091334726886173)
    variable(0.07966760543620623)
    variable(0.07773958613812558)
    variable(0.0774167592137966)
    (100, 1)
    (1, 100)
    


    
![png](/assets/images/img/output_43_1.png)
    



```python
np.arange(0, 1, .01)[:, np.newaxis].shape
```




    (100, 1)
