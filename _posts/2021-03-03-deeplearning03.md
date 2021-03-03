---
layout: single
title: "밑바닥부터 시작하는 딥러닝 3단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 3단계"
categories: deeplearnig
comments: true
published: true
---
# 합성함수  

```python
import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2


class Exp(Function):
    def forward(self, x):
        return np.exp(x)


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)


```

    1.648721270700128
    

![그림 3-1](https://user-images.githubusercontent.com/73815944/109747915-51025280-7c1b-11eb-872c-4af2e232c4be.png)



```python

```
