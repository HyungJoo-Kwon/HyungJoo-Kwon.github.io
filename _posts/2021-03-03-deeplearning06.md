---
layout: page
title: "밑바닥부터 시작하는 딥러닝 6단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 6단계"
categories: deeplearnig
comments: true
published: true
---
# 수동 역전파   

```python


import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)
```

    3.297442541400256
    

![그림 6-1](https://user-images.githubusercontent.com/73815944/109754605-ad6b6f00-7c27-11eb-917d-483727fec0c1.png)
![그림 6-2](https://user-images.githubusercontent.com/73815944/109754607-ae9c9c00-7c27-11eb-929f-33e56cd6fbf0.png)




