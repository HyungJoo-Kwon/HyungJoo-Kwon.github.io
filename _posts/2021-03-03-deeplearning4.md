---
layout: single
title: "밑바닥부터 시작하는 딥러닝 4단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 4단계"
categories: deeplearnig
comments: true
published: true
---
# 미분
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
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2


class Exp(Function):
    def forward(self, x):
        return np.exp(x)


def numerical_diff(f, x, eps=1e-4):     # f : 미분 대상 함수, x : 미분 지점 
    x0 = Variable(x.data - eps)         # 전진차분 보다 중앙차분이 상대적으로 오차가 작음.
    x1 = Variable(x.data + eps)         # (f(x+h) - f(x-h)) / 2h
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)
print(dy)


def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


x = Variable(np.array(0.5))
dy = numerical_diff(f, x)           # e^(x^2)의 미분
print(dy)
```

    4.000000000004
    3.2974426293330694
    

![그림 4-1](https://user-images.githubusercontent.com/73815944/109747956-611a3200-7c1b-11eb-887a-9b90b9ac0538.png)
![그림 4-2](https://user-images.githubusercontent.com/73815944/109747959-61b2c880-7c1b-11eb-8eaa-efbd5777cbd2.png)



```python

```
