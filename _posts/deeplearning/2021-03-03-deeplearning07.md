---
layout: page
title: "밑바닥부터 시작하는 딥러닝 7단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 7단계"
categories: deeplearnig
comments: true
published: true
---
# 역전파 자동화 (재귀)  

```python


import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator  # 1. Get a function
        if f is not None:
            x = f.input  # 2. Get the function's input
            x.grad = f.backward(self.grad)  # 3. Call the function's backward
            x.backward() # 4. 하나 앞 변수의 backward 메서드를 호출(재귀)


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)  # Set parent(function)
        self.input = input
        self.output = output  # Set output
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

# backward
y.grad = np.array(1.0)
y.backward()
print(x.grad)
```

    3.297442541400256
    

assert 문
 - assert 조건  # 조건의 결과가 True가 아니면 예외가 발생

![그림 7-2](https://user-images.githubusercontent.com/73815944/109758007-4309fd00-7c2e-11eb-8e1c-126e42bb269f.png)
![그림 7-3](https://user-images.githubusercontent.com/73815944/109758008-443b2a00-7c2e-11eb-8f81-c4df1b2c0fde.png)




