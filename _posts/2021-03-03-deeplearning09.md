---
layout: page
title: "밑바닥부터 시작하는 딥러닝 9단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 9단계"
categories: deeplearnig
comments: true
published: true
---
# 함수 수정     

```python


import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):     # isinstance(변수, 타입) : 타입 판별 true or false
                raise TypeError('{} is not supported'.format(type(data)))  

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:                           # y.grad = np.array(1.0) 생략 
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):      # numpy.float64, int, float형 등의 타입도 스칼라로 판단
        return np.array(x)  
    return x


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
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


def square(x):
    # f = Square()
    # return f(x)
    return Square()(x)  


def exp(x):
    # f = Exp()
    # return f(x)
    return Exp()(x)


x = Variable(np.array(0.5))
y = square(exp(square(x)))      # 연속적용
y.backward()
print(x.grad)


x = Variable(np.array(1.0))  # OK
x = Variable(None)  # OK
# x = Variable(1.0)  # NG
```

    3.297442541400256
    


```python
# check
x = np.array([1.0])
y = x ** 2
print(type(x), x.ndim)
print(type(y))

print()

x = np.array(1.0)
y = x ** 2
print(type(x), x.ndim)  
print(type(y))                  # 타입 변화

# 타입의 변화가 일어나 as_array() 함수 사용
```

    <class 'numpy.ndarray'> 1
    <class 'numpy.ndarray'>
    
    <class 'numpy.ndarray'> 0
    <class 'numpy.float64'>
    

