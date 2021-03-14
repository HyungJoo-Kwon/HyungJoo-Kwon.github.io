---
layout: page
title: "밑바닥부터 시작하는 딥러닝 13단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 13단계"
categories: deeplearnig
comments: true
published: true
---
# 가변 길이 인수 (역전파 편)    

```python

import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]     # outputs에 담겨 있는 미분값들을 리스트에 담는다
            gxs = f.backward(*gys)                          # f의 역전파 호출 (리스트 언팩)
            if not isinstance(gxs, tuple):                  # gxs 튜플이 아니라면 튜플로 변환
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):                # 역전파로 전파되는 미분값을 grad에 
                x.grad = gx

                if x.creator is not None:
                    funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    f = Square()
    return f(x)


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):     # 덧셈의 역전파는 출력 쪽에서 전해지는 미분값에 1을 곱한 값
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


x = Variable(np.array(2.0))
y = Variable(np.array(3.0))

z = add(square(x), square(y))
z.backward()
print(z.data)
print(x.grad)
print(y.grad)
```

    13.0
    4.0
    6.0
    

![그림 13-1](https://user-images.githubusercontent.com/73815944/110052747-6b633a00-7d9b-11eb-9cce-b40b6030f6f7.png)



