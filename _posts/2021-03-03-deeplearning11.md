---
layout: page
title: "밑바닥부터 시작하는 딥러닝 11단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 11단계"
categories: deeplearnig
comments: true
published: true
---
# 가변 길이 인수(순전파)    

```python


# 리스트와 튜플은 여러 데이터를 한 줄로 저장. 
# 주요 차이는 원소를 변경할 수 있는지 여부.
# 리스트 변경 가능, 튜플 한번 생성되면 원소 변경 불가능
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
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:                     # 메서드의 인수와 반환값을 리스트로 바꿈
    def __call__(self, inputs):
        xs = [x.data for x in inputs]  # Get data from Variable
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]  # Wrap data, 리스트 내포

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):        # 인수는 변수가 두개 담긴 리스트
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,) # 튜플 반환
        # == return y,


xs = [Variable(np.array(2)), Variable(np.array(3))]
f = Add()
ys = f(xs)
y = ys[0]
print(y.data)
print(type(y))
print(type(y.data))
```

    5
    <class '__main__.Variable'>
    <class 'numpy.ndarray'>
    

