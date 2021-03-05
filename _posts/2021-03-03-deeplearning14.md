---
layout: page
title: "밑바닥부터 시작하는 딥러닝 14단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 14단계"
categories: deeplearnig
comments: true
published: true
---
# 같은 변수 반복 사용의 문제 해결   

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

    def cleargrad(self):        # 같은 변수를 사용하여 다른 계산을 할 경우(Variable 인스턴스를 재사용 할 경우)) 
        self.grad = None        # 미분값 초기화가 필요

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:                             # 답, step 13과의 차이
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

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


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


x = Variable(np.array(3.0))
y = add(x, x)
y.backward()
print(f'1 : y.grad {y.grad}')
print(f'1 : x.grad {x.grad}')

x.cleargrad()   # 미분값 초기화
x = Variable(np.array(3.0))  # 같은 Variable를 사용
y = add(add(x, x), x)
y.backward()
print(f'2 : y.grad {y.grad}')
print(f'2 : x.grad {x.grad}')

```

    1 : y.grad 1.0
    1 : x.grad 2.0
    2 : y.grad 1.0
    2 : x.grad 3.0
    


```python
# 복사와 덮어 쓰기
x = np.array(1)
print (id(x))

x += x
print (id(x)) # 덮어 쓰기 == 인플레이스 연산

x = x + x
print (id(x)) # 복사 (새로 생성)

# x.grad = x.grad + gx 를 x.grad += gx로 인플레이스 연산 사용 시 값을 덮어 써서 y.grad와 x.grad가 같은 값을 참조하게 되어
```

    2513712565824
    2513712565824
    2513750896048
    



