---
layout: page
title: "밑바닥부터 시작하는 딥러닝 12단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 12단계"
categories: deeplearnig
comments: true
published: true
---
# 가변 길이 인수(개선)      

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
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):        # *inputs 가변인수 사용, 밑에 참고
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)          # *를 붙여 언팩 (언팩: 리스트의 원소를 낱개로 풀어서 전달) == self.forward(x0, x1)
        if not isinstance(ys, tuple):   # 튜플이 아닌 경우 추가 지원
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0] # 리스트의 원소가 하나라면 첫 번째 원소를 반환한다.

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y


def add(x0, x1):
    return Add()(x0, x1)


x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
y = add(x0, x1)
print(y.data)
```

    5
    


```python
def f(*x):          # 함수를 정의할 때 인수에 *를 붙이면 호출할 때 넘긴 인수들을 *붙인 인수 하나로 모아서 받을 수 있다.
    print(x)

print( f(1,2,3) )
print( f(1,2,3,4,5,6) )
```

    (1, 2, 3)
    None
    (1, 2, 3, 4, 5, 6)
    None
    



