---
layout: page
title: "밑바닥부터 시작하는 딥러닝 18단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 18단계"
categories: deeplearnig
comments: true
published: true
---
# 메모리 절약 모드  

```python
import weakref
import numpy as np
import contextlib


class Config:               # 
    enable_backprop = True      # enable_backprop : 역전파가 가능한지 여부
                                # True : 역전파 활성 모드, False : 역전파 비활성 모드

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)   # getattr(object, name[, default]) : object에 존재하는 속성의 값을 가져온다.
    # print(f'old_value {old_value}')
    setattr(Config, name, value)        # setattr(object, name, value) : object에 존재하는 속성의 값을 바꾸거나, 새로운 속성을 생성하여 값을 부여한다.
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):      # retain_grad = True : 모든 변수가 미분 결과(기울기) 유지
        if self.grad is None:                   # retain_grad = False : 중간 변수의 미분값을 모두 None으로 설정
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # output is weakref
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:                           # --------------------------------------------------          
                for y in f.outputs:                       #  y() : y가 약한 참조이기 때문                
                    y().grad = None  # y is weakref       #  y().grad = None 코드가 실행되면 참조카운트가 0이 되어 미분값 
                                                          #                  데이터가 메모리에서 삭제됨
                                                          # --------------------------------------------------


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

        if Config.enable_backprop:                                           # 역전파 코드 실행
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs                                             # -> inputs는 역전파 계산 시 사용됨.
            self.outputs = [weakref.ref(output) for output in outputs]       # 신경망에는 학습, 추론 단계가 있고, 학습 시에는 미분값을 구해야하지만, 추론 시에는 순전파만 하기 때문에 중간 계산 결과를 버리면 메모리 사용량을 크게 줄일 수 있음.

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
    return Square()(x)


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
t = add(x0, x1)
y = add(x0, t)
y.backward()
print(y.grad, t.grad)  # None None
print(x0.grad, x1.grad)  # 2.0 1.0


with using_config('enable_backprop', False):        # with 문 안에선 이용한 역전파 비활성 모드, 벗어나면 활성 모드
    x = Variable(np.array(2.0))
    y = square(x)

with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)
```

    None None
    2.0 1.0
    old_value True
    old_value True
    


```python
# 모드 전환 ex

Config.enable_backprop = True           # 역전파 활성
x = Variable(np.ones((100, 100, 100)))
y = square(square(square(x)))
y.backward()

Config.enable_backprop = False          # 역전파 비활성
x = Variable(np.ones((100, 100, 100)))
y = square(square(square(x)))
```


```python
# contextlib ex

import contextlib

@contextlib.contextmanager      # 데코레이터를 달면 문맥을 판단하는 함수가 만들어짐
def config_test():
    print('start')  # 전처리
    try:
        yield
    finally:
        print('done')  # 후처리

with config_test():
    print('process...')

# with 블록 안으로 들어갈 때 전처리가 실행되고 블록 범위를 빠져나올 때 후처리가 실행됨.
# with 블록 안에서 예외가 발생할 수 있고, 발생한 예외는 yeild를 실행하는 코드로 전달.
```

    start
    process...
    done
    
