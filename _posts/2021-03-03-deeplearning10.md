---
layout: collection
title: "밑바닥부터 시작하는 딥러닝 10단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 10단계"
categories: deeplearnig
comments: true
published: true
---
# 테스트    

```python

# unittest를 import하고 unittest.TestCase를 상속한 클래스를 구현
# 이름이 test로 시작하는 메서드를 만들고 안에 테스트할 내용을 적는다.
# python -m unittest step10.py   (-m unittest : 파이썬 파일을 테스트 모드로 실행)

import unittest
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


def square(x):
    return Square()(x)


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class SquareTest(unittest.TestCase):    
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected) #________________ ok

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected) # _______________ ok

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)     # np.allclose(a, b, rtol, atol) : |a - b| <= (atol + rtol * |b|)
        self.assertTrue(flg)                    # a, b의 값이 가까운지 판단
                                                # ______________ ok
```


```python
# python -m unittest discover tests
# discover라는 하위 명령을 사용하면 discover 다음에 지정한 디렉터리에서 테스트 파일이 있는지 검색한다.
# 지정한 디렉터리에서 test*.py 형태인 파일을 테스트 파일로 인식하여 tests 디렉터리에 들어 있는 모든 테스트를 한 번에 실행한다.
```
