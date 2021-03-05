---
layout: page
title: "밑바닥부터 시작하는 딥러닝 17단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 17단계"
categories: deeplearnig
comments: true
published: true
---
# 메모리 관리와 순환 참조 
```python
                              
import weakref
import numpy as np


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

    def backward(self):
        if self.grad is None:
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

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]      # 약한 참조 사용
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
```

파이썬(CPython)의 메모리 관리 방식  
1. 참조 카운트 (참조 수를 세는 방식)    
    - 대입 연산자 사용할 때     
    - 함수에 인수로 전달할 때   
    - 컨테이너 타입 객체(리스트, 튜플, 클래스 등)에 추가할 때   
2. garbage collection (쓸모없어진 객체를 회수하는 방식) 


![17-2]/(assets/images/img/그림 17-2.png)     
참조 카운트로 해결할 수 없는 순환 참조 문제 -> 참조 카운트가 0이 되지 않아

![17-3](./img/17-3.png)     
Function 인스턴스와 Variable 인스턴스가 순환 참조 관계를 만든다.


```python
# 약한 참조
import weakref
import numpy as np 
 
a = np.array([1,2,3])
b = weakref.ref(a)

print(b)
print(b())

a = None
print(b)

# 순환 참조를 weakref 모듈로 해결 가능
```

    <weakref at 0x0000024512FA9860; to 'numpy.ndarray' at 0x0000024512FA97B0>
    [1 2 3]
    <weakref at 0x0000024512FA9860; dead>
    


```python
for i in range(10):
    x = Variable(np.random.randn(10000))  # big data
    y = square(square(square(x)))
```

![17-4](./img/17-4.png)

파이썬 메모리 사용량을 측정하려면 외부 라이브러리인 memory profiler 사용 시 확인 가능
