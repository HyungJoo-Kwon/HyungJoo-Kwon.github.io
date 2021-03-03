---
layout: page
title: "밑바닥부터 시작하는 딥러닝 2단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 2단계"
categories: deeplearnig
comments: true
published: true
---
# Function 클래스   

```python
import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


class Function:
    def __call__(self, input):  # f = Function() 형태로 함수의 인스턴스를 변수 f에 대입해두고, f() 형태로 __call__ 메서드 호출 가능.
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, in_data):
        raise NotImplementedError()


class Square(Function):     # Function 클래스를 상속받음. https://wikidocs.net/16073
    def forward(self, x):
        return x ** 2



x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)


```

    <class '__main__.Variable'>
    100
    

![그림 2-1](https://user-images.githubusercontent.com/73815944/109747871-41830980-7c1b-11eb-9046-fe35d2aa2e74.png)



```python

```
