---
layout: page
title: "밑바닥부터 시작하는 딥러닝 46단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 46단계"
categories: deeplearnig
comments: true
published: true
---
# Optimizer로 수행하는 매개변수 갱신    

```python
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
```


```python
# Optimizer 클래스
# 매개변수 갱신을 위한 기반 클래스
class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        # None 이외에 매개변수를 리스트에 모아둠
        params = [p for p in self.target.params() if p.grad is not None] # grad 가 None이면 pass

        # 전처리(옵션)
        for f in self.hooks:
            f(params)
        # 매개변수 갱신
        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)

# __init__() : target과 hooks 두개의 인스턴스 변수 초기화
# setup() : 매개변수를 갖는 클래스(Model or Layer)를 target으로 설정
# update() : 모든 매개변수 갱신
# add_hook() : 전처리를 수행하는 함수를 추가
```


```python
# SGD (stochastic Gradient Descent) 클래스 
# 확률적경사하강법으로 매개변수를 갱신하는 클래스
# 확률적경사하강법 : 데이터 중에서 무작위로(확률적으로) 선별한 데이터에 대해 경사하강법을 수행
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data
```


```python
import numpy as np
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP


np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
max_iter = 10000
hidden_size = 10

model = MLP((hidden_size, 1))
optimizer = optimizers.SGD(lr).setup(model)

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()      # 매개변수 갱신
    if i % 1000 == 0:
        print(loss)
```

    variable(0.8165178492839196)
    variable(0.24990280802148895)
    variable(0.24609876581126014)
    variable(0.23721590814318072)
    variable(0.20793216413350174)
    variable(0.12311905720649353)
    variable(0.07888166506355153)
    variable(0.07655073683421634)
    variable(0.07637803086238225)
    variable(0.07618764131185568)
    

