---
layout: page
title: "밑바닥부터 시작하는 딥러닝 44단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 44단계"
categories: deeplearnig
comments: true
published: true
---
# 매개변수를 모아두는 계층      

```python
# 매개변수 = 가중치, 편향
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


```


```python
# Parameter 구현
from dezero.core import Variable

class Parameter(Variable):
    pass

import numpy as np 

x = Variable(np.array(1.0))
p = Parameter(np.array(2.0))
y = x * p

print(y)
print(isinstance(p, Parameter))
print(isinstance(x, Parameter))
print(isinstance(y, Parameter))
# Parameter 인스턴스와 Variable 인스턴스를 조합하여 계산 가능
# isinstance 함수로 구분 가능
```

    variable(2.0)
    True
    False
    False
    


```python
# Layer 클래스 구현
class Layer:
    def __init__(self):
        self._params = set()                # 집합 : 원소들에 순서가 없고, ID가 같은 객체는 중복 저장 불가

    def __setattr__(self, name, value):             # 인스튼스 변수를 설정할 때 호출되는 특수 메서드. 이름이 name인 인스턴스 변수에 값으로 value를 전달
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)                  # value가 Parameter, Layer 인스턴스라면 self._parmas에 name 추가
        super().__setattr__(name, value)

    def __call__(self, *inputs):                    
        outputs = self.forward(*inputs)             # 입력받은 인수를 건네 foward 메서드를 호출
        if not isinstance(outputs, tuple):
            outputs = (outputs,)                    # 튜플이 아니라면 그 출력을 직접 반환
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):                               # Layer 인스턴스에 담겨 있는 Parameter 인스턴스들을 꺼내줌
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()             # yield : return(처리를 종료하고 값을 반환)처럼 사용 가능하지만 yield는 처리를 일시 중지 하고 값을 반환, params 메서드를 호출할 때마다 일시 중지됐던 처리가 재개됨. 
            else:
                yield obj

    def cleargrads(self):                           # 모든 매개변수의 기울기를 재설정
        for param in self.params():
            param.cleargrad()
```


```python
# Ex
layer = Layer() 

layer.p1 = Parameter(np.array(1))
layer.p2 = Parameter(np.array(2))
layer.p3 = Variable(np.array(3))
layer.p4 = 'test'

print(layer._params)

for name in layer._params:
    print(name, layer.__dict__[name])   # __dict__에는 모든 인스턴스 변수가 딕셔너리 타입으로 저장
```

    {'p2', 'p1'}
    p2 variable(2)
    p1 variable(1)
    


```python
# Linear 클래스 구현
class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None:            # in_size가 저장되어 있지 않다면 나중으로 연기
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):               # 데이터를 흘려보내는 시점에 가중치 초기화
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.linear(x, self.W, self.b)
        return y
```


```python
# Layer를 이용한 신경망 구현
import numpy as np
import dezero.functions as F
import dezero.layers as L


np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

l1 = L.Linear(10)       # 출력 크기 지정
l2 = L.Linear(1)


def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y


lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data
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
    


