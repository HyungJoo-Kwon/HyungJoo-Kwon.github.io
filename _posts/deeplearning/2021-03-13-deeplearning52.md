---
layout: page
title: "밑바닥부터 시작하는 딥러닝 52단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 52단계"
categories: deeplearnig
comments: true
published: true
---
# GPU 지원      

```python
# 딥러닝 계산은 행렬의 곱이 대부분. 행렬의 곱은 곱셈과 덧셈으로 구성되어 병렬로 계산이 가능하고, 병렬 계산에는 cpu보다 gpu가 훨씬 뛰어남
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP
```


```python
# 쿠다 모듈
import numpy as np 
gpu_enable = True
try:
  import cupy as cp
  cupy = cp
except ImportError:
  gpu_enable = False

from dezero import Variable

def get_array_module(x):
    """Returns the array module for `x`.

    Args:
        x (dezero.Variable or numpy.ndarray or cupy.ndarray): Values to
            determine whether NumPy or CuPy should be used.

    Returns:
        module: `cupy` or `numpy` is returned based on the argument.
    """
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp


def as_numpy(x):
    """Convert to `numpy.ndarray`.

    Args:
        x (`numpy.ndarray` or `cupy.ndarray`): Arbitrary object that can be
            converted to `numpy.ndarray`.
    Returns:
        `numpy.ndarray`: Converted array.
    """
    if isinstance(x, Variable):
        x = x.data

    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)


def as_cupy(x):
    """Convert to `cupy.ndarray`.

    Args:
        x (`numpy.ndarray` or `cupy.ndarray`): Arbitrary object that can be
            converted to `cupy.ndarray`.
    Returns:
        `cupy.ndarray`: Converted array.
    """
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        raise Exception('CuPy cannot be loaded. Install CuPy!')
    return cp.asarray(x)

    # get_arrary_module(x)는 인수 x에 대응하는 모듈을 돌려줌 (gpu_enable=True -> cupy, gpu_enable=False -> numpy 반환)
    # 나머지 두 함수는 쿠파이/넘파이 다차원 배열을 서로 변환해주는 함수.
```


```python
# Variable/Layer/DataLoader 클래스 추가 구현  
try:
    import cupy
    array_types = (np.ndarray, cupy.ndarray)    
except ImportError:
    array_types = (np.ndarray)

class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, array_types):
                raise TypeError('{} is not supported'.format(type(data)))

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            xp = dezero.cuda.get_array_module(self.data)                # 데이터의 타입에 따라 넘파이 또는 쿠파이 중 하나의 다차원 배열을 생성
            self.grad = Variable(xp.ones_like(self.data))

    def to_cpu(self):                                                   # gpu -> cpu
        if self.data is not None:
            self.data = dezero.cuda.as_numpy(self.data)

    def to_gpu(self):                                                   # cpu -> gpu
        if self.data is not None:
            self.data = dezero.cuda.as_cupy(self.data)

class Layer:

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

... Page 440
```


```python
import time
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP


max_epoch = 5
batch_size = 100

train_set = dezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
model = MLP((1000, 10))
optimizer = optimizers.SGD().setup(model)

# GPU mode
if dezero.cuda.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()

for epoch in range(max_epoch):
    start = time.time()
    sum_loss = 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)

    elapsed_time = time.time() - start
    print('epoch: {}, loss: {:.4f}, time: {:.4f}[sec]'.format(
        epoch + 1, sum_loss / len(train_set), elapsed_time))
```

    epoch: 1, loss: 1.9142, time: 5.7227[sec]
    epoch: 2, loss: 1.2824, time: 5.4095[sec]
    epoch: 3, loss: 0.9235, time: 5.9271[sec]
    epoch: 4, loss: 0.7391, time: 6.2692[sec]
    epoch: 5, loss: 0.6347, time: 10.2412[sec]
    

![그림 52-1](/assets/images/img/그림 52-1.png)
