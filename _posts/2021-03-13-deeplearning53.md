---
layout: page
title: "밑바닥부터 시작하는 딥러닝 53단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 53단계"
categories: deeplearnig
comments: true
published: true
---
# 모델 저장 및 읽어오기     

```python
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import dezero

import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP
```


```python
import numpy as np 

x = np.array([1, 2, 3])
np.save('test.npy', x)          # np.save() : ndarray 인스턴스를 외부 파일로 저장

x = np.load('test.npy')         # np.load() : 이미 저장되어 있는 데이터를 읽어올 때 사용
print(x)
# ndarray를 저장할 땐 .npy로 확장자를 해주는 게. 만약 확장자를 생략하면 자동으로 .npy가 뒤에 추가됨
print('----------------------------------------------------------------------------')
x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])

np.savez('test.npz', x1=x1, x2=x2)      # 여러 개의 ndarray 인스턴스를 저장하고 읽어옴
                                        # x1=x1, x2=x2처럼 키워드 인수를 지정할 수 있다.
arrays = np.load('test.npz')
x1 = arrays['x1']
x2 = arrays['x2']
print(x1)
print(x2)
print('----------------------------------------------------------------------------')
x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])
data = {'x1':x1, 'x2':x2}       # 키워드를 딕셔너리로 묶음

np.savez('test.npz', **data)    # **data와 같이 앞에 별표 두 개를 붙여주면 딕셔너리가 자동으로 전개되어 전달

arrays = np.load('test.npz')
x1 = arrays['x1']
x2 = arrays['x2']
print(x1)
print(x2)

# np.savez()와 비슷한 기능을 하는 함수로 np.savez_compressed()가 있다.
```

    [1 2 3]
    ----------------------------------------------------------------------------
    [1 2 3]
    [4 5 6]
    ----------------------------------------------------------------------------
    [1 2 3]
    [4 5 6]
    


```python
# Layer 클래스의 매개변수를 평평하게
from dezero.layers import Layer
from dezero.core import Parameter

layer = Layer()

l1 = Layer()
l1.p1 = Parameter(np.array(1))

layer.l1 = l1
layer.p2 = Parameter(np.array(2))
layer.p3 = Parameter(np.array(3))

params_dict = {}
layer._flatten_params(params_dict)
print(params_dict)

# class Layer:
#     def _flatten_params(self, params_dict, parent_key=""):                Layer클래스의 인스턴스 변수인 _params에는 Parameter 또는 Layer의 인스턴스 변수 이름을 가짐
#         for name in self._params:                                         실제 객체는 obj = self.__dict__[name] 으로 꺼내야 함
#             obj = self.__dict__[name]
#             key = parent_key + '/' + name if parent_key else name

#             if isinstance(obj, Layer):                                    꺼낸 obj가 Layer라면 obj의 _flatten_params 메서드를 호출
#                 obj._flatten_params(params_dict, key)
#             else:
#                 params_dict[key] = obj
```

    {'l1/p1': variable(1), 'p3': variable(3), 'p2': variable(2)}
    

![그림 53-1](/assets/images/img/그림 53-1.png)


```python
# Layer 클래스의 save(), load()

# class Layer:
#     def save_weights(self, path):                                                 # 데이터가 메인 메모리에 존재함을 보장함(데이터가 넘파이 ndarray임을)
#         self.to_cpu()

#         params_dict = {}
#         self._flatten_params(params_dict)
#         array_dict = {key: param.data for key, param in params_dict.items()
#                       if param is not None}
#         try:
#             np.savez_compressed(path, **array_dict)
#         except (Exception, KeyboardInterrupt) as e:
#             if os.path.exists(path):
#                 os.remove(path)
#             raise

#     def load_weights(self, path):                                                 # np.laod 함수로 데이터를 읽어 들인 후 대응하는 키 데이터를 매개변수로 설정함.
#         npz = np.load(path)
#         params_dict = {}
#         self._flatten_params(params_dict)
#         for key, param in params_dict.items():
#             param.data = npz[key]
```


```python
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP

max_epoch = 3
batch_size = 100

train_set = dezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
model = MLP((1000, 10))
optimizer = optimizers.SGD().setup(model)

# 매개변수 읽기
if os.path.exists('my_mlp.npz'):
    model.load_weights('my_mlp.npz')

for epoch in range(max_epoch):
    sum_loss = 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)

    print('epoch: {}, loss: {:.4f}'.format(
        epoch + 1, sum_loss / len(train_set)))

model.save_weights('my_mlp.npz')

# 코드 처음 실행 시 my_mlp.npz 파일이 존재하지 않으므로 모델의 매개변수를 무작위로 초기화한 상태에서 학습 시작.
```

    epoch: 1, loss: 1.9273
    epoch: 2, loss: 1.2962
    epoch: 3, loss: 0.9324
    

