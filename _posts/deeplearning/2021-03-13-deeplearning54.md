---
layout: page
title: "밑바닥부터 시작하는 딥러닝 54단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 54단계"
categories: deeplearnig
comments: true
published: true
---
# 드롭아웃과 테스트 모드    

```python
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import test_mode
import dezero.functions as F

# 과대적합이 일어나는 주요 원인 
#   - 훈련 데이터가 적음 : 해결책 (데이터 확장) 
#   - 모델의 표현력이 지나치게 높음 : (가중치 감소, 드롭아웃, 배치 정규화)
```

드롭아웃 : 뉴런을 임의로 삭제(비활성화)하면서 학습하는 방법. 학습 시에는 뉴런을 무작위로 골라 삭제      
![그림 54-1](/assets/images/img/그림 54-1.png)


```python
# 드롭아웃 == 다이렉트 드롭아웃
import numpy as np 

dropout_ratio = 0.6
x = np.ones(10)

for _ in range(10):
    mask = np.random.rand(10) > dropout_ratio           # 생성한 mask는 False의 비율이 평균적으로 60%가 될 것.
    y = x * mask                                    
    print(y)

# 학습 시 
mask = np.random.rand(*x.shape) > dropout_ratio
y = x * mask
print(f'train : {y}')

# 테스트 시
scale = 1 - dropout_ratio       # 학습 시에 살아남은 뉴런의 비율
y = x * scale
print(f'test : {y}')

# 학습할때 평균 40%의 뉴런이 생존했기 때문에 테스트할 때는 모든 뉴런을 사용해 계산한 출력에 0.4를 곱함. 이렇게 학습, 테스트 시의 비율을 일치시킨다.
```

    [1. 0. 1. 1. 1. 0. 0. 0. 1. 0.]
    [1. 0. 0. 0. 0. 1. 0. 0. 0. 1.]
    [1. 0. 1. 1. 0. 1. 0. 1. 0. 0.]
    [1. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
    [0. 1. 0. 0. 1. 0. 0. 1. 1. 0.]
    [1. 0. 1. 0. 0. 0. 1. 0. 1. 0.]
    [1. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    [1. 1. 0. 0. 0. 0. 0. 1. 1. 0.]
    [1. 1. 0. 0. 0. 1. 0. 0. 0. 0.]
    [0. 0. 0. 0. 1. 0. 0. 0. 1. 1.]
    train : [0. 0. 1. 0. 0. 1. 0. 1. 0. 1.]
    test : [0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4]
    


```python
# 역 드롭아웃 : 스케일 맞추기를 '학습할 때' 수행.
# 학습할 때 미리 뉴런의 값에 1/scale을 곱해두고, 테스트 때는 아무런 동작도 하지 않음.

# 학습 시 
scale = 1 - dropout_ratio
mask = np.random.rand(*x.shape) > dropout_ratio      
y = x * mask / scale
print(f'train : {y}')

# 테스트 시
y = x
print(f'test : {y}')

# 테스트 시 아무런 처리도 하지 않기 때문에 테스트 속도가 살짝 향상되고 추론 처리만을 이용하는 경우에 바랍직하다.
# 또한 역 드롭아웃은 학습할 때 dropout_ratio를 동적으로 변경 가능.
# 많은 딥러닝 프레임워크에서 역 드롭아웃 방식을 사용.
```

    train : [2.5 0.  2.5 0.  2.5 0.  0.  0.  0.  2.5]
    test : [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    


```python
# 테스트 모드 추가
class Config:
    enable_backprop = True
    train = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


def test_mode():
    return using_config('train', False)

# dezero/__init__.py에 from dezero.core import Config 가 있어 다른 파일에서 dezero.Config.train 값을 참조 가능.
# with test_mode(): 안에서는 Config.train이 False로 전환
# dezero/__init__.py에 dezero.core import test_mode 추가하여 from dezero import test_mode 형태로 임포트 가능
```


```python
# 드롭아웃 구현
def dropout(x, dropout_ratio=0.5):
    x = as_variable(x)

    if dezero.Config.train:
        xp = cuda.get_array_module(x)
        mask = xp.random.rand(*x.shape) > dropout_ratio
        scale = xp.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x * mask / scale
        return y
    else:
        return x
```

머신러닝에서는 앙상블 학습이 주로 사용      
앙상블 학습 : 여러 모델을 개별적으로 학습시킨 후 추론 시 모든 모델의 출력을 평균 내는 방법.


```python
import numpy as np
from dezero import test_mode
import dezero.functions as F

x = np.ones(5)
print(x, '\n')

# When training
for _ in range(5):
    y = F.dropout(x)
    print(y)
print()

# When testing (predicting)
with test_mode():
    y = F.dropout(x)
    print(y)
```

    [1. 1. 1. 1. 1.] 
    
    variable([0. 0. 2. 2. 2.])
    variable([0. 2. 0. 0. 0.])
    variable([2. 2. 2. 2. 2.])
    variable([2. 0. 0. 0. 2.])
    variable([0. 2. 0. 0. 2.])
    
    variable([1. 1. 1. 1. 1.])
    
