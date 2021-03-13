---
layout: page
title: "밑바닥부터 시작하는 딥러닝 57단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 57단계"
categories: deeplearnig
comments: true
published: true
---
# conv2d 함수와 pooling 함수    

```python
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F
```

im2col에 의한 전개      
im2col은 입력 데이터를 펼쳐줌.  
![그림 57-1](/assets/images/img/그림 57-1.png)    
 * 텐서 곱은 단순히 행렬 곱의 확장으로, 축을 지정하여 두 텐서를 곱셈한 후 누적하는 계산.   
 * 이를 곱셈 누적 연산이라 하며 넘파이에서는 np.tensordot과 np.einsum을 사용해 텐서 곱을 계산 가능      
 
![그림 57-2](/assets/images/img/그림 57-2.png)

conv2d 함수 구현    
im2col(x, kernel_size, stride=1, pad=0, to_matrix=True)     
![표 57-1](/assets/images/img/표 57-1.png)


```python
import numpy as np
from dezero import Variable
import dezero.functions as F


# im2col
x1 = np.random.rand(1, 3, 7, 7)     # 배치 크기 = 1
col1 = F.im2col(x1, kernel_size=5, stride=1, pad=0, to_matrix=True)
print(col1.shape)  # (9, 75)    75 = channel(3) * kenel_size (5 x 5)의 원ㅗ수

x2 = np.random.rand(10, 3, 7, 7)  # 배치 크기 = 10
kernel_size = (5, 5)
stride = (1, 1)
pad = (0, 0)
col2 = F.im2col(x2, kernel_size, stride, pad, to_matrix=True)
print(col2.shape)  # (90, 75)

def pair(x):                # 인수 x 가 int라면 (x,x) 튜플 형태 반환, 원소 2개짜리 튜플이면 그대로 돌려줌
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError

print(pair(1))
print(pair((1,2)))
```

    (9, 75)
    (90, 75)
    (1, 1)
    (1, 2)
    


```python
def conv2d_simple(x, W, b=None, stride=1, pad=0):
    x, W = as_variable(x), as_variable(W)

    Weight = W
    N, C, H, W = x.shape
    OC, C, KH, KW = Weight.shape
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    col = im2col(x, (KH, KW), stride, pad, to_matrix=True)      # 입력데이터를 im2col로 전개
    Weight = Weight.reshape(OC, -1).transpose()         # weight를 재정렬. 
    # reshape()의 마지막 인수를 -1로 주면 그앞의 인수들로 정의한 다차원 배열에 전체 원소들을 적절히 분배
    # ex (10, 3, 5, 5) 의 원소가 총 750개인데, reshape(10, -1)을 수행하면 (10, 75)로 변환 
    t = linear(col, Weight, b)              # 행렬 곱 계산
    y = t.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)  # 아래 그림 참고
    return y


# conv2d
N, C, H, W = 1, 5, 15, 15
OC, (KH, KW) = 8, (3, 3)
x = Variable(np.random.randn(N, C, H, W))
W = np.random.randn(OC, C, KH, KW)
y = F.conv2d_simple(x, W, b=None, stride=1, pad=1)
y.backward()
print(y.shape)  # (1, 8, 15, 15)
print(x.grad.shape)  # (1, 5, 15, 15)
```

    (1, 8, 15, 15)
    (1, 5, 15, 15)
    

![그림 57-3](/assets/images/img/그림 57-3.png)


```python
# Conv2d 계층 구현 dezero/layers.py
class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        """Two-dimensional convolutional layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y

```

![표 57-2](/assets/images/img/표 57-2.png)

pooling 함수 구현       
풀링 적용 영역은 채널마다 독립적으로.   
![그림 57-4](/assets/images/img/그림 57-4.png)    
![그림 57-5](/assets/images/img/그림 57-5.png)


```python
def pooling_simple(x, kernel_size, stride=1, pad=0):
    x = as_variable(x)

    N, C, H, W = x.shape
    KH, KW = pair(kernel_size)
    PH, PW = pair(pad)
    SH, SW = pair(stride)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    col = im2col(x, kernel_size, stride, pad, to_matrix=True)   # 데이터 전개
    col = col.reshape(-1, KH * KW)              
    y = col.max(axis=1)                                         # 각 행의 최댓값을 찾는다.
    y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)          # 형상 변환
    return y
```
