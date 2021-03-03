---
layout: page
title: "밑바닥부터 시작하는 딥러닝 1단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 1단계"
categories: deeplearnig
comments: true
published: true
---
# Variable 클래스   

```python
import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


data = np.array(1.0)
x = Variable(data)
print(x.data)

x.data = np.array(2.0)
print(x.data)


# x = np.array(1)  // 0차원 배열, 스칼라
# x.ndim  >>  0  

# x = np.array([1, 2, 3]) // 1차원 배열, 벡터
# x.ndim  >>  1

# x = np.array([[1, 2, 3],[4, 5, 6]]) // 2차원 배열, 행렬, 3차원 벡터
# x.ndim  >>  2


```

    1.0
    2.0
    

![그림 1-2](https://user-images.githubusercontent.com/73815944/109747363-54e1a500-7c1a-11eb-9c64-e188d642bb16.png)



