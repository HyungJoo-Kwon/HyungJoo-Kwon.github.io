---
layout: page
title: "밑바닥부터 시작하는 딥러닝 31단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 31단계"
categories: deeplearnig
comments: true
published: true
---
# 고차 미분(이론)   
```python
class Sin(Function):
    ...

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx
```

![그림 31-1](/assets/images/img/그림 31-1.png)
![그림 31-1](/assets/images/img/그림 31-2.png)

Variable 클래스의 grad는 ndarray 인스턴스를 참조하도록 새로운 Variable 클래스를 만듬    
![그림 31-3](/assets/images/img/그림 31-3.png)    
적용    
![그림 31-4](/assets/images/img/그림 31-4.png)
실제

gx.backward()를 호출함으로써 y의 x에 대한 2차 미분이 가능   
![그림 31-5](/assets/images/img/그림 31-5.png)


