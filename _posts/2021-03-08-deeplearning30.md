---
layout: page
title: "밑바닥부터 시작하는 딥러닝 30단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 30단계"
categories: deeplearnig
comments: true
published: true
--- 
# 고차 미분(준비)   
```python
class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0
```

![그림 30-1](/assets/images/img/그림 30-1.png)


```python
x = Variable(np.array(2.0))
x.backward()
x.grad = np.array(1.0)
```

![그림 30-2](/assets/images/img/그림 30-2.png))


```python
class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]                   # ---------------------------------------
        ys = self.forward(*xs)                          # 순전파 계산 
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]   # ---------------------------------------

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:                                       # ---------------------------------------
                output.set_creator(self)                                 # Variable과 Function의 관계가 만들어짐.
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]   # ---------------------------------------

        return outputs if len(outputs) > 1 else outputs[0]
```


```python
class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    def backward(self, gy):
        x = self.inputs[0].cata
        gx = gy * np.cos(x)
        return gx

def sin(x):
    return Sin()(x)

x = Variable(np.array(1.0))
y = sin(x)
```

![그림 30-3](/assets/images/img/그림 30-3.png)


```python
class Variable:
    ...

    def backward(self, retain_grad=False):      # retain_grad = True : 모든 변수가 미분 결과(기울기) 유지
        if self.grad is None:                   # retain_grad = False : 중간 변수의 미분값을 모두 None으로 설정
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
            gys = [output().grad for output in f.outputs]  # output is weakref  # ------------------------------------
            gxs = f.backward(*gys)                                              # 역전파 계산    
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx                                        # ------------------------------------

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:                           # --------------------------------------------------          
                for y in f.outputs:                       #  y() : y가 약한 참조이기 때문                
                    y().grad = None  # y is weakref       #  y().grad = None 코드가 실행되면 참조카운트가 0이 되어 미분값 
                                                          #                  데이터가 메모리에서 삭제됨
                                                          # --------------------------------------------------
x = Variable(np.array(1.0))
y = sin(x)
y.backward(retain_grad=True)
```

![그림 30-4](/assets/images/img/그림 30-4.png)


