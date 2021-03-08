---
layout: page
title: "밑바닥부터 시작하는 딥러닝 32단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 32단계"
categories: deeplearnig
comments: true
published: true
---
# 고차 미분(구현)   

```python
# dezero/core.py 새로 구현
class Variable: 
    ...
     def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            # self.grad = np.ones_like(self.data)
            self.grad = Variable(np.ones_like(self.data))   # self.grad가 Variable 인스턴스를 담게 됨.

```


```python
def backward(self, retain_grad=False, create_graph=False):      # create_graph 인수 추가, False면 역전파로 인한 계산은 역전파 비활성 모드에서 이뤄진다. 
        if self.grad is None:
            xp = dezero.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

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
            gys = [output().grad for output in f.outputs]  # output is weakref

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y is weakref
```


