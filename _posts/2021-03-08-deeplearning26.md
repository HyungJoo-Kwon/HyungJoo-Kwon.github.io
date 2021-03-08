---
layout: page
title: "밑바닥부터 시작하는 딥러닝 26단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 26단계"
categories: deeplearnig
comments: true
published: true
---
# 계산 그래프 시각화    

```python
import numpy as np 
from dezero import Variable
from dezero.utils import get_dot_graph

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
y = x0 + x1

x0.name = 'x0'
x1.name = 'x1'
y.name = 'y'

txt = get_dot_graph(y, verbose=False)
print(txt)

with open('sample.dot', 'w') as o:
    o.write(txt)
```

    digraph g {
    1792489929024 [label="y", color=orange, style=filled]
    1792489929552 [label="Add", color=lightblue, style=filled, shape=box]
    1792489929648 -> 1792489929552
    1792489927584 -> 1792489929552
    1792489929552 -> 1792489929024
    1792489929648 [label="x0", color=orange, style=filled]
    1792489927584 [label="x1", color=orange, style=filled]
    }
    

![26-1](/assets/images/img/그림 26-1.png)


```python
def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)              # id() : 주어진 객체의 ID를 반환, 

x = Variable(np.random.randn(2,3))
x.name = 'x'
print(_dot_var(x))
print(_dot_var(x, verbose=True))
```

    1792489170400 [label="x", color=orange, style=filled]
    
    1792489170400 [label="x: (2, 3) float64", color=orange, style=filled]
    
    


```python
def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y))  # y는 약한 참조
    return txt

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
y = x0 + x1 
txt = _dot_func(y.creator)
print(txt)
```

    1792491574560 [label="Add", color=lightblue, style=filled, shape=box]
    1792491447680 -> 1792491574560
    1792491573360 -> 1792491574560
    1792491574560 -> 1792489344592
    
    

![그림 26-2](/assets/images/img/그림 26-2.png)


```python
def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            # funcs.sort(key=lambda x: x.generation) 노드를 추적하는 순서는 문제가 되지 않으므로 generation 값으로 정렬을 안해도.
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)
            
            if x.creator is not None:
                add_func(x.creator)
    
    return 'digraph g {\n' + txt + '}'
```


```python
import os
import subprocess

def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    # dot 데이터를 파일에 저장
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')  # 사용자의 홈 디렉터리를 뜻하는 '~'를 절대 경로로 풀어줌.
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    # dot 명령 호출
    extension = os.path.splitext(to_file)[1][1:]
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)
```


```python
import numpy as np 
from dezero import Variable
from dezero.utils import plot_dot_graph

def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = goldstein(x, y)
z.backward()

x.name = 'x'
y.name = 'y'
z.name = 'z'
plot_dot_graph(z, verbose=False, to_file='goldstein.png')
```




    
![png](/assets/images/output_7_0.png)
    



Need the dot binary from the graphviz package (www.graphviz.org).
'''
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph


def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z


x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = goldstein(x, y)
z.backward()

x.name = 'x'
y.name = 'y'
z.name = 'z'
plot_dot_graph(z, verbose=False, to_file='goldstein.png')


