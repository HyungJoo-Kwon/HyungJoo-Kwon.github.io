---
layout: page
title: "밑바닥부터 시작하는 딥러닝 34단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 34단계"
categories: deeplearnig
comments: true
published: true
---
# sin 함수 고차 미분    
```python
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F

x = Variable(np.linspace(-7, 7, 200))       # -7부터 7까지 200등분 균일하게
y = F.sin(x)
y.backward(create_graph=True)

logs = [y.data]

for i in range(3):              # n차 미분
    logs.append(x.grad.data)
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

print(f'len(logs) {len(logs)}')


labels = ["y=sin(x)", "y'", "y''", "y'''"]
for i, v in enumerate(logs):
    plt.plot(x.data, logs[i], label=labels[i])
plt.legend(loc='lower right')
plt.show()

# sin(x) -> cos(x) -> -sin(x) -> -cos(x)
```

    len(logs) 4
    


    
![png](/assets/images/output_34_1.png)
    

