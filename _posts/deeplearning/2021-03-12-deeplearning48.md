---
layout: page
title: "밑바닥부터 시작하는 딥러닝 48단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 48단계"
categories: deeplearnig
comments: true
published: true
---
# 다중 클래스 분류      

```python
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
```


```python
# 스파이럴 데이터셋
import dezero

x, t = dezero.datasets.get_spiral(train=True)   # train=True면 학습용 데이터, False면 테스트용 데이터를 반환
print(x.shape)
print(t.shape)

print(x[10], t[10])
print(x[110], t[110])
```

    (300, 2)
    (300,)
    [-0.12995958 -0.00324155] 1
    [ 0.3282343  -0.54941994] 0
    

![그림 48-1](/assets/images/img/그림 48-1.png)

미니배치 : 데이터가 많을 때는 모든 데이터를 한꺼번에 처리하는 대신 조금씩 무작위로 모아서 처리하는데 이때 모아서 처리하는 데이터 뭉치


```python
import math
import numpy as np
import matplotlib.pyplot as plt
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

# Hyperparameters
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# 데이터 읽기 / 모델, 옵티마이저 생성
x, t = dezero.datasets.get_spiral(train=True)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(x)
# print(f'data_size {data_size}')
max_iter = math.ceil(data_size / batch_size) # 소수점 반올림
# print(f'max_iter {max_iter}')


epochs = []
losses = []
for epoch in range(max_epoch):
    # Shuffle index for data
    index = np.random.permutation(data_size)    # 무작위로 정렬된 색인 리스트
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    # Print loss every epoch
    avg_loss = sum_loss / data_size
    print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))
    epochs.append(epoch+1)
    losses.append(avg_loss)
plt.figure(1)
plt.plot(epochs, losses)

plt.figure(2)
# Plot boundary area the model predict
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]

with dezero.no_grad():
    score = model(X)
predict_cls = np.argmax(score.data, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)

# Plot data points of the dataset
N, CLS_NUM = 100, 3
markers = ['o', 'x', '^']
colors = ['orange', 'blue', 'green']
for i in range(len(x)):
    c = t[i]
    plt.scatter(x[i][0], x[i][1], s=40,  marker=markers[c], c=colors[c])
plt.show()
```

    epoch 1, loss 1.13
    epoch 2, loss 1.05
    epoch 3, loss 0.95
    epoch 4, loss 0.92
    epoch 5, loss 0.87
    epoch 6, loss 0.89
    epoch 7, loss 0.84
    epoch 8, loss 0.78
    epoch 9, loss 0.80
    epoch 10, loss 0.79
    epoch 11, loss 0.78
    epoch 12, loss 0.76
    epoch 13, loss 0.77
    epoch 14, loss 0.76
    epoch 15, loss 0.76
    epoch 16, loss 0.77
    epoch 17, loss 0.78
    epoch 18, loss 0.74
    epoch 19, loss 0.74
    epoch 20, loss 0.72
    epoch 21, loss 0.73
    epoch 22, loss 0.74
    epoch 23, loss 0.77
    epoch 24, loss 0.73
    epoch 25, loss 0.74
    epoch 26, loss 0.74
    epoch 27, loss 0.72
    epoch 28, loss 0.72
    epoch 29, loss 0.72
    epoch 30, loss 0.73
    epoch 31, loss 0.71
    epoch 32, loss 0.72
    epoch 33, loss 0.72
    epoch 34, loss 0.71
    epoch 35, loss 0.72
    epoch 36, loss 0.71
    epoch 37, loss 0.71
    epoch 38, loss 0.70
    epoch 39, loss 0.71
    epoch 40, loss 0.70
    epoch 41, loss 0.71
    epoch 42, loss 0.70
    epoch 43, loss 0.70
    epoch 44, loss 0.70
    epoch 45, loss 0.69
    epoch 46, loss 0.69
    epoch 47, loss 0.71
    epoch 48, loss 0.70
    epoch 49, loss 0.69
    epoch 50, loss 0.69
    epoch 51, loss 0.68
    epoch 52, loss 0.67
    epoch 53, loss 0.68
    epoch 54, loss 0.70
    epoch 55, loss 0.68
    epoch 56, loss 0.66
    epoch 57, loss 0.67
    epoch 58, loss 0.66
    epoch 59, loss 0.64
    epoch 60, loss 0.64
    epoch 61, loss 0.64
    epoch 62, loss 0.63
    epoch 63, loss 0.63
    epoch 64, loss 0.61
    epoch 65, loss 0.61
    epoch 66, loss 0.60
    epoch 67, loss 0.62
    epoch 68, loss 0.59
    epoch 69, loss 0.60
    epoch 70, loss 0.57
    epoch 71, loss 0.58
    epoch 72, loss 0.57
    epoch 73, loss 0.56
    epoch 74, loss 0.56
    epoch 75, loss 0.55
    epoch 76, loss 0.55
    epoch 77, loss 0.55
    epoch 78, loss 0.54
    epoch 79, loss 0.53
    epoch 80, loss 0.53
    epoch 81, loss 0.52
    epoch 82, loss 0.53
    epoch 83, loss 0.52
    epoch 84, loss 0.49
    epoch 85, loss 0.50
    epoch 86, loss 0.49
    epoch 87, loss 0.49
    epoch 88, loss 0.48
    epoch 89, loss 0.47
    epoch 90, loss 0.47
    epoch 91, loss 0.46
    epoch 92, loss 0.46
    epoch 93, loss 0.45
    epoch 94, loss 0.44
    epoch 95, loss 0.45
    epoch 96, loss 0.44
    epoch 97, loss 0.43
    epoch 98, loss 0.43
    epoch 99, loss 0.42
    epoch 100, loss 0.43
    epoch 101, loss 0.42
    epoch 102, loss 0.41
    epoch 103, loss 0.42
    epoch 104, loss 0.41
    epoch 105, loss 0.40
    epoch 106, loss 0.40
    epoch 107, loss 0.40
    epoch 108, loss 0.39
    epoch 109, loss 0.38
    epoch 110, loss 0.39
    epoch 111, loss 0.38
    epoch 112, loss 0.38
    epoch 113, loss 0.38
    epoch 114, loss 0.36
    epoch 115, loss 0.36
    epoch 116, loss 0.36
    epoch 117, loss 0.36
    epoch 118, loss 0.36
    epoch 119, loss 0.35
    epoch 120, loss 0.35
    epoch 121, loss 0.36
    epoch 122, loss 0.34
    epoch 123, loss 0.35
    epoch 124, loss 0.33
    epoch 125, loss 0.33
    epoch 126, loss 0.32
    epoch 127, loss 0.34
    epoch 128, loss 0.32
    epoch 129, loss 0.33
    epoch 130, loss 0.31
    epoch 131, loss 0.30
    epoch 132, loss 0.31
    epoch 133, loss 0.31
    epoch 134, loss 0.30
    epoch 135, loss 0.29
    epoch 136, loss 0.29
    epoch 137, loss 0.29
    epoch 138, loss 0.28
    epoch 139, loss 0.29
    epoch 140, loss 0.28
    epoch 141, loss 0.27
    epoch 142, loss 0.27
    epoch 143, loss 0.28
    epoch 144, loss 0.27
    epoch 145, loss 0.27
    epoch 146, loss 0.26
    epoch 147, loss 0.26
    epoch 148, loss 0.26
    epoch 149, loss 0.26
    epoch 150, loss 0.25
    epoch 151, loss 0.25
    epoch 152, loss 0.25
    epoch 153, loss 0.24
    epoch 154, loss 0.24
    epoch 155, loss 0.24
    epoch 156, loss 0.24
    epoch 157, loss 0.24
    epoch 158, loss 0.24
    epoch 159, loss 0.23
    epoch 160, loss 0.23
    epoch 161, loss 0.23
    epoch 162, loss 0.23
    epoch 163, loss 0.23
    epoch 164, loss 0.22
    epoch 165, loss 0.22
    epoch 166, loss 0.22
    epoch 167, loss 0.21
    epoch 168, loss 0.22
    epoch 169, loss 0.22
    epoch 170, loss 0.21
    epoch 171, loss 0.21
    epoch 172, loss 0.22
    epoch 173, loss 0.22
    epoch 174, loss 0.21
    epoch 175, loss 0.21
    epoch 176, loss 0.20
    epoch 177, loss 0.21
    epoch 178, loss 0.20
    epoch 179, loss 0.20
    epoch 180, loss 0.20
    epoch 181, loss 0.20
    epoch 182, loss 0.19
    epoch 183, loss 0.20
    epoch 184, loss 0.19
    epoch 185, loss 0.19
    epoch 186, loss 0.19
    epoch 187, loss 0.19
    epoch 188, loss 0.19
    epoch 189, loss 0.19
    epoch 190, loss 0.19
    epoch 191, loss 0.19
    epoch 192, loss 0.19
    epoch 193, loss 0.18
    epoch 194, loss 0.19
    epoch 195, loss 0.18
    epoch 196, loss 0.18
    epoch 197, loss 0.18
    epoch 198, loss 0.18
    epoch 199, loss 0.19
    epoch 200, loss 0.18
    epoch 201, loss 0.17
    epoch 202, loss 0.18
    epoch 203, loss 0.18
    epoch 204, loss 0.17
    epoch 205, loss 0.18
    epoch 206, loss 0.17
    epoch 207, loss 0.17
    epoch 208, loss 0.17
    epoch 209, loss 0.17
    epoch 210, loss 0.17
    epoch 211, loss 0.17
    epoch 212, loss 0.17
    epoch 213, loss 0.18
    epoch 214, loss 0.17
    epoch 215, loss 0.17
    epoch 216, loss 0.17
    epoch 217, loss 0.17
    epoch 218, loss 0.17
    epoch 219, loss 0.16
    epoch 220, loss 0.17
    epoch 221, loss 0.16
    epoch 222, loss 0.16
    epoch 223, loss 0.16
    epoch 224, loss 0.16
    epoch 225, loss 0.16
    epoch 226, loss 0.16
    epoch 227, loss 0.17
    epoch 228, loss 0.18
    epoch 229, loss 0.16
    epoch 230, loss 0.16
    epoch 231, loss 0.15
    epoch 232, loss 0.16
    epoch 233, loss 0.17
    epoch 234, loss 0.16
    epoch 235, loss 0.16
    epoch 236, loss 0.15
    epoch 237, loss 0.16
    epoch 238, loss 0.16
    epoch 239, loss 0.16
    epoch 240, loss 0.16
    epoch 241, loss 0.15
    epoch 242, loss 0.15
    epoch 243, loss 0.15
    epoch 244, loss 0.15
    epoch 245, loss 0.15
    epoch 246, loss 0.15
    epoch 247, loss 0.15
    epoch 248, loss 0.15
    epoch 249, loss 0.15
    epoch 250, loss 0.15
    epoch 251, loss 0.15
    epoch 252, loss 0.15
    epoch 253, loss 0.15
    epoch 254, loss 0.15
    epoch 255, loss 0.15
    epoch 256, loss 0.15
    epoch 257, loss 0.14
    epoch 258, loss 0.15
    epoch 259, loss 0.14
    epoch 260, loss 0.15
    epoch 261, loss 0.15
    epoch 262, loss 0.15
    epoch 263, loss 0.14
    epoch 264, loss 0.14
    epoch 265, loss 0.14
    epoch 266, loss 0.14
    epoch 267, loss 0.14
    epoch 268, loss 0.14
    epoch 269, loss 0.14
    epoch 270, loss 0.14
    epoch 271, loss 0.14
    epoch 272, loss 0.14
    epoch 273, loss 0.14
    epoch 274, loss 0.14
    epoch 275, loss 0.14
    epoch 276, loss 0.14
    epoch 277, loss 0.14
    epoch 278, loss 0.14
    epoch 279, loss 0.14
    epoch 280, loss 0.13
    epoch 281, loss 0.13
    epoch 282, loss 0.14
    epoch 283, loss 0.13
    epoch 284, loss 0.13
    epoch 285, loss 0.13
    epoch 286, loss 0.13
    epoch 287, loss 0.14
    epoch 288, loss 0.13
    epoch 289, loss 0.13
    epoch 290, loss 0.13
    epoch 291, loss 0.13
    epoch 292, loss 0.13
    epoch 293, loss 0.14
    epoch 294, loss 0.13
    epoch 295, loss 0.13
    epoch 296, loss 0.13
    epoch 297, loss 0.13
    epoch 298, loss 0.12
    epoch 299, loss 0.13
    epoch 300, loss 0.13
    


    
![svg](/assets/images/img/output_48_1.png)
    



    
![svg](/assets/images/img/output_48_2.png)
    

