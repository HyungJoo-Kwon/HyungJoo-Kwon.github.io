---
layout: page
title: "밑바닥부터 시작하는 딥러닝 50단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 50단계"
categories: deeplearnig
comments: true
published: true
---
# DataLoader    

```python
# Dataset 클래스에서 미니배치를 뽑아주는 DataLoader 클래스
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
```


```python
# 반복자 : 원소를 반복하여 꺼내줌
# 파이썬의 반복자는 리스트나 튜플 등 여러 원소를 담고 있는 데이터 타입으로부터 데이터를 순차적으로 추출하는 기능 제공
t = [1,2,3]
x = iter(t)
next(x)     # 1
next(x)     # 2
next(x)     # 3
# next(x) : error 4번째 실행에서는 원소가 더는 존재하지 않기 때문에 stopiteration 예외 발생
```




    3




```python
# 파이썬에서 예시 반복자 구현
class MyIterator:
    def __init__(self, max_cnt):
        self.max_cnt = max_cnt
        self.cnt = 0
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt == self.max_cnt:
            raise StopIteration()

        self.cnt += 1
        return self.cnt

# __iter__ : 자기 자신(self)을 반환하도록 함.
# __next__ : 다음 원소를 반환하도록 함

obj = MyIterator(5)
for x in obj:
    print(x)
```

    1
    2
    3
    4
    5
    


```python
# dezero/dataloaders.py
import math
import random
import numpy as np

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, gpu=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)
        self.gpu = gpu

        self.reset()

    def reset(self):
        self.iteration = 0          # 반복 횟수 초기화
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))       # 데이터 shuffle
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size:(i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]

        xp = cuda.cupy if self.gpu else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()

    def to_cpu(self):
        self.gpu = False

    def to_gpu(self):
        self.gpu = True
```


```python
# DataLoader 사용
from dezero.datasets import Spiral
from dezero import DataLoader

batch_size = 10
max_epoch = 1

train_set = Spiral(train=True)
test_set = Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

for epoch in range(max_epoch):
    for x, t in train_loader:
        print(x.shape, t.shape)     # 훈련 데이터
        break

    for x, t in test_loader:
        print(x.shape, t.shape)     # 테스트 데이터
        break
    

```

    (10, 2) (10,)
    (10, 2) (10,)
    


```python
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP


max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = dezero.datasets.Spiral(train=True)
test_set = dezero.datasets.Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

epochs = []
train_losses = []
test_losses = []
train_accuracys = []
test_accuracys = []

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    
    print('train loss: {:.4f}, accuracy: {:.4f}'.format(
        sum_loss / len(train_set), sum_acc / len(train_set)))

    print('epoch: {}'.format(epoch+1))    
    train_losses.append(sum_loss / len(train_set))
    train_accuracys.append(sum_acc / len(train_set))

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():                          # 기울기 불필요 모드
        for x, t in test_loader:                    # 테스트용 미니배치 데이터
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)                  # 테스트 데이터의 인식 정확도
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print('test loss: {:.4f}, accuracy: {:.4f}'.format(
        sum_loss / len(test_set), sum_acc / len(test_set))) 
    epochs.append(epoch+1)
    test_losses.append(sum_loss / len(test_set))
    test_accuracys.append(sum_acc / len(test_set))

```

    train loss: 1.0944, accuracy: 0.4033
    epoch: 1
    test loss: 1.0468, accuracy: 0.3267
    train loss: 0.9882, accuracy: 0.4933
    epoch: 2
    test loss: 0.9729, accuracy: 0.4333
    train loss: 0.9403, accuracy: 0.5133
    epoch: 3
    test loss: 0.8965, accuracy: 0.6233
    train loss: 0.8820, accuracy: 0.5700
    epoch: 4
    test loss: 0.8771, accuracy: 0.5967
    train loss: 0.8617, accuracy: 0.5600
    epoch: 5
    test loss: 0.8670, accuracy: 0.5867
    train loss: 0.8313, accuracy: 0.5300
    epoch: 6
    test loss: 0.8654, accuracy: 0.6000
    train loss: 0.8086, accuracy: 0.5833
    epoch: 7
    test loss: 0.7950, accuracy: 0.5600
    train loss: 0.7948, accuracy: 0.5733
    epoch: 8
    test loss: 0.7921, accuracy: 0.5900
    train loss: 0.7728, accuracy: 0.5500
    epoch: 9
    test loss: 0.7718, accuracy: 0.5300
    train loss: 0.7643, accuracy: 0.5633
    epoch: 10
    test loss: 0.7796, accuracy: 0.5800
    train loss: 0.7862, accuracy: 0.5600
    epoch: 11
    test loss: 0.7701, accuracy: 0.5633
    train loss: 0.7914, accuracy: 0.5500
    epoch: 12
    test loss: 0.8218, accuracy: 0.6067
    train loss: 0.7633, accuracy: 0.5567
    epoch: 13
    test loss: 0.7757, accuracy: 0.5800
    train loss: 0.7612, accuracy: 0.5833
    epoch: 14
    test loss: 0.7800, accuracy: 0.5667
    train loss: 0.7408, accuracy: 0.5667
    epoch: 15
    test loss: 0.7654, accuracy: 0.5867
    train loss: 0.7453, accuracy: 0.5733
    epoch: 16
    test loss: 0.7648, accuracy: 0.6033
    train loss: 0.7843, accuracy: 0.5667
    epoch: 17
    test loss: 0.7415, accuracy: 0.5267
    train loss: 0.7592, accuracy: 0.5633
    epoch: 18
    test loss: 0.7388, accuracy: 0.5267
    train loss: 0.7396, accuracy: 0.5567
    epoch: 19
    test loss: 0.7791, accuracy: 0.5667
    train loss: 0.7376, accuracy: 0.5533
    epoch: 20
    test loss: 0.7478, accuracy: 0.5200
    train loss: 0.7469, accuracy: 0.5600
    epoch: 21
    test loss: 0.7617, accuracy: 0.6100
    train loss: 0.7349, accuracy: 0.5667
    epoch: 22
    test loss: 0.7476, accuracy: 0.5967
    train loss: 0.7650, accuracy: 0.5600
    epoch: 23
    test loss: 0.7336, accuracy: 0.5433
    train loss: 0.7495, accuracy: 0.5667
    epoch: 24
    test loss: 0.7366, accuracy: 0.5500
    train loss: 0.7414, accuracy: 0.5700
    epoch: 25
    test loss: 0.7516, accuracy: 0.6133
    train loss: 0.7298, accuracy: 0.5900
    epoch: 26
    test loss: 0.7327, accuracy: 0.5533
    train loss: 0.7297, accuracy: 0.5700
    epoch: 27
    test loss: 0.7356, accuracy: 0.5933
    train loss: 0.7267, accuracy: 0.5900
    epoch: 28
    test loss: 0.7240, accuracy: 0.5233
    train loss: 0.7359, accuracy: 0.5800
    epoch: 29
    test loss: 0.7203, accuracy: 0.5533
    train loss: 0.7273, accuracy: 0.5600
    epoch: 30
    test loss: 0.7382, accuracy: 0.5433
    train loss: 0.7338, accuracy: 0.5867
    epoch: 31
    test loss: 0.7492, accuracy: 0.5933
    train loss: 0.7092, accuracy: 0.5900
    epoch: 32
    test loss: 0.7241, accuracy: 0.5500
    train loss: 0.6970, accuracy: 0.5767
    epoch: 33
    test loss: 0.7483, accuracy: 0.6167
    train loss: 0.7048, accuracy: 0.5967
    epoch: 34
    test loss: 0.7170, accuracy: 0.5433
    train loss: 0.7069, accuracy: 0.5967
    epoch: 35
    test loss: 0.7164, accuracy: 0.5467
    train loss: 0.7199, accuracy: 0.5800
    epoch: 36
    test loss: 0.7318, accuracy: 0.6133
    train loss: 0.7035, accuracy: 0.6167
    epoch: 37
    test loss: 0.7266, accuracy: 0.5900
    train loss: 0.6992, accuracy: 0.5767
    epoch: 38
    test loss: 0.7419, accuracy: 0.6000
    train loss: 0.7103, accuracy: 0.6000
    epoch: 39
    test loss: 0.7077, accuracy: 0.5333
    train loss: 0.7064, accuracy: 0.5733
    epoch: 40
    test loss: 0.7335, accuracy: 0.5900
    train loss: 0.7022, accuracy: 0.5767
    epoch: 41
    test loss: 0.7087, accuracy: 0.6100
    train loss: 0.7047, accuracy: 0.6000
    epoch: 42
    test loss: 0.6951, accuracy: 0.5800
    train loss: 0.7071, accuracy: 0.6000
    epoch: 43
    test loss: 0.7009, accuracy: 0.6133
    train loss: 0.7041, accuracy: 0.6033
    epoch: 44
    test loss: 0.6914, accuracy: 0.5833
    train loss: 0.6854, accuracy: 0.5767
    epoch: 45
    test loss: 0.7122, accuracy: 0.6100
    train loss: 0.6771, accuracy: 0.6067
    epoch: 46
    test loss: 0.6874, accuracy: 0.5600
    train loss: 0.6781, accuracy: 0.5900
    epoch: 47
    test loss: 0.6995, accuracy: 0.6467
    train loss: 0.6874, accuracy: 0.5933
    epoch: 48
    test loss: 0.7021, accuracy: 0.6300
    train loss: 0.6641, accuracy: 0.6000
    epoch: 49
    test loss: 0.6897, accuracy: 0.6000
    train loss: 0.6683, accuracy: 0.6200
    epoch: 50
    test loss: 0.6850, accuracy: 0.5867
    train loss: 0.6554, accuracy: 0.6033
    epoch: 51
    test loss: 0.6698, accuracy: 0.6367
    train loss: 0.6528, accuracy: 0.6200
    epoch: 52
    test loss: 0.6789, accuracy: 0.5800
    train loss: 0.6500, accuracy: 0.6467
    epoch: 53
    test loss: 0.6691, accuracy: 0.6467
    train loss: 0.6449, accuracy: 0.6267
    epoch: 54
    test loss: 0.6562, accuracy: 0.6633
    train loss: 0.6549, accuracy: 0.6500
    epoch: 55
    test loss: 0.6892, accuracy: 0.6467
    train loss: 0.6504, accuracy: 0.6400
    epoch: 56
    test loss: 0.6421, accuracy: 0.6167
    train loss: 0.6396, accuracy: 0.6667
    epoch: 57
    test loss: 0.6451, accuracy: 0.6167
    train loss: 0.6251, accuracy: 0.6733
    epoch: 58
    test loss: 0.6415, accuracy: 0.6200
    train loss: 0.6150, accuracy: 0.6433
    epoch: 59
    test loss: 0.6285, accuracy: 0.6267
    train loss: 0.6122, accuracy: 0.6633
    epoch: 60
    test loss: 0.6169, accuracy: 0.6400
    train loss: 0.5987, accuracy: 0.6767
    epoch: 61
    test loss: 0.6183, accuracy: 0.6167
    train loss: 0.5920, accuracy: 0.6667
    epoch: 62
    test loss: 0.6076, accuracy: 0.6967
    train loss: 0.5914, accuracy: 0.6700
    epoch: 63
    test loss: 0.6020, accuracy: 0.6300
    train loss: 0.5911, accuracy: 0.6867
    epoch: 64
    test loss: 0.5981, accuracy: 0.6533
    train loss: 0.5626, accuracy: 0.7000
    epoch: 65
    test loss: 0.6124, accuracy: 0.7100
    train loss: 0.5707, accuracy: 0.7067
    epoch: 66
    test loss: 0.5760, accuracy: 0.6567
    train loss: 0.5574, accuracy: 0.6900
    epoch: 67
    test loss: 0.5834, accuracy: 0.6967
    train loss: 0.5560, accuracy: 0.7133
    epoch: 68
    test loss: 0.5589, accuracy: 0.6933
    train loss: 0.5355, accuracy: 0.7267
    epoch: 69
    test loss: 0.5530, accuracy: 0.6967
    train loss: 0.5303, accuracy: 0.7300
    epoch: 70
    test loss: 0.5380, accuracy: 0.7267
    train loss: 0.5207, accuracy: 0.7167
    epoch: 71
    test loss: 0.5315, accuracy: 0.7367
    train loss: 0.5094, accuracy: 0.7533
    epoch: 72
    test loss: 0.5340, accuracy: 0.7367
    train loss: 0.5110, accuracy: 0.7400
    epoch: 73
    test loss: 0.5127, accuracy: 0.7433
    train loss: 0.4921, accuracy: 0.7500
    epoch: 74
    test loss: 0.5051, accuracy: 0.7400
    train loss: 0.4882, accuracy: 0.7700
    epoch: 75
    test loss: 0.5268, accuracy: 0.7467
    train loss: 0.4859, accuracy: 0.7533
    epoch: 76
    test loss: 0.4997, accuracy: 0.7400
    train loss: 0.4701, accuracy: 0.7767
    epoch: 77
    test loss: 0.4814, accuracy: 0.7800
    train loss: 0.4574, accuracy: 0.7733
    epoch: 78
    test loss: 0.4736, accuracy: 0.7700
    train loss: 0.4547, accuracy: 0.7533
    epoch: 79
    test loss: 0.4665, accuracy: 0.7900
    train loss: 0.4533, accuracy: 0.7667
    epoch: 80
    test loss: 0.4601, accuracy: 0.7900
    train loss: 0.4431, accuracy: 0.7967
    epoch: 81
    test loss: 0.4496, accuracy: 0.7900
    train loss: 0.4309, accuracy: 0.8167
    epoch: 82
    test loss: 0.4421, accuracy: 0.7867
    train loss: 0.4282, accuracy: 0.8167
    epoch: 83
    test loss: 0.4431, accuracy: 0.7833
    train loss: 0.4132, accuracy: 0.8167
    epoch: 84
    test loss: 0.4353, accuracy: 0.8200
    train loss: 0.4132, accuracy: 0.8167
    epoch: 85
    test loss: 0.4260, accuracy: 0.8367
    train loss: 0.4047, accuracy: 0.8167
    epoch: 86
    test loss: 0.4248, accuracy: 0.8400
    train loss: 0.3968, accuracy: 0.8367
    epoch: 87
    test loss: 0.4031, accuracy: 0.8333
    train loss: 0.3942, accuracy: 0.8267
    epoch: 88
    test loss: 0.4091, accuracy: 0.8333
    train loss: 0.3746, accuracy: 0.8333
    epoch: 89
    test loss: 0.3962, accuracy: 0.8100
    train loss: 0.3772, accuracy: 0.8333
    epoch: 90
    test loss: 0.3831, accuracy: 0.8300
    train loss: 0.3669, accuracy: 0.8467
    epoch: 91
    test loss: 0.3844, accuracy: 0.8433
    train loss: 0.3554, accuracy: 0.8600
    epoch: 92
    test loss: 0.3752, accuracy: 0.8567
    train loss: 0.3491, accuracy: 0.8700
    epoch: 93
    test loss: 0.3720, accuracy: 0.8400
    train loss: 0.3440, accuracy: 0.8533
    epoch: 94
    test loss: 0.3595, accuracy: 0.8600
    train loss: 0.3408, accuracy: 0.8633
    epoch: 95
    test loss: 0.3533, accuracy: 0.8700
    train loss: 0.3336, accuracy: 0.8600
    epoch: 96
    test loss: 0.3515, accuracy: 0.8800
    train loss: 0.3295, accuracy: 0.8700
    epoch: 97
    test loss: 0.3437, accuracy: 0.8933
    train loss: 0.3208, accuracy: 0.9133
    epoch: 98
    test loss: 0.3433, accuracy: 0.8533
    train loss: 0.3212, accuracy: 0.8667
    epoch: 99
    test loss: 0.3302, accuracy: 0.8633
    train loss: 0.3146, accuracy: 0.8700
    epoch: 100
    test loss: 0.3248, accuracy: 0.8600
    train loss: 0.3090, accuracy: 0.8633
    epoch: 101
    test loss: 0.3268, accuracy: 0.8833
    train loss: 0.3024, accuracy: 0.8933
    epoch: 102
    test loss: 0.3195, accuracy: 0.8533
    train loss: 0.3009, accuracy: 0.8933
    epoch: 103
    test loss: 0.3122, accuracy: 0.8900
    train loss: 0.2930, accuracy: 0.9067
    epoch: 104
    test loss: 0.3087, accuracy: 0.8633
    train loss: 0.2881, accuracy: 0.9067
    epoch: 105
    test loss: 0.3111, accuracy: 0.8567
    train loss: 0.2785, accuracy: 0.9167
    epoch: 106
    test loss: 0.3026, accuracy: 0.8867
    train loss: 0.2786, accuracy: 0.8933
    epoch: 107
    test loss: 0.2952, accuracy: 0.8833
    train loss: 0.2756, accuracy: 0.9100
    epoch: 108
    test loss: 0.2912, accuracy: 0.8800
    train loss: 0.2744, accuracy: 0.9200
    epoch: 109
    test loss: 0.2965, accuracy: 0.8667
    train loss: 0.2705, accuracy: 0.9267
    epoch: 110
    test loss: 0.2856, accuracy: 0.8933
    train loss: 0.2645, accuracy: 0.9067
    epoch: 111
    test loss: 0.2814, accuracy: 0.8867
    train loss: 0.2624, accuracy: 0.9100
    epoch: 112
    test loss: 0.2835, accuracy: 0.8700
    train loss: 0.2608, accuracy: 0.9067
    epoch: 113
    test loss: 0.2752, accuracy: 0.8933
    train loss: 0.2524, accuracy: 0.9167
    epoch: 114
    test loss: 0.2740, accuracy: 0.9033
    train loss: 0.2484, accuracy: 0.9167
    epoch: 115
    test loss: 0.2688, accuracy: 0.8967
    train loss: 0.2480, accuracy: 0.9200
    epoch: 116
    test loss: 0.2768, accuracy: 0.8600
    train loss: 0.2445, accuracy: 0.9133
    epoch: 117
    test loss: 0.2635, accuracy: 0.9067
    train loss: 0.2444, accuracy: 0.9267
    epoch: 118
    test loss: 0.2593, accuracy: 0.8967
    train loss: 0.2379, accuracy: 0.9267
    epoch: 119
    test loss: 0.2576, accuracy: 0.8867
    train loss: 0.2361, accuracy: 0.9200
    epoch: 120
    test loss: 0.2573, accuracy: 0.9200
    train loss: 0.2372, accuracy: 0.9333
    epoch: 121
    test loss: 0.2534, accuracy: 0.9067
    train loss: 0.2305, accuracy: 0.9333
    epoch: 122
    test loss: 0.2511, accuracy: 0.8867
    train loss: 0.2277, accuracy: 0.9367
    epoch: 123
    test loss: 0.2531, accuracy: 0.9000
    train loss: 0.2318, accuracy: 0.9233
    epoch: 124
    test loss: 0.2456, accuracy: 0.9067
    train loss: 0.2248, accuracy: 0.9400
    epoch: 125
    test loss: 0.2454, accuracy: 0.9133
    train loss: 0.2229, accuracy: 0.9433
    epoch: 126
    test loss: 0.2427, accuracy: 0.9033
    train loss: 0.2264, accuracy: 0.9167
    epoch: 127
    test loss: 0.2433, accuracy: 0.9000
    train loss: 0.2152, accuracy: 0.9333
    epoch: 128
    test loss: 0.2373, accuracy: 0.9133
    train loss: 0.2176, accuracy: 0.9367
    epoch: 129
    test loss: 0.2402, accuracy: 0.8967
    train loss: 0.2149, accuracy: 0.9200
    epoch: 130
    test loss: 0.2332, accuracy: 0.9033
    train loss: 0.2136, accuracy: 0.9333
    epoch: 131
    test loss: 0.2377, accuracy: 0.8933
    train loss: 0.2143, accuracy: 0.9200
    epoch: 132
    test loss: 0.2304, accuracy: 0.9167
    train loss: 0.2141, accuracy: 0.9267
    epoch: 133
    test loss: 0.2449, accuracy: 0.8833
    train loss: 0.2088, accuracy: 0.9133
    epoch: 134
    test loss: 0.2362, accuracy: 0.8900
    train loss: 0.2046, accuracy: 0.9333
    epoch: 135
    test loss: 0.2278, accuracy: 0.9133
    train loss: 0.2021, accuracy: 0.9333
    epoch: 136
    test loss: 0.2275, accuracy: 0.9200
    train loss: 0.2046, accuracy: 0.9333
    epoch: 137
    test loss: 0.2236, accuracy: 0.9167
    train loss: 0.1991, accuracy: 0.9267
    epoch: 138
    test loss: 0.2207, accuracy: 0.9233
    train loss: 0.1960, accuracy: 0.9433
    epoch: 139
    test loss: 0.2216, accuracy: 0.9267
    train loss: 0.1981, accuracy: 0.9333
    epoch: 140
    test loss: 0.2178, accuracy: 0.9233
    train loss: 0.1945, accuracy: 0.9367
    epoch: 141
    test loss: 0.2194, accuracy: 0.9200
    train loss: 0.1969, accuracy: 0.9233
    epoch: 142
    test loss: 0.2158, accuracy: 0.9233
    train loss: 0.1942, accuracy: 0.9333
    epoch: 143
    test loss: 0.2127, accuracy: 0.9167
    train loss: 0.1886, accuracy: 0.9433
    epoch: 144
    test loss: 0.2288, accuracy: 0.8933
    train loss: 0.1966, accuracy: 0.9133
    epoch: 145
    test loss: 0.2103, accuracy: 0.9167
    train loss: 0.1899, accuracy: 0.9300
    epoch: 146
    test loss: 0.2124, accuracy: 0.9100
    train loss: 0.1890, accuracy: 0.9400
    epoch: 147
    test loss: 0.2104, accuracy: 0.9200
    train loss: 0.1865, accuracy: 0.9333
    epoch: 148
    test loss: 0.2080, accuracy: 0.9267
    train loss: 0.1825, accuracy: 0.9367
    epoch: 149
    test loss: 0.2285, accuracy: 0.8967
    train loss: 0.1890, accuracy: 0.9233
    epoch: 150
    test loss: 0.2152, accuracy: 0.9067
    train loss: 0.1828, accuracy: 0.9400
    epoch: 151
    test loss: 0.2042, accuracy: 0.9300
    train loss: 0.1807, accuracy: 0.9467
    epoch: 152
    test loss: 0.2136, accuracy: 0.9067
    train loss: 0.1795, accuracy: 0.9333
    epoch: 153
    test loss: 0.2076, accuracy: 0.9033
    train loss: 0.1797, accuracy: 0.9300
    epoch: 154
    test loss: 0.2052, accuracy: 0.9200
    train loss: 0.1816, accuracy: 0.9400
    epoch: 155
    test loss: 0.2015, accuracy: 0.9233
    train loss: 0.1791, accuracy: 0.9300
    epoch: 156
    test loss: 0.2012, accuracy: 0.9333
    train loss: 0.1791, accuracy: 0.9367
    epoch: 157
    test loss: 0.1978, accuracy: 0.9267
    train loss: 0.1751, accuracy: 0.9400
    epoch: 158
    test loss: 0.2046, accuracy: 0.9033
    train loss: 0.1719, accuracy: 0.9433
    epoch: 159
    test loss: 0.2039, accuracy: 0.9067
    train loss: 0.1730, accuracy: 0.9433
    epoch: 160
    test loss: 0.1963, accuracy: 0.9300
    train loss: 0.1690, accuracy: 0.9467
    epoch: 161
    test loss: 0.1973, accuracy: 0.9267
    train loss: 0.1676, accuracy: 0.9567
    epoch: 162
    test loss: 0.1951, accuracy: 0.9400
    train loss: 0.1734, accuracy: 0.9433
    epoch: 163
    test loss: 0.1941, accuracy: 0.9300
    train loss: 0.1725, accuracy: 0.9233
    epoch: 164
    test loss: 0.1922, accuracy: 0.9367
    train loss: 0.1665, accuracy: 0.9433
    epoch: 165
    test loss: 0.1951, accuracy: 0.9233
    train loss: 0.1698, accuracy: 0.9433
    epoch: 166
    test loss: 0.1907, accuracy: 0.9333
    train loss: 0.1737, accuracy: 0.9300
    epoch: 167
    test loss: 0.1923, accuracy: 0.9300
    train loss: 0.1661, accuracy: 0.9367
    epoch: 168
    test loss: 0.1944, accuracy: 0.9300
    train loss: 0.1635, accuracy: 0.9567
    epoch: 169
    test loss: 0.1894, accuracy: 0.9233
    train loss: 0.1688, accuracy: 0.9400
    epoch: 170
    test loss: 0.1878, accuracy: 0.9300
    train loss: 0.1675, accuracy: 0.9300
    epoch: 171
    test loss: 0.1899, accuracy: 0.9367
    train loss: 0.1602, accuracy: 0.9500
    epoch: 172
    test loss: 0.1859, accuracy: 0.9400
    train loss: 0.1644, accuracy: 0.9433
    epoch: 173
    test loss: 0.1901, accuracy: 0.9267
    train loss: 0.1597, accuracy: 0.9500
    epoch: 174
    test loss: 0.1878, accuracy: 0.9267
    train loss: 0.1599, accuracy: 0.9400
    epoch: 175
    test loss: 0.1840, accuracy: 0.9367
    train loss: 0.1612, accuracy: 0.9533
    epoch: 176
    test loss: 0.1909, accuracy: 0.9200
    train loss: 0.1582, accuracy: 0.9500
    epoch: 177
    test loss: 0.1829, accuracy: 0.9433
    train loss: 0.1599, accuracy: 0.9433
    epoch: 178
    test loss: 0.1847, accuracy: 0.9333
    train loss: 0.1541, accuracy: 0.9400
    epoch: 179
    test loss: 0.1892, accuracy: 0.9267
    train loss: 0.1539, accuracy: 0.9500
    epoch: 180
    test loss: 0.1814, accuracy: 0.9367
    train loss: 0.1572, accuracy: 0.9400
    epoch: 181
    test loss: 0.1960, accuracy: 0.9333
    train loss: 0.1612, accuracy: 0.9400
    epoch: 182
    test loss: 0.1824, accuracy: 0.9367
    train loss: 0.1533, accuracy: 0.9533
    epoch: 183
    test loss: 0.1836, accuracy: 0.9233
    train loss: 0.1526, accuracy: 0.9533
    epoch: 184
    test loss: 0.1787, accuracy: 0.9433
    train loss: 0.1530, accuracy: 0.9533
    epoch: 185
    test loss: 0.1892, accuracy: 0.9100
    train loss: 0.1558, accuracy: 0.9433
    epoch: 186
    test loss: 0.1840, accuracy: 0.9267
    train loss: 0.1559, accuracy: 0.9400
    epoch: 187
    test loss: 0.1775, accuracy: 0.9367
    train loss: 0.1513, accuracy: 0.9500
    epoch: 188
    test loss: 0.1775, accuracy: 0.9367
    train loss: 0.1491, accuracy: 0.9533
    epoch: 189
    test loss: 0.1850, accuracy: 0.9267
    train loss: 0.1488, accuracy: 0.9500
    epoch: 190
    test loss: 0.1790, accuracy: 0.9333
    train loss: 0.1499, accuracy: 0.9467
    epoch: 191
    test loss: 0.1755, accuracy: 0.9400
    train loss: 0.1469, accuracy: 0.9433
    epoch: 192
    test loss: 0.1788, accuracy: 0.9267
    train loss: 0.1528, accuracy: 0.9433
    epoch: 193
    test loss: 0.1769, accuracy: 0.9367
    train loss: 0.1482, accuracy: 0.9433
    epoch: 194
    test loss: 0.1737, accuracy: 0.9467
    train loss: 0.1425, accuracy: 0.9533
    epoch: 195
    test loss: 0.1770, accuracy: 0.9333
    train loss: 0.1493, accuracy: 0.9433
    epoch: 196
    test loss: 0.1757, accuracy: 0.9300
    train loss: 0.1436, accuracy: 0.9467
    epoch: 197
    test loss: 0.1752, accuracy: 0.9300
    train loss: 0.1449, accuracy: 0.9433
    epoch: 198
    test loss: 0.1721, accuracy: 0.9433
    train loss: 0.1456, accuracy: 0.9467
    epoch: 199
    test loss: 0.1734, accuracy: 0.9300
    train loss: 0.1447, accuracy: 0.9500
    epoch: 200
    test loss: 0.1710, accuracy: 0.9433
    train loss: 0.1426, accuracy: 0.9500
    epoch: 201
    test loss: 0.1707, accuracy: 0.9433
    train loss: 0.1406, accuracy: 0.9600
    epoch: 202
    test loss: 0.1733, accuracy: 0.9400
    train loss: 0.1388, accuracy: 0.9667
    epoch: 203
    test loss: 0.1760, accuracy: 0.9333
    train loss: 0.1427, accuracy: 0.9433
    epoch: 204
    test loss: 0.1696, accuracy: 0.9467
    train loss: 0.1403, accuracy: 0.9567
    epoch: 205
    test loss: 0.1754, accuracy: 0.9433
    train loss: 0.1436, accuracy: 0.9400
    epoch: 206
    test loss: 0.1698, accuracy: 0.9433
    train loss: 0.1410, accuracy: 0.9567
    epoch: 207
    test loss: 0.1723, accuracy: 0.9333
    train loss: 0.1418, accuracy: 0.9567
    epoch: 208
    test loss: 0.1685, accuracy: 0.9400
    train loss: 0.1408, accuracy: 0.9467
    epoch: 209
    test loss: 0.1695, accuracy: 0.9367
    train loss: 0.1382, accuracy: 0.9533
    epoch: 210
    test loss: 0.1676, accuracy: 0.9467
    train loss: 0.1359, accuracy: 0.9633
    epoch: 211
    test loss: 0.1815, accuracy: 0.9367
    train loss: 0.1377, accuracy: 0.9600
    epoch: 212
    test loss: 0.1677, accuracy: 0.9433
    train loss: 0.1350, accuracy: 0.9467
    epoch: 213
    test loss: 0.1669, accuracy: 0.9467
    train loss: 0.1340, accuracy: 0.9600
    epoch: 214
    test loss: 0.1727, accuracy: 0.9367
    train loss: 0.1425, accuracy: 0.9567
    epoch: 215
    test loss: 0.1654, accuracy: 0.9467
    train loss: 0.1404, accuracy: 0.9533
    epoch: 216
    test loss: 0.1685, accuracy: 0.9333
    train loss: 0.1349, accuracy: 0.9567
    epoch: 217
    test loss: 0.1688, accuracy: 0.9367
    train loss: 0.1410, accuracy: 0.9533
    epoch: 218
    test loss: 0.1678, accuracy: 0.9300
    train loss: 0.1342, accuracy: 0.9533
    epoch: 219
    test loss: 0.1663, accuracy: 0.9400
    train loss: 0.1361, accuracy: 0.9500
    epoch: 220
    test loss: 0.1649, accuracy: 0.9433
    train loss: 0.1361, accuracy: 0.9500
    epoch: 221
    test loss: 0.1632, accuracy: 0.9533
    train loss: 0.1318, accuracy: 0.9633
    epoch: 222
    test loss: 0.1661, accuracy: 0.9433
    train loss: 0.1345, accuracy: 0.9567
    epoch: 223
    test loss: 0.1629, accuracy: 0.9467
    train loss: 0.1319, accuracy: 0.9600
    epoch: 224
    test loss: 0.1629, accuracy: 0.9533
    train loss: 0.1334, accuracy: 0.9533
    epoch: 225
    test loss: 0.1620, accuracy: 0.9500
    train loss: 0.1339, accuracy: 0.9633
    epoch: 226
    test loss: 0.1675, accuracy: 0.9367
    train loss: 0.1364, accuracy: 0.9467
    epoch: 227
    test loss: 0.1613, accuracy: 0.9467
    train loss: 0.1319, accuracy: 0.9533
    epoch: 228
    test loss: 0.1616, accuracy: 0.9467
    train loss: 0.1301, accuracy: 0.9600
    epoch: 229
    test loss: 0.1640, accuracy: 0.9300
    train loss: 0.1298, accuracy: 0.9467
    epoch: 230
    test loss: 0.1656, accuracy: 0.9333
    train loss: 0.1289, accuracy: 0.9567
    epoch: 231
    test loss: 0.1633, accuracy: 0.9500
    train loss: 0.1276, accuracy: 0.9600
    epoch: 232
    test loss: 0.1632, accuracy: 0.9433
    train loss: 0.1269, accuracy: 0.9567
    epoch: 233
    test loss: 0.1675, accuracy: 0.9333
    train loss: 0.1311, accuracy: 0.9600
    epoch: 234
    test loss: 0.1630, accuracy: 0.9433
    train loss: 0.1335, accuracy: 0.9500
    epoch: 235
    test loss: 0.1602, accuracy: 0.9467
    train loss: 0.1275, accuracy: 0.9633
    epoch: 236
    test loss: 0.1596, accuracy: 0.9500
    train loss: 0.1227, accuracy: 0.9633
    epoch: 237
    test loss: 0.1619, accuracy: 0.9467
    train loss: 0.1275, accuracy: 0.9600
    epoch: 238
    test loss: 0.1588, accuracy: 0.9433
    train loss: 0.1270, accuracy: 0.9533
    epoch: 239
    test loss: 0.1657, accuracy: 0.9367
    train loss: 0.1251, accuracy: 0.9500
    epoch: 240
    test loss: 0.1583, accuracy: 0.9433
    train loss: 0.1273, accuracy: 0.9533
    epoch: 241
    test loss: 0.1589, accuracy: 0.9500
    train loss: 0.1268, accuracy: 0.9567
    epoch: 242
    test loss: 0.1584, accuracy: 0.9500
    train loss: 0.1242, accuracy: 0.9633
    epoch: 243
    test loss: 0.1580, accuracy: 0.9433
    train loss: 0.1232, accuracy: 0.9633
    epoch: 244
    test loss: 0.1642, accuracy: 0.9433
    train loss: 0.1247, accuracy: 0.9600
    epoch: 245
    test loss: 0.1562, accuracy: 0.9500
    train loss: 0.1273, accuracy: 0.9567
    epoch: 246
    test loss: 0.1555, accuracy: 0.9500
    train loss: 0.1305, accuracy: 0.9500
    epoch: 247
    test loss: 0.1552, accuracy: 0.9533
    train loss: 0.1274, accuracy: 0.9500
    epoch: 248
    test loss: 0.1579, accuracy: 0.9467
    train loss: 0.1282, accuracy: 0.9500
    epoch: 249
    test loss: 0.1589, accuracy: 0.9500
    train loss: 0.1286, accuracy: 0.9600
    epoch: 250
    test loss: 0.1548, accuracy: 0.9533
    train loss: 0.1251, accuracy: 0.9667
    epoch: 251
    test loss: 0.1566, accuracy: 0.9400
    train loss: 0.1213, accuracy: 0.9533
    epoch: 252
    test loss: 0.1615, accuracy: 0.9167
    train loss: 0.1232, accuracy: 0.9633
    epoch: 253
    test loss: 0.1588, accuracy: 0.9433
    train loss: 0.1238, accuracy: 0.9567
    epoch: 254
    test loss: 0.1564, accuracy: 0.9400
    train loss: 0.1212, accuracy: 0.9633
    epoch: 255
    test loss: 0.1571, accuracy: 0.9533
    train loss: 0.1261, accuracy: 0.9600
    epoch: 256
    test loss: 0.1546, accuracy: 0.9433
    train loss: 0.1238, accuracy: 0.9600
    epoch: 257
    test loss: 0.1575, accuracy: 0.9533
    train loss: 0.1224, accuracy: 0.9633
    epoch: 258
    test loss: 0.1583, accuracy: 0.9333
    train loss: 0.1239, accuracy: 0.9533
    epoch: 259
    test loss: 0.1534, accuracy: 0.9467
    train loss: 0.1288, accuracy: 0.9567
    epoch: 260
    test loss: 0.1522, accuracy: 0.9500
    train loss: 0.1245, accuracy: 0.9567
    epoch: 261
    test loss: 0.1530, accuracy: 0.9500
    train loss: 0.1225, accuracy: 0.9600
    epoch: 262
    test loss: 0.1549, accuracy: 0.9433
    train loss: 0.1193, accuracy: 0.9533
    epoch: 263
    test loss: 0.1523, accuracy: 0.9533
    train loss: 0.1177, accuracy: 0.9633
    epoch: 264
    test loss: 0.1553, accuracy: 0.9433
    train loss: 0.1140, accuracy: 0.9633
    epoch: 265
    test loss: 0.1566, accuracy: 0.9500
    train loss: 0.1210, accuracy: 0.9500
    epoch: 266
    test loss: 0.1507, accuracy: 0.9533
    train loss: 0.1224, accuracy: 0.9567
    epoch: 267
    test loss: 0.1511, accuracy: 0.9500
    train loss: 0.1234, accuracy: 0.9500
    epoch: 268
    test loss: 0.1504, accuracy: 0.9533
    train loss: 0.1163, accuracy: 0.9567
    epoch: 269
    test loss: 0.1511, accuracy: 0.9500
    train loss: 0.1191, accuracy: 0.9567
    epoch: 270
    test loss: 0.1507, accuracy: 0.9533
    train loss: 0.1187, accuracy: 0.9567
    epoch: 271
    test loss: 0.1543, accuracy: 0.9367
    train loss: 0.1182, accuracy: 0.9633
    epoch: 272
    test loss: 0.1567, accuracy: 0.9267
    train loss: 0.1137, accuracy: 0.9567
    epoch: 273
    test loss: 0.1490, accuracy: 0.9533
    train loss: 0.1205, accuracy: 0.9467
    epoch: 274
    test loss: 0.1541, accuracy: 0.9400
    train loss: 0.1159, accuracy: 0.9633
    epoch: 275
    test loss: 0.1508, accuracy: 0.9467
    train loss: 0.1151, accuracy: 0.9633
    epoch: 276
    test loss: 0.1518, accuracy: 0.9533
    train loss: 0.1166, accuracy: 0.9633
    epoch: 277
    test loss: 0.1578, accuracy: 0.9233
    train loss: 0.1172, accuracy: 0.9667
    epoch: 278
    test loss: 0.1548, accuracy: 0.9400
    train loss: 0.1134, accuracy: 0.9633
    epoch: 279
    test loss: 0.1499, accuracy: 0.9533
    train loss: 0.1162, accuracy: 0.9533
    epoch: 280
    test loss: 0.1506, accuracy: 0.9500
    train loss: 0.1189, accuracy: 0.9700
    epoch: 281
    test loss: 0.1478, accuracy: 0.9567
    train loss: 0.1188, accuracy: 0.9600
    epoch: 282
    test loss: 0.1484, accuracy: 0.9533
    train loss: 0.1142, accuracy: 0.9667
    epoch: 283
    test loss: 0.1524, accuracy: 0.9433
    train loss: 0.1123, accuracy: 0.9700
    epoch: 284
    test loss: 0.1500, accuracy: 0.9433
    train loss: 0.1192, accuracy: 0.9600
    epoch: 285
    test loss: 0.1508, accuracy: 0.9467
    train loss: 0.1114, accuracy: 0.9600
    epoch: 286
    test loss: 0.1471, accuracy: 0.9533
    train loss: 0.1109, accuracy: 0.9633
    epoch: 287
    test loss: 0.1481, accuracy: 0.9567
    train loss: 0.1143, accuracy: 0.9533
    epoch: 288
    test loss: 0.1534, accuracy: 0.9467
    train loss: 0.1126, accuracy: 0.9600
    epoch: 289
    test loss: 0.1486, accuracy: 0.9500
    train loss: 0.1125, accuracy: 0.9733
    epoch: 290
    test loss: 0.1485, accuracy: 0.9533
    train loss: 0.1136, accuracy: 0.9700
    epoch: 291
    test loss: 0.1472, accuracy: 0.9500
    train loss: 0.1130, accuracy: 0.9600
    epoch: 292
    test loss: 0.1495, accuracy: 0.9433
    train loss: 0.1118, accuracy: 0.9700
    epoch: 293
    test loss: 0.1456, accuracy: 0.9533
    train loss: 0.1111, accuracy: 0.9600
    epoch: 294
    test loss: 0.1591, accuracy: 0.9367
    train loss: 0.1170, accuracy: 0.9533
    epoch: 295
    test loss: 0.1451, accuracy: 0.9533
    train loss: 0.1115, accuracy: 0.9567
    epoch: 296
    test loss: 0.1459, accuracy: 0.9567
    train loss: 0.1096, accuracy: 0.9667
    epoch: 297
    test loss: 0.1464, accuracy: 0.9533
    train loss: 0.1128, accuracy: 0.9500
    epoch: 298
    test loss: 0.1476, accuracy: 0.9533
    train loss: 0.1090, accuracy: 0.9633
    epoch: 299
    test loss: 0.1471, accuracy: 0.9467
    train loss: 0.1082, accuracy: 0.9667
    epoch: 300
    test loss: 0.1458, accuracy: 0.9600
    


```python
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(epochs, train_losses, label='train')
plt.plot(epochs, test_losses, label='test')
plt.legend()
plt.show

plt.figure(2)
plt.plot(epochs, train_accuracys, label='train')
plt.plot(epochs, test_accuracys, label='test')
plt.legend()
plt.show

```




    <function matplotlib.pyplot.show(close=None, block=None)>




    
![svg](/assets/images/img/output_50_1.png)
    



    
![svg](/assets/images/img/output_50_2.png)
    



