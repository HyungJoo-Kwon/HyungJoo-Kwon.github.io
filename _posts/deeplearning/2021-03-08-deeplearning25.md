---
layout: page
title: "밑바닥부터 시작하는 딥러닝 25단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 25단계"
categories: deeplearnig
comments: true
published: true
---
# Graphviz를 이용한 계산 그래프 시각화      

```python


# https://graphbiz.gitlab.io/download/
# conda install graphviz
# conda install python-graphviz

# dot sample.dot -T png -o sample.png  :  sample.dot라는 파일을 sample.png로 변환
```


```python
digraph g{
    x
    y
}
```

![그림 25-1](/assets/images/img/그림 25-1.png)


```python
digraph g{
    [label='x', color=orange, style=filled]
    [label='y', color=orange, style=filled]
}
```

![25-1](/assets/images/img/그림 25-2.png)


```python
digraph g{
    [label='x', color=orange, style=filled]
    [label='y', color=orange, style=filled]
    [label='Exp', color=lightblue, style=filled, shape=box]
}
```

![그림 25-3](/assets/images/img/그림 25-3.png)


```python
digraph g{
1 [label='x', color=orange, style=filled]
2 [label='y', color=orange, style=filled]
3 [label='Exp', color=lightblue, style=filled, shape=box]
1 -> 3
3 -> 2
}
```

![그림 25-4](/assets/images/img/그림 25-4.png)


