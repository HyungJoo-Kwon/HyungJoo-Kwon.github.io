---
layout: page
title: "밑바닥부터 시작하는 딥러닝 23단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 23단계"
categories: deeplearnig
comments: true
published: true
---
# 패키지로 정리 

```python
# Add import path for the dezero directory.
if '__file__' in globals():         # 문장에서 __file__이라는 전역 변수가 정의되어 있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable


x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)

# python step23.py 처럼 터미널에서 python 명령으로 실행한다면  __file__ 변수가 정의되어 있음. 이 경우 현재 파일이 위치한 디렉터리의 부모 디렉터리(..)를 모듈 검색 경로에 추가.
```

    variable(16.0)
    variable(8.0)
    


```python
sys.path
```




    ['c:\\Users\\82103\\Desktop\\study\\deep_learning\\밑바닥부터 시작하는 딥러닝3',
     'c:\\Users\\82103\\.vscode\\extensions\\ms-toolsai.jupyter-2021.3.619093157\\pythonFiles',
     'c:\\Users\\82103\\.vscode\\extensions\\ms-toolsai.jupyter-2021.3.619093157\\pythonFiles\\lib\\python',
     'C:\\Users\\82103\\anaconda3\\python38.zip',
     'C:\\Users\\82103\\anaconda3\\DLLs',
     'C:\\Users\\82103\\anaconda3\\lib',
     'C:\\Users\\82103\\anaconda3',
     '',
     'C:\\Users\\82103\\AppData\\Roaming\\Python\\Python38\\site-packages',
     'C:\\Users\\82103\\anaconda3\\lib\\site-packages',
     'C:\\Users\\82103\\anaconda3\\lib\\site-packages\\win32',
     'C:\\Users\\82103\\anaconda3\\lib\\site-packages\\win32\\lib',
     'C:\\Users\\82103\\anaconda3\\lib\\site-packages\\Pythonwin',
     'C:\\Users\\82103\\anaconda3\\lib\\site-packages\\IPython\\extensions',
     'C:\\Users\\82103\\.ipython']



- 모듈 
    - 모듈은 파이썬 파일. import하여 사용하는 것을 가정하고 만들어진 파이썬 파일을 모듈.
- 패키지
    - 패키지는 여러 모듈을 묶은 것. 패키지를 만들려면 먼저 디렉터리를 만들고 그 안에 모듈을 추가.
- 라이브러리
    - 라이브러리는 여러 패키지를 묶은 것. 하나 이상의 디렉터리로 구성.    

