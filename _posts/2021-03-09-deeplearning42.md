---
layout: page
title: "밑바닥부터 시작하는 딥러닝 42단계"
description: "학습"
headline: "밑바닥부터 시작하는 딥러닝 42단계"
categories: deeplearnig
comments: true
published: true
---
# 선형 회귀     

```python
# 회귀 : x로부터 실숫값 y를 예측하는 것
# 선형 회귀 : 회귀 모델 중 예측값이 선형(직선)을 이루는 것
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
```

평균 제곱 오차 : 총 N개의 점에 대해 (x, y)의 각 점에서 제곱오차를 구한 다음 더한 후 평균    
예측치와 데이터의 오차를 나타내는 지표(잔차)를 최소화 해야      
![식 42.1](/assets/images/img/그림 42.1.png)      

![그림 42-2](/assets/images/img/그림 42-2.png)    
y = Wx + b에서 손실 함수의 출력을 최소화하는 W와 b를 찾는 것


```python
import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F

# Generate toy dataset, 실험용 작은 데이터 셋
np.random.seed(0)   # 시드값 고정
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))


def predict(x):
    y = F.matmul(x, W) + b      # b가 브로드캐스트  됨
    return y

# y = F.matmul(x, W) : 선형 변환 (완전연결계층에 해당)
# y = F.matmul(x. W) + b : 아핀 변환
```

코드 형상   
![그림 42-3](/assets/images/img/그림 42-3.png)

x의 데이터 차원이 4일 경우  
![그림 42-4](/assets/images/img/그림 42-4.png)


```python
def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)


lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    # Update .data attribute (No need grads when updating params)
    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(f'W : {W}, b : {b}, loss : {loss}')
```

    W : variable([[0.64433458]]), b : variable([1.29473389]), loss : variable(42.296340129442335)
    W : variable([[1.12672345]]), b : variable([2.26959351]), loss : variable(23.97380754378544)
    W : variable([[1.48734571]]), b : variable([3.00386712]), loss : variable(13.609686745040522)
    W : variable([[1.75641886]]), b : variable([3.557186]), loss : variable(7.747049961219976)
    W : variable([[1.95666851]]), b : variable([3.97439789]), loss : variable(4.43057410592155)
    W : variable([[2.10518573]]), b : variable([4.28923203]), loss : variable(2.5542803813535926)
    W : variable([[2.21482401]]), b : variable([4.52705574]), loss : variable(1.4925998690471942)
    W : variable([[2.29524981]]), b : variable([4.70694745]), loss : variable(0.8916952181756932)
    W : variable([[2.35373273]]), b : variable([4.84325585]), loss : variable(0.5514270962227453)
    W : variable([[2.39573972]]), b : variable([4.9467725]), loss : variable(0.35859153083192785)
    W : variable([[2.425382]]), b : variable([5.02561369]), loss : variable(0.2491573197756112)
    W : variable([[2.44575118]]), b : variable([5.08588371]), loss : variable(0.18690658765397886)
    W : variable([[2.45917205]]), b : variable([5.13217364]), loss : variable(0.15135336296314875)
    W : variable([[2.4673927]]), b : variable([5.16793652]), loss : variable(0.13091003006317078)
    W : variable([[2.47172747]]), b : variable([5.19576949]), loss : variable(0.11902210735018462)
    W : variable([[2.47316455]]), b : variable([5.21762597]), loss : variable(0.11198198322254362)
    W : variable([[2.47244676]]), b : variable([5.23497527]), loss : variable(0.10769231158094322)
    W : variable([[2.47013247]]), b : variable([5.24892259]), loss : variable(0.10496655795675108)
    W : variable([[2.46664127]]), b : variable([5.26029927]), loss : variable(0.10313337115761934)
    W : variable([[2.46228843]]), b : variable([5.26973075]), loss : variable(0.10181280604960247)
    W : variable([[2.45731071]]), b : variable([5.27768752]), loss : variable(0.10078974954301653)
    W : variable([[2.4518859]]), b : variable([5.28452363]), loss : variable(0.09994232708821608)
    W : variable([[2.44614738]]), b : variable([5.29050548]), loss : variable(0.09920140749444824)
    W : variable([[2.44019517]]), b : variable([5.29583359]), loss : variable(0.09852769772358987)
    W : variable([[2.4341042]]), b : variable([5.30065891]), loss : variable(0.09789878700703991)
    W : variable([[2.4279305]]), b : variable([5.30509512]), loss : variable(0.09730181854646197)
    W : variable([[2.42171596]]), b : variable([5.30922787]), loss : variable(0.0967293443169877)
    W : variable([[2.41549177]]), b : variable([5.3131217]), loss : variable(0.09617698031441604)
    W : variable([[2.40928112]]), b : variable([5.31682532]), loss : variable(0.09564208018092028)
    W : variable([[2.40310116]]), b : variable([5.32037549]), loss : variable(0.09512298485383605)
    W : variable([[2.39696452]]), b : variable([5.3238]), loss : variable(0.09461859803040694)
    W : variable([[2.39088043]]), b : variable([5.32711987]), loss : variable(0.09412814592514404)
    W : variable([[2.38485555]]), b : variable([5.33035108]), loss : variable(0.09365104127065577)
    W : variable([[2.37889464]]), b : variable([5.33350575]), loss : variable(0.09318680628411545)
    W : variable([[2.37300101]]), b : variable([5.33659314]), loss : variable(0.09273502898904006)
    W : variable([[2.3671769]]), b : variable([5.33962035]), loss : variable(0.09229533840647547)
    W : variable([[2.36142374]]), b : variable([5.34259283]), loss : variable(0.09186739042193233)
    W : variable([[2.35574235]]), b : variable([5.34551483]), loss : variable(0.09145085969346782)
    W : variable([[2.35013309]]), b : variable([5.34838966]), loss : variable(0.09104543497939387)
    W : variable([[2.34459602]]), b : variable([5.35121993]), loss : variable(0.09065081640275062)
    W : variable([[2.33913091]]), b : variable([5.35400772]), loss : variable(0.0902667138137311)
    W : variable([[2.33373736]]), b : variable([5.35675472]), loss : variable(0.08989284577554084)
    W : variable([[2.32841486]]), b : variable([5.35946234]), loss : variable(0.0895289389052284)
    W : variable([[2.32316275]]), b : variable([5.36213172]), loss : variable(0.08917472741757472)
    W : variable([[2.31798034]]), b : variable([5.36476385]), loss : variable(0.08882995278605695)
    W : variable([[2.31286689]]), b : variable([5.3673596]), loss : variable(0.08849436347219049)
    W : variable([[2.30782159]]), b : variable([5.36991973]), loss : variable(0.08816771469564999)
    W : variable([[2.30284363]]), b : variable([5.3724449]), loss : variable(0.08784976822950144)
    W : variable([[2.29793221]]), b : variable([5.37493575]), loss : variable(0.08754029221162851)
    W : variable([[2.29308646]]), b : variable([5.37739285]), loss : variable(0.08723906096725743)
    W : variable([[2.28830557]]), b : variable([5.37981674]), loss : variable(0.08694585483964615)
    W : variable([[2.2835887]]), b : variable([5.38220792]), loss : variable(0.0866604600272269)
    W : variable([[2.278935]]), b : variable([5.38456689]), loss : variable(0.08638266842618712)
    W : variable([[2.27434366]]), b : variable([5.38689411]), loss : variable(0.0861122774778659)
    W : variable([[2.26981384]]), b : variable([5.38919004]), loss : variable(0.08584909002056745)
    W : variable([[2.26534474]]), b : variable([5.39145512]), loss : variable(0.08559291414552149)
    W : variable([[2.26093555]]), b : variable([5.39368978]), loss : variable(0.08534356305679344)
    W : variable([[2.25658547]]), b : variable([5.39589443]), loss : variable(0.08510085493499035)
    W : variable([[2.25229371]]), b : variable([5.39806949]), loss : variable(0.08486461280463413)
    W : variable([[2.2480595]]), b : variable([5.40021536]), loss : variable(0.08463466440508784)
    W : variable([[2.24388206]]), b : variable([5.40233244]), loss : variable(0.08441084206493275)
    W : variable([[2.23976064]]), b : variable([5.40442111]), loss : variable(0.08419298257969864)
    W : variable([[2.23569448]]), b : variable([5.40648177]), loss : variable(0.08398092709285435)
    W : variable([[2.23168285]]), b : variable([5.40851479]), loss : variable(0.08377452097997208)
    W : variable([[2.22772501]]), b : variable([5.41052054]), loss : variable(0.08357361373597812)
    W : variable([[2.22382025]]), b : variable([5.41249938]), loss : variable(0.08337805886540826)
    W : variable([[2.21996785]]), b : variable([5.41445169]), loss : variable(0.08318771377558704)
    W : variable([[2.21616712]]), b : variable([5.41637782]), loss : variable(0.08300243967265336)
    W : variable([[2.21241735]]), b : variable([5.41827811]), loss : variable(0.08282210146035615)
    W : variable([[2.20871786]]), b : variable([5.42015291]), loss : variable(0.08264656764154595)
    W : variable([[2.20506799]]), b : variable([5.42200258]), loss : variable(0.08247571022229121)
    W : variable([[2.20146706]]), b : variable([5.42382744]), loss : variable(0.08230940461854906)
    W : variable([[2.19791443]]), b : variable([5.42562782]), loss : variable(0.08214752956532231)
    W : variable([[2.19440943]]), b : variable([5.42740407]), loss : variable(0.08198996702823673)
    W : variable([[2.19095144]]), b : variable([5.42915649]), loss : variable(0.08183660211747391)
    W : variable([[2.18753983]]), b : variable([5.43088541]), loss : variable(0.08168732300399718)
    W : variable([[2.18417396]]), b : variable([5.43259114]), loss : variable(0.08154202083800904)
    W : variable([[2.18085323]]), b : variable([5.434274]), loss : variable(0.08140058966958151)
    W : variable([[2.17757703]]), b : variable([5.4359343]), loss : variable(0.08126292637140048)
    W : variable([[2.17434477]]), b : variable([5.43757232]), loss : variable(0.0811289305635685)
    W : variable([[2.17115586]]), b : variable([5.43918838]), loss : variable(0.08099850454041051)
    W : variable([[2.16800971]]), b : variable([5.44078277]), loss : variable(0.0808715531992303)
    W : variable([[2.16490575]]), b : variable([5.44235578]), loss : variable(0.08074798397096412)
    W : variable([[2.16184341]]), b : variable([5.44390769]), loss : variable(0.08062770675268231)
    W : variable([[2.15882214]]), b : variable([5.44543879]), loss : variable(0.08051063384188899)
    W : variable([[2.15584139]]), b : variable([5.44694936]), loss : variable(0.08039667987257197)
    W : variable([[2.15290062]]), b : variable([5.44843967]), loss : variable(0.08028576175295649)
    W : variable([[2.14999928]]), b : variable([5.44990999]), loss : variable(0.08017779860491725)
    W : variable([[2.14713684]]), b : variable([5.4513606]), loss : variable(0.08007271170500452)
    W : variable([[2.1443128]]), b : variable([5.45279175]), loss : variable(0.07997042442704135)
    W : variable([[2.14152662]]), b : variable([5.45420371]), loss : variable(0.07987086218625004)
    W : variable([[2.13877781]]), b : variable([5.45559674]), loss : variable(0.07977395238486742)
    W : variable([[2.13606587]]), b : variable([5.45697108]), loss : variable(0.07967962435920853)
    W : variable([[2.13339029]]), b : variable([5.458327]), loss : variable(0.07958780932814088)
    W : variable([[2.13075059]]), b : variable([5.45966473]), loss : variable(0.07949844034293135)
    W : variable([[2.12814629]]), b : variable([5.46098452]), loss : variable(0.07941145223842926)
    W : variable([[2.12557692]]), b : variable([5.46228661]), loss : variable(0.07932678158554966)
    W : variable([[2.123042]]), b : variable([5.46357124]), loss : variable(0.07924436664502324)
    W : variable([[2.12054108]]), b : variable([5.46483864]), loss : variable(0.07916414732237737)
    W : variable([[2.11807369]]), b : variable([5.46608905]), loss : variable(0.07908606512411756)
    


```python
# Plot
plt.scatter(x.data, y.data, s=10)
plt.xlabel('x')
plt.ylabel('y')
y_pred = predict(x)
plt.plot(x.data, y_pred.data, color='r')
plt.show()
```


    
![result](/assets/images/img/output_43_0.png)
    



```python
# 기존의 mean_squared_error()는 그림 42-6처럼 이름 없는 변수 3개가 있고 이 변수들은 계산 그래프가 존재하는 동안 메모리에 계속 남아있다.
# 아래처럼 수정 후 forward 메서드의 범위를 벗어나는 순간 메모리에서 삭제.
class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

```

![그림 42-6](/assets/images/img/그림 42-6.png)
![그림 42-7](/assets/images/img/그림 42-7.png)
