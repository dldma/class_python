# Week 05 - KNN 표준화 복습과 최근접 이웃 회귀

## 배운 내용

- 표준화한 훈련·테스트·예측 데이터의 스케일 일치
- `kneighbors()`로 표준화 후 최근접 이웃 확인 및 산점도 표시
- 분류(classification)와 회귀(regression)의 차이
- 농어 길이로 무게를 예측하는 KNN 회귀
- `train_test_split()`, `reshape(-1, 1)` 사용
- `KNeighborsRegressor`, `fit()`, `score()`, 결정계수 $R^2$
- 훈련 점수와 테스트 점수 비교, 과소적합 도입

> **자료 범위:** 5주차 오프라인 강의 녹취록을 요약했다. 앞부분은 4주차의 표준화 실습을 마무리하며, 이후에는 농어 무게를 예측하는 회귀로 넘어간다. 명백한 자동 전사 오류(예: `kneighbors`, `KNeighborsRegressor`)는 정확한 함수 이름으로 정리했다. **농어 길이·무게의 원본 배열은 녹취록에 전부 수록되어 있지 않으므로 임의로 작성하지 않았다.** 아래 회귀 코드는 수업 자료에 `perch_length`, `perch_weight`를 입력해 둔 뒤 실행하는 형태다.

## 1. 표준화 후 최근접 이웃 확인 (4주차 이어서)

KNN은 데이터 간 **거리**로 이웃을 찾는다. 길이와 무게의 값 범위가 크게 다르면 무게처럼 값이 큰 특성이 거리 계산에 더 큰 영향을 줄 수 있다. 이를 해결하려고 훈련 데이터의 평균과 표준편차를 사용해 표준화한다.

$$z = \frac{x - \mu}{\sigma}$$

- $\mu$: **훈련 데이터**에서 계산한 특성별 평균
- $\sigma$: **훈련 데이터**에서 계산한 특성별 표준편차
- 훈련·테스트·새 샘플 모두 **훈련 데이터로 구한 동일한** 평균·표준편차를 사용해야 한다.

```python
import numpy as np

# 앞 주차에서 train_input, test_input, kn이 생성되었다고 가정
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

train_scaled = (train_input - mean) / std
test_scaled = (test_input - mean) / std

# 새 샘플: 길이 25, 무게 150 (2차원 입력)
one = np.array([[25, 150]])
one_scaled = (one - mean) / std

# 표준화된 데이터로 다시 모델 학습
kn.fit(train_scaled, train_target)
print(kn.predict(one_scaled))
```

`one`도 표준화해야 훈련 데이터와 같은 좌표계에서 거리 계산을 할 수 있다. 실제 모델 판단은 **표준화한 데이터로 재학습한 모델의 결과**를 확인해야 한다.

### 최근접 이웃 찾기와 그림 그리기

```python
import matplotlib.pyplot as plt

# 거리와 훈련 데이터 내 인덱스
# 주의: kneighbors의 반환 순서는 (거리, 인덱스)
distances, indexes = kn.kneighbors(one_scaled)
print(distances)
print(indexes)

plt.scatter(train_scaled[:, 0], train_scaled[:, 1], label='Train')
plt.scatter(test_scaled[:, 0], test_scaled[:, 1], label='Test')
plt.scatter(one_scaled[:, 0], one_scaled[:, 1], marker='^', label='Sample')
plt.scatter(train_scaled[indexes[0], 0], train_scaled[indexes[0], 1],
            marker='D', label='Neighbors')
plt.xlabel('Standardized length')
plt.ylabel('Standardized weight')
plt.legend()
plt.show()
```

- `kneighbors(one_scaled)`: 지정한 샘플의 가까운 이웃에 대한 거리와 인덱스 반환
- `indexes[0]`: 샘플 하나의 최근접 이웃 인덱스들
- `train_scaled[indexes[0], 0]`: 해당 이웃의 길이 좌표
- `train_scaled[indexes[0], 1]`: 해당 이웃의 무게 좌표

강의에서는 표준화 후 새 샘플이 도미 쪽 이웃에 더 가깝게 분포한 모습을 살펴보았다. 그래프에서도 **원본 데이터가 아닌 표준화된 배열**을 사용해야 한다.

## 2. 분류와 회귀의 차이

| 구분 | 분류(Classification) | 회귀(Regression) |
|---|---|---|
| 목표 | 어느 종류인지 판단 | 연속적인 숫자 예측 |
| 예시 | 도미(1) / 빙어(0) | 농어의 길이로 무게 예측 |
| KNN 예측 방식 | 가까운 이웃의 **다수결** | 가까운 이웃의 **값 평균**(기본 설정) |
| scikit-learn 클래스 | `KNeighborsClassifier` | `KNeighborsRegressor` |

예를 들어 가까운 이웃 3개의 무게가 60g, 80g, 100g이면 단순 평균을 이용한 회귀 예측값은 **80g**이다. 이 수치는 평균 방식을 설명하기 위한 예시이며, 강의에서 사용한 농어 원본 데이터는 아니다.

## 3. 농어 데이터: 입력과 타깃 구분

이번 실습의 목적은 **농어의 길이를 입력하면 농어의 무게를 예측**하는 것이다.

- 입력 특성 `perch_length`: 농어 길이
- 정답(타깃) `perch_weight`: 농어 무게
- 생선 종류를 맞히는 문제가 아니므로 도미/빙어 같은 0·1 타깃이나 `stratify`를 사용하지 않는다.

```python
import numpy as np
import matplotlib.pyplot as plt

# 수업에 제시된 농어 길이·무게 원본 데이터를 먼저 입력해야 함
# perch_length = [...]
# perch_weight = [...]

perch_length = np.array(perch_length)
perch_weight = np.array(perch_weight)

print(perch_length.shape)
print(perch_weight.shape)

plt.scatter(perch_length, perch_weight)
plt.xlabel('Length')
plt.ylabel('Weight')
plt.show()
```

강의에서는 산점도를 통해 **길이가 길수록 무게가 증가하는 전반적인 경향**을 관찰했다. 실행하려면 먼저 실제 수업 데이터 배열을 정의해야 한다.

## 4. 훈련·테스트 분할 및 `reshape()`

```python
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42
)

# 1차원 길이 배열을 (샘플 수, 특성 수=1) 형태로 변경
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

print(train_input.shape, test_input.shape)
print(train_target.shape, test_target.shape)
```

- `train_test_split()`: 훈련·테스트 데이터를 함께 나눔
- `random_state=42`: 실행할 때마다 같은 분할을 재현하기 위한 난수 시드
- `reshape(-1, 1)`: **열 하나의 2차원 배열**로 변형. `-1`은 행 수를 자동 계산한다는 의미
- 타깃(`train_target`, `test_target`)은 1차원 배열로 두어도 된다.

> KNN에 전달하는 입력 `X`의 모양은 보통 `(샘플 개수, 특성 개수)`이다. 농어 길이는 특성이 하나이므로 두 번째 차원이 `1`이다.

## 5. KNN 회귀 모델 학습과 평가

```python
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()  # 기본 최근접 이웃 수: 5
knr.fit(train_input, train_target)

print('훈련 R²:', knr.score(train_input, train_target))
print('테스트 R²:', knr.score(test_input, test_target))
```

- `KNeighborsRegressor()`: KNN 회귀 모델 생성
- `fit(X, y)`: 길이(X)와 무게(y)로 모델 학습
- `predict(X)`: 길이에 대응하는 무게 예측
- `score(X, y)`: 회귀 모델에서는 기본적으로 **결정계수 $R^2$** 반환

강의에서는 훈련 점수가 약 **0.96**, 테스트 점수가 약 **0.99**로 나온 경우를 설명했다. 정확한 값은 실제 원본 배열과 실행 결과로 확인해야 한다.

## 6. 결정계수 $R^2$ 이해

$$R^2 = 1 - \frac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y})^2}$$

- $y_i$: 실제 무게(정답)
- $\hat{y}_i$: 모델이 예측한 무게
- $\bar{y}$: 평가에 사용하는 정답 무게의 평균
- 실제값과 예측값의 차이가 작을수록 $R^2$가 커진다.

| 값 | 의미 |
|---|---|
| `1.0` | 평가한 데이터의 실제값을 완벽하게 예측 |
| `0.0` | 정답의 평균값만 예측하는 기준 모델과 같은 수준 |
| 음수 | 평균값만 예측하는 기준 모델보다 성능이 낮음 |

**주의:** 분류 모델의 `score()`가 반환하는 정확도와 회귀 모델의 `score()`가 반환하는 $R^2$는 **서로 다른 지표**다. 예컨대 회귀 `score = 0.99`는 '99%의 농어를 정답 처리했다'는 의미가 아니다.

## 7. 과소적합 도입

강의 말미에는 **훈련 점수(약 0.96)가 테스트 점수(약 0.99)보다 낮은 사례**를 제시하고, 과소적합(underfitting)을 다음 시간에 더 자세히 다루겠다고 했다.

- **과소적합:** 훈련 데이터의 패턴을 충분히 학습하지 못하는 상황
- 일반적으로는 훈련·테스트 성능이 모두 낮게 나타나는 경우가 대표적이다.
- **테스트 점수가 훈련 점수보다 높다는 사실만으로 과소적합이 확정되는 것은 아니다.** 데이터 분할과 점수 규모도 함께 봐야 한다.

## 핵심 함수 정리

| 코드 | 역할 |
|---|---|
| `np.mean(X, axis=0)` | 특성별 평균 계산 |
| `np.std(X, axis=0)` | 특성별 표준편차 계산 |
| `kn.kneighbors(X)` | 가까운 이웃의 거리와 인덱스 조회 |
| `plt.scatter(x, y)` | 산점도 그리기 |
| `train_test_split(X, y, random_state=42)` | 훈련·테스트 데이터 분리 |
| `X.reshape(-1, 1)` | 특성이 하나인 2차원 입력으로 변경 |
| `KNeighborsRegressor()` | 최근접 이웃 회귀 모델 생성 |
| `knr.fit(X, y)` | 회귀 모델 학습 |
| `knr.predict(X)` | 연속적인 수치 예측 |
| `knr.score(X, y)` | 회귀 모델의 $R^2$ 계산 |

## 사진 삽입 위치

실제 사진을 `offline_class/images` 폴더에 넣은 뒤 파일명에 맞춰 아래 경로를 사용한다.

```markdown
![표준화 전후의 최근접 이웃](../images/week05_scaled_neighbors.png)

![농어 길이와 무게 산점도](../images/week05_perch_scatter.png)

![KNN 회귀 훈련 및 테스트 점수](../images/week05_regression_score.png)
```

## 이번 주 핵심 요약

1. KNN 거리 계산에서는 **훈련·테스트·새 샘플의 동일한 스케일**이 중요하다.
2. KNN **분류**는 주변 이웃의 다수결, KNN **회귀**는 주변 이웃 타깃 값의 평균으로 예측한다(기본 설정 기준).
3. 농어 회귀에서는 **길이가 입력, 무게가 정답**이다.
4. `reshape(-1, 1)`로 입력 길이 데이터를 2차원으로 만든다.
5. 회귀의 `score()`는 정확도가 아니라 **결정계수 $R^2$**다.
6. 훈련 성능과 테스트 성능을 함께 비교해 모델의 학습 상태를 판단한다.
