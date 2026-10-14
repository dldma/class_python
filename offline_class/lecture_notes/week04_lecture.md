# Week 04 - 데이터 분할과 KNN의 특성 스케일

## 배운 내용

- NumPy의 `column_stack()`, `ones()`, `zeros()`, `concatenate()`로 생선 데이터 준비
- `train_test_split()`으로 훈련 세트와 테스트 세트 자동 분할
- `stratify`, `random_state`, `test_size`의 의미
- KNN 모델 학습·평가와 새로운 샘플 예측
- `kneighbors()`로 최근접 이웃 확인하기
- 길이와 무게의 스케일 차이가 거리 계산에 미치는 영향
- 평균과 표준편차를 이용한 표준화(Z점수) 도입

> 이 노트는 4주차 오프라인 수업 녹취록을 정리한 것이다. 음성 인식 오류는 명백한 함수명과 용어에 한해 바로잡았다. 원본 생선 길이·무게 49개 값이 녹취록에는 없으므로, 이전 실습의 `fish_length`, `fish_weight`가 정의되어 있다고 가정한다. 녹취록은 표준편차 설명 도중 끝나므로 아래 표준화 코드는 강의 설명을 기반으로 한 **추가 실습 예시**다.

## 1. NumPy를 이용한 입력 데이터와 정답 만들기

이전에는 `zip()`과 리스트 내포로 길이·무게를 쌍으로 만들었다. 이번에는 NumPy 함수를 사용한다.

```python
import numpy as np

# fish_length, fish_weight: 이전 실습에서 준비한 각 49개 값
fish_data = np.column_stack((fish_length, fish_weight))
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

print(len(fish_data), len(fish_target))  # 49 49
print(fish_data[:5])
print(fish_target)
```

- `np.column_stack((a, b))`: `a`와 `b`를 열 단위로 붙여 `[길이, 무게]` 형태의 2차원 배열 생성
- `np.ones(35)`: `1`을 35개 가진 배열 생성(도미)
- `np.zeros(14)`: `0`을 14개 가진 배열 생성(빙어)
- `np.concatenate((a, b))`: 배열 두 개 연결

따라서 `fish_data.shape`는 `(49, 2)`이고, `fish_target.shape`는 `(49,)`이다.

## 2. 훈련·테스트 데이터 자동 분할

이전 시간에는 인덱스를 직접 섞었지만, 이번 시간에는 `train_test_split()`을 사용한다.

```python
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, stratify=fish_target, random_state=42
)

print(len(train_input), len(train_target))  # 36 36
print(len(test_input), len(test_target))    # 13 13
print(train_target)
print(test_target)
```

- `stratify=fish_target`: 전체 데이터의 도미·빙어 비율(35:14)을 훈련·테스트 세트에도 **가능한 비슷하게** 유지
- `random_state=42`: 분할 결과를 재현할 수 있도록 난수 상태를 고정
- `test_size`: 테스트 세트의 비율 지정; 기본값은 `0.25`(별도 지정하지 않았을 때)

```python
# 테스트 비율을 직접 지정하는 예시
train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target,
    test_size=0.2,
    stratify=fish_target,
    random_state=42
)
```

> `test_size` 지정 예시는 앞의 기본 분할과 별개로 실행한다. 이후 코드에서 강의의 기본 분할(36개/13개)을 그대로 활용하려면 첫 번째 분할 코드를 다시 실행하자.

## 3. KNN 모델 학습과 성능 평가

```python
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)

print(kn.score(train_input, train_target))
print(kn.score(test_input, test_target))
```

- `fit()`: **훈련 세트로만** 학습
- `score()`: 정답과 예측을 비교해 정확도 계산
- 강의에서는 훈련·테스트 정확도가 모두 `1.0`인 결과를 확인했다. 실제 결과는 입력한 원본 데이터와 분할 설정에 따라 달라질 수 있다.

## 4. 수상한 도미: 길이 25, 무게 150

강의에서는 겉보기에는 도미에 가까워 보이는 샘플 `[25, 150]`이 스케일 조정 전 KNN에서 빙어(`0`)로 분류되는 사례를 살펴봤다.

```python
new_fish = np.array([[25, 150]])
print(kn.predict(new_fish))
print(kn.predict_proba(new_fish))
```

강의에서 제시한 분류 결과는 `0`이며, 확률 추정값은 `[0.8, 0.2]`이다. 이 경우 가까운 다섯 이웃 가운데 빙어가 네 개, 도미가 하나라는 뜻이다. `predict_proba()`의 열 순서는 모델의 `kn.classes_`로 확인한다.

## 5. 산점도와 가장 가까운 5개 이웃 표시

```python
import matplotlib.pyplot as plt

# 가까운 이웃의 거리와 훈련 데이터 안에서의 인덱스
distances, indexes = kn.kneighbors(new_fish)
print(distances)
print(indexes)

plt.scatter(train_input[:, 0], train_input[:, 1], label='train')
plt.scatter(test_input[:, 0], test_input[:, 1], label='test')
plt.scatter(new_fish[:, 0], new_fish[:, 1], marker='x', label='new fish')
plt.scatter(train_input[indexes[0], 0], train_input[indexes[0], 1],
            marker='^', label='5 neighbors')
plt.xlabel('length')
plt.ylabel('weight')
plt.legend()
plt.show()
```

`kneighbors()`의 반환값은 거리 배열과 이웃 인덱스 배열이다. `indexes[0]`을 이용하면 샘플 하나에 대응하는 다섯 이웃을 선택할 수 있다.

## 6. 길이·무게 특성의 스케일 차이

KNN은 데이터 간 **거리**로 이웃을 선택한다. 그런데 길이는 대략 수십 단위, 무게는 수백~천 단위이므로 실제 거리 계산에서는 **무게 차이가 훨씬 크게 작용**할 수 있다.

강의에서는 그래프의 X축 범위를 `0~1000`으로 넓혀 이 차이를 직관적으로 보여줬다.

```python
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(new_fish[:, 0], new_fish[:, 1], marker='x')
plt.xlim(0, 1000)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

단, `xlim()`은 **그래프 표시 범위만** 바꾸며 학습용 데이터의 값 자체를 바꾸지는 않는다.

## 7. 데이터 표준화(Z점수) 개념

길이와 무게를 비슷한 기준으로 비교하려면 각 특성을 평균과 표준편차로 변환할 수 있다.

\[
Z = \frac{x - \mu}{\sigma}
\]

- `x`: 원래 측정값
- `μ`: 해당 특성의 훈련 데이터 평균
- `σ`: 해당 특성의 훈련 데이터 표준편차

**핵심:** 평균과 표준편차는 **훈련 데이터에서만** 계산하고, 같은 값을 테스트 데이터와 새 샘플에도 적용해야 한다. 테스트 데이터에서 따로 평균을 계산하면 평가에 필요한 정보를 미리 사용하는 문제가 생길 수 있다.

### 추가 실습 예시: 표준화 적용

녹취록은 평균 및 표준편차를 설명하다 종료되므로, 아래는 강의에서 완전히 확인된 실습 결과가 아니라 개념을 이어 실습할 수 있도록 추가한 코드다.

```python
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

train_scaled = (train_input - mean) / std
test_scaled = (test_input - mean) / std
new_fish_scaled = (new_fish - mean) / std

kn_scaled = KNeighborsClassifier()
kn_scaled.fit(train_scaled, train_target)

print(kn_scaled.score(train_scaled, train_target))
print(kn_scaled.score(test_scaled, test_target))
print(kn_scaled.predict(new_fish_scaled))
```

`axis=0`은 각 열(길이·무게)의 평균과 표준편차를 각각 계산한다. 표준화된 분류 결과는 정확한 원본 데이터와 분할 설정을 실행해 확인해야 한다.

## 주요 함수 정리

| 함수 또는 옵션 | 역할 |
| --- | --- |
| `np.column_stack()` | 배열을 열 방향으로 결합 |
| `np.ones()` / `np.zeros()` | 1 또는 0으로 채운 배열 생성 |
| `np.concatenate()` | 여러 배열을 이어 붙임 |
| `train_test_split()` | 데이터와 타깃을 훈련·테스트로 분할 |
| `stratify` | 클래스 비율을 가깝게 유지 |
| `random_state` | 데이터 분할 재현성 확보 |
| `test_size` | 테스트 세트 비율 설정 |
| `fit()` / `score()` | 학습 / 정확도 평가 |
| `predict()` / `predict_proba()` | 클래스 예측 / 확률 추정 |
| `kneighbors()` | 최근접 이웃의 거리와 인덱스 반환 |
| `plt.scatter()` / `plt.xlim()` | 산점도 / X축 범위 지정 |
| `np.mean()` / `np.std()` | 평균 / 표준편차 계산 |

## 핵심 정리

1. 기존의 직접 구현 작업을 NumPy와 scikit-learn 함수로 간단히 처리할 수 있다.
2. 훈련 데이터와 테스트 데이터는 분리하고, 클래스 비율도 고려해야 한다.
3. KNN은 거리 기반 모델이므로 특성의 수치 범위 차이가 분류 결과에 큰 영향을 준다.
4. 그래프 축 범위를 바꾸는 것과 **실제 데이터를 표준화하는 것**은 다르다.
5. 표준화 기준은 훈련 세트에서 구하고 테스트 세트와 새 샘플에 똑같이 적용한다.

## 이미지 추가 위치

사진을 `offline_class/images`에 저장했다면 아래처럼 연결하면 된다.

```markdown
![4주차 훈련·테스트 산점도](../images/week04_train_test.png)

![수상한 도미와 최근접 이웃](../images/week04_neighbors.png)

![길이·무게 스케일 차이](../images/week04_scale.png)
```
