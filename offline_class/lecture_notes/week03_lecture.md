# Week 03 - KNN의 이웃 수 조절과 훈련·테스트 데이터 분리

## 배운 내용

- K-최근접 이웃(KNN) 알고리즘 복습과 이웃 수 `n_neighbors` 조절
- 학습 데이터로 다시 평가하는 방식의 한계
- 훈련 세트와 테스트 세트를 구분하는 이유
- 순서대로 데이터를 분리할 때 발생하는 문제
- NumPy 배열, 난수 시드, 인덱스 섞기(`shuffle`)
- 섞인 인덱스로 훈련·테스트 세트 생성, 시각화 및 재평가

> 이 노트는 3주차 오프라인 수업 녹취를 정리한 것이다. 음성 인식 오류가 많은 구간은 개념과 코드 흐름을 중심으로 정리했다. 원본 49마리의 길이·무게 수치는 녹취에 전부 포함되지 않아 2주차의 `fish_data`, `fish_target`이 이미 생성되어 있다고 가정한다.

## 1. 지난 시간 복습: KNN 분류기

KNN은 새로운 생선과 **가까운 K개의 이웃**을 찾고, 이웃들의 클래스를 비교해 다수결로 분류한다. 기본 `KNeighborsClassifier()`의 `n_neighbors`는 5이다.

```python
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)

print(kn.score(fish_data, fish_target))
print(kn.predict([[10, 150]]))
print(kn.predict_proba([[10, 150]]))
```

- `fit(X, y)`: 입력 데이터와 정답을 이용해 학습
- `score(X, y)`: 분류 정확도 확인 (`1.0`은 100%)
- `predict(X)`: 예측 클래스 반환
- `predict_proba(X)`: 클래스별 확률 추정값 반환

강의에서는 `[10, 150]`에 대해 빙어(`0`)로 예측하고, 이웃 5개 중 빙어 4개·도미 1개에 해당하는 `0.8`, `0.2` 예시를 설명했다.

## 2. K값(`n_neighbors`) 변경

이웃 수는 조절할 수 있는 **하이퍼파라미터**이다. K가 지나치게 크면 가까운 데이터의 특성보다 전체 데이터의 다수 클래스에 영향을 크게 받을 수 있다.

```python
kn49 = KNeighborsClassifier(n_neighbors=49)
kn49.fit(fish_data, fish_target)
print(kn49.score(fish_data, fish_target))
```

전체 49마리 중 도미는 35마리, 빙어는 14마리이다. `K=49`이면 모든 이웃에서 도미가 다수이므로 항상 도미(`1`)로 예측한다. 이 경우 전체 학습 데이터 정확도는 `35 / 49 ≈ 0.7143`이다. 강의에서는 이와 같이 K가 너무 크면 분류 결과가 나빠질 수 있음을 설명했다.

K에 따른 정확도를 직접 확인할 수 있다.

```python
k_values = range(3, 21)
scores = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(fish_data, fish_target)
    scores.append(model.score(fish_data, fish_target))

import matplotlib.pyplot as plt

plt.plot(list(k_values), scores, marker='o')
plt.xlabel('n_neighbors (K)')
plt.ylabel('Training accuracy')
plt.show()
```

강의에서는 이웃 수가 증가하다가 약 18개 부근에서 기존의 100% 정확도가 떨어지는 예를 보여주었다. **정확한 변화는 사용한 데이터와 K값에 따라 확인해야 한다.** 학습 정확도만 보고 최적의 K를 결정해서는 안 된다.

## 3. 훈련 데이터와 테스트 데이터 분리

모델을 만든 데이터로 다시 평가하면 진짜 예측 성능을 알기 어렵다. 따라서 전체 데이터를 다음 두 부분으로 나눈다.

- **훈련 세트(train)**: 모델을 학습시키는 데이터
- **테스트 세트(test)**: 학습에 사용하지 않고 모델을 평가하는 데이터

### 잘못된 방법: 데이터 순서대로 분할

```python
train_input = fish_data[:35]
train_target = fish_target[:35]

test_input = fish_data[35:]
test_target = fish_target[35:]

print(len(train_input), len(train_target))  # 35 35
print(len(test_input), len(test_target))    # 14 14

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
print(kn.score(train_input, train_target))
print(kn.score(test_input, test_target))
```

원본 데이터는 **도미 35마리 다음에 빙어 14마리** 순서로 저장되어 있다. 이렇게 자르면 훈련에는 도미만, 테스트에는 빙어만 들어간다. 강의의 테스트 정확도는 `0.0`이었다.

**핵심:** 단순히 나누는 것만으로는 부족하다. 훈련·테스트 데이터에 두 종류가 적절하게 포함되도록 분리해야 한다.

## 4. NumPy로 인덱스 섞기

강의에서는 NumPy를 사용해 순서를 섞은 **인덱스**로 데이터를 선택했다.

```python
import numpy as np

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

np.random.seed(42)  # 예시 시드: 동일한 섞기 결과를 재현
index = np.arange(49)
np.random.shuffle(index)

print(index)
```

- `np.array()`: 리스트를 NumPy 배열로 변환
- `np.arange(49)`: 0부터 48까지의 인덱스 생성
- `np.random.shuffle(index)`: 인덱스 배열의 순서를 섞음
- `np.random.seed(42)`: 동일 환경에서 난수 결과를 재현하기 위한 설정. **42는 정리용 예시이며 녹취에서 실제 사용한 시드값은 명확하지 않다.**

**주의:** 같은 인덱스로 입력 데이터와 정답을 가져와야 데이터와 정답의 짝이 유지된다.

## 5. 섞인 인덱스로 데이터 나누기

```python
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

print(train_input.shape)   # (35, 2)
print(train_target.shape)  # (35,)
print(test_input.shape)    # (14, 2)
print(test_target.shape)   # (14,)
print(train_target)
print(test_target)
```

`index[:35]`는 섞인 인덱스 가운데 35개, `index[35:]`는 나머지 14개이다. 두 세트에 도미(`1`)와 빙어(`0`)가 어떻게 분포하는지 타깃 배열로 확인한다.

## 6. 훈련·테스트 데이터 산점도

```python
import matplotlib.pyplot as plt

plt.scatter(train_input[:, 0], train_input[:, 1], label='Train')
plt.scatter(test_input[:, 0], test_input[:, 1], label='Test')
plt.xlabel('Length')
plt.ylabel('Weight')
plt.legend()
plt.show()
```

- `[:, 0]`: 모든 행의 첫 번째 열(생선 길이)
- `[:, 1]`: 모든 행의 두 번째 열(생선 무게)
- 두 세트의 데이터가 어떻게 분포하는지 산점도로 비교

## 7. 모델 다시 학습하고 평가하기

```python
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)

print('Train score:', kn.score(train_input, train_target))
print('Test score:', kn.score(test_input, test_target))
```

강의에서는 올바르게 데이터를 섞은 후 테스트 정확도가 `1.0`으로 나온 결과를 확인했다. 실행 결과는 **원본 데이터, 난수 시드, 라이브러리 설정**에 따라 다를 수 있다. 테스트 데이터로 학습하지 않는 것이 중요하다.

## 핵심 함수·문법 정리

| 함수·문법 | 역할 |
| --- | --- |
| `KNeighborsClassifier(n_neighbors=k)` | 분류에 참고할 이웃 수 설정 |
| `fit(X, y)` | 모델 학습 |
| `score(X, y)` | 정확도 평가 |
| `np.array(data)` | NumPy 배열로 변환 |
| `np.arange(n)` | `0`부터 `n-1`까지 인덱스 생성 |
| `np.random.seed(n)` | 난수 결과 재현을 위한 시드 지정 |
| `np.random.shuffle(index)` | 인덱스 순서 무작위 섞기 |
| `index[:35]`, `index[35:]` | 훈련·테스트에 사용할 인덱스 분리 |
| `array[index]` | 인덱스를 이용한 데이터 선택 |
| `array[:, 0]`, `array[:, 1]` | 모든 행의 특정 열 선택 |
| `plt.scatter(x, y)` | 산점도 그리기 |

## 실습 이미지 삽입

이미지 파일을 `offline_class/images/`에 저장한 뒤 아래의 파일명을 실제 사진 이름으로 바꾼다.

```markdown
![3주차 K값과 정확도 그래프](../images/week03_knn_score.png)

![3주차 훈련·테스트 데이터 산점도](../images/week03_train_test.png)
```

## 이번 주 핵심 요약

1. KNN의 이웃 수 K는 조절할 수 있으며, 무조건 크게 한다고 정확해지지는 않는다.
2. 모델을 평가할 때는 학습에 사용하지 않은 테스트 데이터를 써야 한다.
3. 도미 35개·빙어 14개를 순서대로 분할하면 한 클래스만 학습하게 되는 문제가 발생한다.
4. NumPy에서 인덱스를 섞고, **동일한 인덱스**를 입력 데이터와 타깃에 적용하여 나눈다.
5. 섞은 뒤 산점도와 타깃 분포를 확인하고 모델을 다시 평가한다.
