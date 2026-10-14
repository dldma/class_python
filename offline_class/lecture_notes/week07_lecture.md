# 고급파이썬프로그래밍 — 7주차 오프라인 강의

> **학습 주제:** K-최근접 이웃 회귀의 한계, 선형 회귀, 2차 다항 회귀 및 중간고사 정리  
> **자료:** 7주차 수업 녹취록을 바탕으로 정리. 자동 음성 인식 오류가 있는 용어는 표준 Python 명칭으로 표기했다.  
> **실습 전제:** 농어 원본 배열 `perch_length`, `perch_weight`는 이전 수업에서 준비한 데이터를 사용한다. **녹취록에 원본 배열 전체가 없으므로 임의의 수치로 만들지 않았다.**

## 1. K-최근접 이웃 회귀의 한계

- `KNeighborsRegressor`는 입력 데이터에 **가장 가까운 k개 이웃의 타깃 값 평균**으로 예측한다.
- 농어의 길이가 50일 때 무게 예측값이 약 **1,033g**이었고, 길이를 100으로 늘려도 선택되는 이웃이 같아 같은 값이 나온다는 예시를 살펴봤다.
- 즉, 훈련 데이터 범위를 벗어나면 길이가 계속 늘어나더라도 무게 예측이 더 증가하지 않을 수 있다.
- 이를 해결하기 위해 훈련 데이터의 경향을 식으로 나타내는 **선형 회귀**를 학습했다.

```python
# 이전 시간에 준비한 train_input, train_target 사용
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors=3)
knr.fit(train_input, train_target)
print(knr.predict([[50]]))
print(knr.predict([[100]]))
```

## 2. 선형 회귀(Linear Regression)

선형 회귀는 데이터를 잘 설명하는 **직선의 방정식**을 찾는 방법이다.

\[
y = ax + b
\]

- `x`: 농어의 길이(입력 특성)
- `y`: 농어의 무게(예측 대상)
- `a`: 기울기/계수 → `lr.coef_`
- `b`: 절편 → `lr.intercept_`

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_input, train_target)

print('계수:', lr.coef_)
print('절편:', lr.intercept_)
print('길이 50 예측:', lr.predict([[50]]))

# predict()가 내부에서 적용하는 직선의 방정식을 직접 계산
print('수식 계산:', lr.coef_[0] * 50 + lr.intercept_)
```

강의에서는 계수 약 **39**, 절편 약 **-709**, 길이 50의 예측 무게 약 **1,240g**을 예시로 소개했다. 정확한 소수점 값은 실습 데이터와 실행 결과로 확인해야 한다.

### 직선 그래프

```python
import matplotlib.pyplot as plt

plt.scatter(train_input[:, 0], train_target, label='Train')
plt.plot([15, 50],
         [15 * lr.coef_[0] + lr.intercept_,
          50 * lr.coef_[0] + lr.intercept_], label='Linear regression')
plt.scatter(50, lr.predict([[50]])[0], marker='^', label='Length 50')
plt.xlabel('Length')
plt.ylabel('Weight')
plt.legend()
plt.show()
```

![선형 회귀 직선 그래프](../images/week07_linear_regression.png)

## 3. 직선 모델의 또 다른 한계

- 단순 직선으로는 길이가 작은 일부 농어의 **무게가 음수로 예측**될 수 있다.
- 실제 무게는 음수가 될 수 없으므로 농어 데이터의 곡선형 경향을 더 잘 반영할 방법이 필요하다.
- 이를 위해 입력 특성 `길이`에서 **길이의 제곱**을 새 특성으로 만든다.

```python
# 테스트 데이터에 대한 예측값과 실제 무게 비교
import numpy as np

test_pred = lr.predict(test_input)
print(np.column_stack((test_pred, test_target)))
```

## 4. 2차 다항 회귀(Polynomial Regression)

직선 `y = ax + b` 대신 이차식으로 모델링한다.

\[
y = ax^2 + bx + c
\]

- 입력 특성: `x²`(길이의 제곱), `x`(길이)
- 목표값: 기존과 동일한 **농어의 무게** (`train_target`, `test_target`)
- `LinearRegression` 모델은 그대로 사용하되 **입력 특성을 2개로 확장**한다.

```python
# 원래 train_input, test_input은 (샘플 수, 1) 형태
train_poly = np.column_stack((train_input[:, 0] ** 2, train_input[:, 0]))
test_poly = np.column_stack((test_input[:, 0] ** 2, test_input[:, 0]))

print(train_poly.shape)  # 강의의 42개 훈련 샘플 기준 (42, 2)
print(test_poly.shape)   # 강의의 14개 테스트 샘플 기준 (14, 2)

lr.fit(train_poly, train_target)
print('a, b:', lr.coef_)
print('c:', lr.intercept_)

sample = np.array([[5]])
sample_poly = np.column_stack((sample[:, 0] ** 2, sample[:, 0]))
print('길이 5 예측:', lr.predict(sample_poly))
print('테스트 점수:', lr.score(test_poly, test_target))
```

강의에서는 길이 **5cm**를 넣었을 때, 앞선 직선 회귀와 달리 **약 33.6g**의 양수 예측이 나오는 사례를 확인했다. 이는 해당 데이터에서의 결과이며 모든 길이에 대해 음수가 발생하지 않는다는 보장은 아니다.

### 곡선 그래프

```python
point = np.arange(15, 50)
pred_curve = lr.coef_[0] * point ** 2 + lr.coef_[1] * point + lr.intercept_

plt.scatter(train_input[:, 0], train_target, label='Train')
plt.scatter(test_input[:, 0], test_target, label='Test')
plt.plot(point, pred_curve, label='Polynomial regression')
plt.scatter(sample[0, 0], lr.predict(sample_poly)[0], marker='^', label='Length 5')
plt.xlabel('Length')
plt.ylabel('Weight')
plt.legend()
plt.show()
```

![2차 다항 회귀 곡선 그래프](../images/week07_polynomial_regression.png)

> **실습 주의:** 앞의 선형 회귀 `lr`를 다항 회귀 데이터로 다시 학습시키면 모델의 `coef_`, `intercept_`가 바뀐다. 직선과 곡선을 같은 노트북에서 각각 다시 그릴 때는 `linear_lr`, `poly_lr`처럼 **서로 다른 객체 이름**을 사용하는 편이 안전하다.

## 5. 중간고사 안내 (교수님 수업 발언 기준)

- **시험 일정:** 21일 수요일, 수업 시간 중 실시한다고 안내함. 정확한 날짜·시간은 강의실 공지를 재확인할 것.
- **시험 범위:** **5장 선형 회귀까지**, 1차 직선 회귀와 길이 제곱을 추가한 **2차 다항 회귀 포함**.
- **형식:** 필기, **단답형과 코드 중심**. 전체 프로그램 작성보다는 빈칸에 함수/변수 채우기, 코드 일부 작성, 실행 결과 예상, 간단한 결과 해석을 강조함.
- **교수님 강조:** 개념 설명을 길게 쓰는 시험이 아니라 코드를 이해하고 쓸 수 있어야 한다고 안내함.

### 꼭 복습할 세 가지 모델

| 모델 | 클래스 | 핵심 원리 |
|---|---|---|
| KNN 분류 | `KNeighborsClassifier` | 가까운 이웃의 다수결로 클래스 예측 |
| KNN 회귀 | `KNeighborsRegressor` | 가까운 이웃의 타깃 평균으로 값 예측 |
| 선형 회귀 | `LinearRegression` | 직선 또는 확장한 특성으로 회귀식 학습 |

### 코드 필기 핵심 함수/속성

| 목적 | 코드 | 패키지 |
|---|---|---|
| 훈련·테스트 분리 | `train_test_split()` | `sklearn.model_selection` |
| KNN 분류 | `KNeighborsClassifier()` | `sklearn.neighbors` |
| KNN 회귀 | `KNeighborsRegressor()` | `sklearn.neighbors` |
| 선형 회귀 | `LinearRegression()` | `sklearn.linear_model` |
| 학습 | `.fit(X, y)` | 모델 공통 |
| 평가 | `.score(X, y)` | 모델 공통 |
| 예측 | `.predict(X)` | 모델 공통 |
| 분류 확률 | `.predict_proba(X)` | 지원하는 분류 모델 |
| 이웃 거리·위치 | `.kneighbors(X)` | KNN 모델 |
| 이웃 수 조절 | `n_neighbors` | KNN 생성자 인자 |
| 회귀 계수·절편 | `.coef_`, `.intercept_` | 선형 회귀 |
| 평균절대오차 | `mean_absolute_error()` | `sklearn.metrics` |
| 열 방향 결합 | `np.column_stack()` | NumPy |
| 배열 연결 | `np.concatenate()` | NumPy |
| 연속 값 생성 | `np.arange()` | NumPy |
| 배열 형태 변경 | `.reshape(-1, 1)` | NumPy 배열 |
| 산점도·선 그래프 | `plt.scatter()`, `plt.plot()` | Matplotlib |

시험 준비에 포함할 데이터 관련 개념:

- 훈련 데이터와 테스트 데이터의 역할 및 분리 필요성
- **분류**(클래스 예측)와 **회귀**(숫자 값 예측)의 차이
- **특성 스케일 차이**와 Z점수 표준화: `(데이터 - 훈련 데이터 평균) / 훈련 데이터 표준편차`
- KNN의 이웃 수 `n_neighbors`와 `kneighbors()`
- 분류 `.score()`는 정확도, 회귀 `.score()`는 결정계수 \(R^2\)
- `mean_absolute_error()`는 평균 절대 오차(MAE)를 계산함

## 6. 시험 대비 빈칸 연습

```python
# ① 데이터 나누기
from sklearn.model_selection import __________
train_input, test_input, train_target, test_target = \
    __________(perch_length, perch_weight, random_state=42)

# ② 입력 특성을 2차원으로
train_input = train_input.__________(-1, 1)

# ③ 선형 회귀
from sklearn.linear_model import __________
lr = __________()
lr.__________(train_input, train_target)
print(lr.__________([[50]]))

# ④ 2차 다항 특성 만들기
train_poly = np.__________((train_input[:, 0] ** 2, train_input[:, 0]))

# ⑤ 계수와 절편 확인
print(lr.__________)
print(lr.__________)
```

<details>
<summary>정답 확인</summary>

① `train_test_split`, `train_test_split`  
② `reshape`  
③ `LinearRegression`, `LinearRegression`, `fit`, `predict`  
④ `column_stack`  
⑤ `coef_`, `intercept_`

</details>

## 7. 이번 주 핵심 요약

1. KNN 회귀는 가까운 이웃의 평균에 의존하므로 훈련 범위를 벗어나는 값에 한계가 있다.
2. 선형 회귀는 `y = ax + b`에서 계수·절편을 학습하지만, 농어 데이터에서는 작은 길이에 음수 무게를 예측할 수 있다.
3. `np.column_stack((x**2, x))`로 입력 특성을 확장하면 `LinearRegression`으로도 `y = ax² + bx + c` 형태의 **2차 다항 회귀**를 학습할 수 있다.
4. 중간고사는 **세 모델과 데이터 전처리·학습·평가 코드**를 직접 이해하고 빈칸을 채우는 연습이 중요하다.

---

**자료 범위:** 이 노트는 업로드된 7주차 오프라인 녹취록을 정리한 것으로, 교수님이 별도로 올렸다고 언급한 실제 소스 코드나 시험 안내 파일은 포함되어 있지 않다. 실습 코드는 강의에서 설명한 동작을 기준으로 재구성한 예시다.
