# @en Ridge: linear regression whose weights are pulled towards zero by an L2 penalty.
# @ru Ridge: линейная регрессия, чьи веса штраф L2 тянет к нулю.
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# @en Worked example: same five points, alpha = 1 and alpha = 10.
# @ru Пример: те же пять точек, alpha = 1 и alpha = 10.
x = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 6])
for alpha in (1.0, 10.0):
    m = Ridge(alpha=alpha).fit(x, y)
    print(f"alpha={alpha}: w = {m.coef_[0]:.3f}, b = {m.intercept_:.3f}")  # 0.727/2.018, 0.400/3.000

# @en Two almost identical features: plain least squares splits the weight erratically, Ridge shares it.
# @ru Два почти одинаковых признака: обычные МНК делят вес хаотично, Ridge — поровну.
rng = np.random.default_rng(0)
a = rng.normal(size=200)
X = np.c_[a, a + rng.normal(scale=0.01, size=200)]
target = 3 * a + rng.normal(scale=0.5, size=200)
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.25, random_state=42)

ols = LinearRegression().fit(X_train, y_train)
# @en Defaults: alpha=1.0, fit_intercept=True.
# @ru По умолчанию: alpha=1.0, fit_intercept=True.
ridge = Ridge().fit(X_train, y_train)
print("least squares weights:", ols.coef_.round(2))
print("ridge weights:        ", ridge.coef_.round(2))
print("test R2  OLS:", round(r2_score(y_test, ols.predict(X_test)), 3), " Ridge:", round(r2_score(y_test, ridge.predict(X_test)), 3))
print("prediction for [1, 1]:", ridge.predict([[1.0, 1.0]]).round(2))
