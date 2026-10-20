# @en Lasso: an L1 penalty sets useless weights to exactly zero.
# @ru Lasso: штраф L1 обнуляет бесполезные веса.
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# @en Worked example: the slope shrinks by alpha and stops at zero.
# @ru Пример: наклон уменьшается на alpha и останавливается в нуле.
x = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 6])
for alpha in (0.5, 1.0, 2.0):
    m = Lasso(alpha=alpha).fit(x, y)
    print(f"alpha={alpha}: w = {m.coef_[0]:.3f}, b = {m.intercept_:.3f}")  # 0.55, 0.30, 0.00

# @en 10 features, only 3 of them matter.
# @ru 10 признаков, важны только 3.
X, y, true_w = make_regression(n_samples=300, n_features=10, n_informative=3, noise=5.0, coef=True, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# @en Defaults: alpha=1.0, max_iter=1000, fit_intercept=True.
# @ru По умолчанию: alpha=1.0, max_iter=1000, fit_intercept=True.
model = Lasso().fit(X_train, y_train)
print("true weights: ", true_w.round(1))
print("lasso weights:", model.coef_.round(1))
print("features kept:", int(np.sum(model.coef_ != 0)), "of", X.shape[1])
print("test R2:", round(r2_score(y_test, model.predict(X_test)), 3), " iterations:", model.n_iter_)
print("prediction:", model.predict(X_test[:1]).round(1))
