# @en Histogram gradient boosting: features are binned first, so each split is found fast.
# @ru Гистограммный градиентный бустинг: признаки сначала раскладываются по корзинам, поэтому разбиения ищутся быстро.
import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# @en Worked example: the same four points as in the gradient boosting chapter.
# @en min_samples_leaf=1 is needed here, the default (20) would refuse to split 4 points.
# @ru Пример: те же четыре точки, что в главе о градиентном бустинге.
# @ru Здесь нужен min_samples_leaf=1: по умолчанию (20) 4 точки не будут разбиты.
x = np.array([[1], [2], [3], [4]])
y = np.array([0, 1, 1, 1])
tiny = HistGradientBoostingClassifier(max_iter=1, max_depth=1, min_samples_leaf=1, early_stopping=False).fit(x, y)
print("raw scores after 1 tree:", tiny.decision_function(x).round(4))  # 0.6986 and 1.2319

# @en 20 000 rows with some missing values: HistGradientBoosting handles NaN without imputation.
# @ru 20 000 строк с пропусками: HistGradientBoosting понимает NaN без заполнения.
X, y = make_classification(n_samples=20000, n_features=10, n_informative=6, random_state=0)
rng = np.random.default_rng(0)
X[rng.random(X.shape) < 0.05] = np.nan
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# @en Defaults: max_iter=100, learning_rate=0.1, max_leaf_nodes=31, max_depth=None, l2_regularization=0.0;
# @en early_stopping='auto' switches on above 10 000 rows (10 % of the training rows become validation).
# @ru По умолчанию: max_iter=100, learning_rate=0.1, max_leaf_nodes=31, max_depth=None, l2_regularization=0.0;
# @ru early_stopping='auto' включается при числе строк больше 10 000 (10 % обучающих строк идут на проверку).
t = time.perf_counter()
model = HistGradientBoostingClassifier(random_state=0).fit(X_train, y_train)
hist_s = time.perf_counter() - t
print("test accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 3))
print("trees actually built:", model.n_iter_, "(early stopping)")
print(f"fit time: {hist_s:.2f} s")

# @en Classic gradient boosting needs complete data, so compare on rows without NaN.
# @ru Классическому градиентному бустингу нужны полные данные, поэтому сравниваем на строках без NaN.
ok = ~np.isnan(X_train).any(axis=1)
t = time.perf_counter()
GradientBoostingClassifier(random_state=0).fit(X_train[ok], y_train[ok])
print(f"classic GradientBoosting on {ok.sum()} complete rows: {time.perf_counter() - t:.2f} s")
print("P(class) for one row:", model.predict_proba(X_test[:1]).round(3))
