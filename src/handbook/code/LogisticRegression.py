# @en Logistic regression: probabilities from a weighted sum and a sigmoid.
# @ru Логистическая регрессия: вероятности из взвешенной суммы и сигмоиды.
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

# @en The six points from the worked example.
# @ru Шесть точек из примера.
x = np.array([[-2], [-1], [0], [1], [2], [3]])
y = np.array([0, 0, 1, 0, 1, 1])
small = LogisticRegression(C=1.0).fit(x, y)
w, b = small.coef_[0, 0], small.intercept_[0]
print(f"w = {w:.3f}, b = {b:.3f}")  # w = 0.767, b = -0.383

# @en The sigmoid by hand gives the same probability as predict_proba.
# @ru Сигмоида вручную даёт ту же вероятность, что и predict_proba.
z = w * 1.5 + b
print("by hand:", round(1 / (1 + np.exp(-z)), 3), " sklearn:", small.predict_proba([[1.5]])[0, 1].round(3))

# @en A real-sized problem: 500 rows, 4 features, 2 classes.
# @ru Задача реального размера: 500 строк, 4 признака, 2 класса.
X, y = make_classification(n_samples=500, n_features=4, n_informative=3, n_redundant=0, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# @en Defaults: C=1.0, max_iter=100, fit_intercept=True, class_weight=None (L2 penalty, lbfgs solver).
# @ru По умолчанию: C=1.0, max_iter=100, fit_intercept=True, class_weight=None (штраф L2, решатель lbfgs).
model = LogisticRegression()
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)
print("test accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 3))
print("test log-loss:", round(log_loss(y_test, proba), 3))
print("weights:", model.coef_.round(2), "iterations used:", model.n_iter_)

# @en Probability of each class for a new row.
# @ru Вероятность каждого класса для новой строки.
print("P(class):", model.predict_proba([[0.2, -1.0, 0.5, 1.5]]).round(3))
