# @en Gradient boosting: small trees added one by one, each fitted to the remaining error.
# @ru Градиентный бустинг: маленькие деревья добавляются по одному, каждое учится на оставшейся ошибке.
import numpy as np
from sklearn.datasets import make_circles
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# @en Worked example: four points, one stump, learning rate 0.1.
# @ru Пример: четыре точки, один пень, скорость обучения 0.1.
x = np.array([[1], [2], [3], [4]])
y = np.array([0, 1, 1, 1])
one = GradientBoostingClassifier(n_estimators=1, max_depth=1, learning_rate=0.1).fit(x, y)
print("start: log-odds", round(float(np.log(0.75 / 0.25)), 4))  # 1.0986
print("after 1 tree, raw scores:", one.decision_function(x).round(4))  # 0.6986 and 1.2319
print("probabilities:", one.predict_proba(x)[:, 1].round(3))  # 0.668 and 0.774

X, y = make_circles(n_samples=800, noise=0.15, factor=0.5, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# @en Defaults: n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0.
# @ru По умолчанию: n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0.
model = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)
print("test accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 3))

# @en staged_predict replays the ensemble after each stage (Sightline animates exactly this).
# @ru staged_predict воспроизводит ансамбль после каждого этапа (именно это анимирует Sightline).
for stage, pred in enumerate(model.staged_predict(X_test), start=1):
    if stage in (1, 10, 50, 100):
        print(f"after {stage:>3} trees: test accuracy {accuracy_score(y_test, pred):.3f}")

# @en Smaller steps need more trees; together with subsample < 1 this usually generalises better.
# @ru Меньший шаг требует больше деревьев; вместе с subsample < 1 это обычно обобщает лучше.
slow = GradientBoostingClassifier(learning_rate=0.03, n_estimators=300, subsample=0.8, random_state=0)
slow.fit(X_train, y_train)
print("lr=0.03, 300 trees, subsample=0.8:", round(accuracy_score(y_test, slow.predict(X_test)), 3))
print("P(class) for one row:", model.predict_proba(X_test[:1]).round(3))
