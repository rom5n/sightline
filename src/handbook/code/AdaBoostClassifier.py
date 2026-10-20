# @en AdaBoost: stumps trained in turn; misclassified points get more weight for the next one.
# @ru AdaBoost: пни обучаются по очереди; ошибочно классифицированные точки получают больший вес для следующего.
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# @en Worked example: six points, two stumps.
# @ru Пример: шесть точек, два пня.
x = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 1, 1, 1, 0])
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=2, learning_rate=1.0).fit(x, y)
print("errors:", ada.estimator_errors_.round(4))  # 0.1667, 0.2
print("say (alpha):", ada.estimator_weights_.round(4))  # ln 5 = 1.6094, ln 4 = 1.3863
print("thresholds:", [float(e.tree_.threshold[0]) for e in ada.estimators_])  # 2.5, 5.5
print("predictions:", ada.predict(x))  # x = 6 is still wrong after two stumps

X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# @en Defaults: n_estimators=50, learning_rate=1.0, estimator = a depth-1 tree (stump).
# @ru По умолчанию: n_estimators=50, learning_rate=1.0, estimator — дерево глубины 1 (пень).
model = AdaBoostClassifier(random_state=0).fit(X_train, y_train)
print("test accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 3))
for stage, pred in enumerate(model.staged_predict(X_test), start=1):
    if stage in (1, 5, 20, 50):
        print(f"after {stage:>2} stumps: {accuracy_score(y_test, pred):.3f}")

# @en Deeper base trees (Sightline: "estimator depth") fit more complex borders per round,
# @en but on noisy labels they also overfit sooner.
# @ru Более глубокие базовые деревья (в Sightline — «глубина базового дерева») за раунд учат более сложные границы,
# @ru но на шумных метках и переобучаются раньше.
deep = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=50, random_state=0).fit(X_train, y_train)
print("depth-3 base trees:", round(accuracy_score(y_test, deep.predict(X_test)), 3))
print("P(class) for one row:", model.predict_proba(X_test[:1]).round(3))
