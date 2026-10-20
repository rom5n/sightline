# @en Random forest: many trees on bootstrap samples with random feature subsets, probabilities averaged.
# @ru Случайный лес: много деревьев на бутстреп-выборках со случайными подмножествами признаков, вероятности усредняются.
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# @en Worked example: three trees, a 2 : 1 vote for class 1 gives probability 0.667.
# @ru Пример: три дерева, голоса 2 : 1 за класс 1 дают вероятность 0.667.
X8 = np.array([[1, 1], [2, 1], [1, 2], [4, 4], [5, 4], [4, 5], [3, 3], [2, 4]])
y8 = np.array([0, 0, 0, 1, 1, 1, 1, 0])
small = RandomForestClassifier(n_estimators=3, random_state=1).fit(X8, y8)
q = [[2.5, 3.5]]
print("each tree:", [float(t.predict_proba(q)[0, 1]) for t in small.estimators_], " forest:", small.predict_proba(q)[0].round(3))

X, y = make_moons(n_samples=600, noise=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# @en Defaults: n_estimators=100, max_depth=None, max_features='sqrt', min_samples_leaf=1.
# @ru По умолчанию: n_estimators=100, max_depth=None, max_features='sqrt', min_samples_leaf=1.
forest = RandomForestClassifier(oob_score=True, random_state=0).fit(X_train, y_train)
print("test accuracy:", round(accuracy_score(y_test, forest.predict(X_test)), 3))
print("out-of-bag accuracy:", round(forest.oob_score_, 3))
print("feature importances:", forest.feature_importances_.round(3))

# @en More trees: accuracy settles; adding trees does not cause overfitting.
# @ru Больше деревьев: точность выходит на плато; от добавления деревьев переобучения нет.
for n in (1, 10, 100, 300):
    f = RandomForestClassifier(n_estimators=n, random_state=0).fit(X_train, y_train)
    print(f"{n:>3} trees: test accuracy {accuracy_score(y_test, f.predict(X_test)):.3f}")
print("P(class) for one row:", forest.predict_proba(X_test[:1]).round(3))
