# @en Decision tree: yes/no questions chosen to make each group as pure as possible.
# @ru Дерево решений: вопросы «да/нет», выбранные так, чтобы каждая группа была как можно чище.
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

# @en The six points from the worked example: the tree finds x <= 2.5 and then x <= 5.5.
# @ru Шесть точек из примера: дерево находит x <= 2.5, а затем x <= 5.5.
x = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 1, 1, 1, 0])
small = DecisionTreeClassifier(random_state=0).fit(x, y)
print(export_text(small, feature_names=["x"]))


# @en Gini impurity by hand for the root split x <= 2.5.
# @ru Неоднородность Джини вручную для корневого разбиения x <= 2.5.
def gini(labels):
    p = np.bincount(labels, minlength=2) / len(labels)
    return 1 - np.sum(p ** 2)


left, right = y[x[:, 0] <= 2.5], y[x[:, 0] > 2.5]
print("weighted gini:", round((len(left) * gini(left) + len(right) * gini(right)) / len(y), 3))  # 0.25

# @en Real data: noisy moons; a limited depth keeps the tree from memorising noise.
# @ru Реальные данные: зашумлённые луны; ограничение глубины не даёт дереву запомнить шум.
X, y = make_moons(n_samples=500, noise=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
# @en Defaults: max_depth=None, criterion='gini', min_samples_split=2, min_samples_leaf=1.
# @ru По умолчанию: max_depth=None, criterion='gini', min_samples_split=2, min_samples_leaf=1.
for depth in (None, 4):
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_train, y_train)
    acc = accuracy_score(y_test, tree.predict(X_test))
    print(f"max_depth={depth}: leaves {tree.get_n_leaves()}, test accuracy {acc:.3f}")
print("path of one point (node ids):", tree.decision_path(X_test[:1]).indices)
print("prediction:", tree.predict(X_test[:1]), "probabilities:", tree.predict_proba(X_test[:1]).round(2))
