# @en Bagging: copies of one model on bootstrap samples, predictions averaged.
# @ru Бэггинг: копии одной модели на бутстреп-выборках, предсказания усредняются.
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# @en Worked example: three bootstrap samples of eight rows (row numbers can repeat).
# @ru Пример: три бутстреп-выборки по восемь строк (номера строк могут повторяться).
X8 = np.array([[1, 1], [2, 1], [1, 2], [4, 4], [5, 4], [4, 5], [3, 3], [2, 4]])
y8 = np.array([0, 0, 0, 1, 1, 1, 1, 0])
small = BaggingClassifier(n_estimators=3, random_state=0).fit(X8, y8)
print("samples:", [sorted(s.tolist()) for s in small.estimators_samples_])
votes = [int(e.predict([[3, 2.5]])[0]) for e in small.estimators_]
print("votes for (3, 2.5):", votes, "->", small.predict_proba([[3, 2.5]])[0].round(3))

X, y = make_moons(n_samples=600, noise=0.35, random_state=0)

# @en One deep tree against a bag of 10 of them, averaged over 10 different train/test splits
# @en (one split alone is too noisy to compare). Defaults: n_estimators=10, max_samples=None
# @en (as many draws as rows), max_features=1.0.
# @ru Одно глубокое дерево против мешка из 10 таких, в среднем по 10 разным разбиениям
# @ru (одного разбиения мало для сравнения). По умолчанию: n_estimators=10, max_samples=None
# @ru (столько выборов, сколько строк), max_features=1.0.
tree_cv = cross_val_score(DecisionTreeClassifier(random_state=0), X, y, cv=10)
bag_cv = cross_val_score(BaggingClassifier(DecisionTreeClassifier(), random_state=0), X, y, cv=10)
print("one tree:", round(tree_cv.mean(), 3), " bagged trees:", round(bag_cv.mean(), 3))  # 0.833 vs 0.857

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
bag = BaggingClassifier(DecisionTreeClassifier(), random_state=0).fit(X_train, y_train)

# @en Any model can be bagged; here k-nearest neighbours, each on half of the rows.
# @ru «Упаковать» можно любую модель; здесь k ближайших соседей, каждый на половине строк.
knn_bag = BaggingClassifier(KNeighborsClassifier(), n_estimators=20, max_samples=0.5, random_state=0)
knn_bag.fit(X_train, y_train)
print("bagged KNN:", round(accuracy_score(y_test, knn_bag.predict(X_test)), 3))
print("P(class) for one row:", bag.predict_proba(X_test[:1]).round(2))
