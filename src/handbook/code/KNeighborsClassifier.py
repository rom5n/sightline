# @en k nearest neighbours: no training, a new point takes the vote of the k closest stored points.
# @ru k ближайших соседей: обучения нет, новая точка получает голосование k ближайших сохранённых точек.
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# @en Worked example: six points, query (3, 2), k = 3.
# @ru Пример: шесть точек, запрос (3, 2), k = 3.
X6 = np.array([[0.5, 0.5], [2, 1], [1, 3.5], [4, 4.5], [5.5, 3], [4, 2]])
y6 = np.array([0, 0, 0, 1, 1, 1])
q = np.array([[3, 2]])
print("distances:", np.sqrt(((X6 - q) ** 2).sum(axis=1)).round(3))  # 2.915 1.414 2.5 2.693 2.693 1.0
for weights in ("uniform", "distance"):
    m = KNeighborsClassifier(n_neighbors=3, weights=weights).fit(X6, y6)
    print(f"{weights}: P(class) = {m.predict_proba(q)[0].round(3)}")  # 0.667/0.333 and 0.525/0.475
dist, idx = m.kneighbors(q)
print("the 3 neighbours:", idx[0].tolist(), "at", dist[0].round(3).tolist())

X, y = make_moons(n_samples=500, noise=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# @en Defaults: n_neighbors=5, weights='uniform', metric='minkowski', p=2 (Euclidean). Scale the features.
# @ru По умолчанию: n_neighbors=5, weights='uniform', metric='minkowski', p=2 (евклидово). Масштабируйте признаки.
# @en Small k follows the noise, large k blurs the border; Sightline plots this curve for k = 1..30.
# @ru Малое k повторяет шум, большое размывает границу; Sightline рисует эту кривую для k = 1..30.
for k in (1, 5, 15, 50):
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)).fit(X_train, y_train)
    print(f"k={k:>2}: test accuracy {accuracy_score(y_test, model.predict(X_test)):.3f}")
print("P(class) for one row:", model.predict_proba(X_test[:1]).round(3))
