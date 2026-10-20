# @en k-means: assign each point to its nearest centre, move each centre to the mean of its points, repeat.
# @ru k-средних: каждая точка идёт к ближайшему центру, центр сдвигается в среднее своих точек, и так по кругу.
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# @en Worked example: seven points, k = 2, starting centres (1, 1) and (5, 7).
# @ru Пример: семь точек, k = 2, начальные центры (1, 1) и (5, 7).
P = np.array([[1, 1], [1.5, 2], [3, 3.5], [5, 7], [3.5, 5], [4.5, 5], [3.5, 4.5]])
centres = np.array([[1.0, 1.0], [5.0, 7.0]])
for step in range(1, 4):
    d = np.linalg.norm(P[:, None, :] - centres[None], axis=2)
    labels = d.argmin(axis=1)
    new = np.array([P[labels == k].mean(axis=0) for k in range(2)])
    print(f"step {step}: labels {labels.tolist()}, centres {new.round(3).tolist()}")
    if np.allclose(new, centres):
        break
    centres = new
km = KMeans(n_clusters=2, init=np.array([[1, 1], [5, 7]]), n_init=1).fit(P)
print("sklearn:", km.cluster_centers_.round(3).tolist(), "inertia", round(km.inertia_, 3))  # 10.708

X, _ = make_blobs(n_samples=600, centers=4, cluster_std=1.0, random_state=0)
# @en Defaults: n_clusters=8, init='k-means++', n_init='auto' (1 run with k-means++), max_iter=300.
# @ru По умолчанию: n_clusters=8, init='k-means++', n_init='auto' (1 запуск с k-means++), max_iter=300.
# @en Try several k: inertia always falls, silhouette peaks at a sensible k.
# @ru Попробуем разные k: инерция всегда падает, а силуэт достигает максимума при разумном k.
for k in (2, 3, 4, 6, 8):
    model = KMeans(n_clusters=k, random_state=0).fit(X)
    print(f"k={k}: inertia {model.inertia_:8.1f}, silhouette {silhouette_score(X, model.labels_):.3f}")
model = KMeans(n_clusters=4, random_state=0).fit(X)
print("cluster of a new point:", model.predict([[0.0, 4.0]]), "distances to centres:", model.transform([[0.0, 4.0]]).round(2))
