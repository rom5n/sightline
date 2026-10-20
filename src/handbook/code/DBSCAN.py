# @en DBSCAN: clusters grow from dense points; points in sparse areas are labelled noise (-1).
# @ru DBSCAN: группы растут из плотных точек; точки в разреженных местах помечаются шумом (-1).
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# @en Worked example: points on a line, eps = 1.5, min_samples = 3 (the point itself counts).
# @ru Пример: точки на прямой, eps = 1.5, min_samples = 3 (сама точка тоже считается).
x = np.array([[1], [2], [3], [8], [9], [10], [20]])
db = DBSCAN(eps=1.5, min_samples=3).fit(x)
counts = [int(np.sum(np.abs(x[:, 0] - v) <= 1.5)) for v in x[:, 0]]
print("neighbours within eps:", counts)  # 2 3 2 2 3 2 1
print("core points:", x[db.core_sample_indices_, 0].tolist())  # 2 and 9
print("labels:", db.labels_.tolist())  # [0, 0, 0, 1, 1, 1, -1]

X, _ = make_moons(n_samples=400, noise=0.06, random_state=0)
X = StandardScaler().fit_transform(X)

# @en A k-distance curve helps to choose eps: look for the knee.
# @ru Кривая k-расстояний помогает выбрать eps: ищите «колено».
dist, _ = NearestNeighbors(n_neighbors=5).fit(X).kneighbors(X)
print("5th-neighbour distance, 50/90/99 % quantiles:", np.quantile(dist[:, -1], [0.5, 0.9, 0.99]).round(3))

# @en Defaults: eps=0.5, min_samples=5, metric='euclidean'.
# @ru По умолчанию: eps=0.5, min_samples=5, metric='euclidean'.
for eps in (0.1, 0.3, 0.6):
    labels = DBSCAN(eps=eps).fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"eps={eps}: {n_clusters} clusters, {int(np.sum(labels == -1))} noise points")
# @en DBSCAN has no predict(): new points need fit_predict on the new data or a classifier trained on the labels.
# @ru У DBSCAN нет predict(): для новых точек нужен fit_predict на новых данных или классификатор, обученный на метках.
