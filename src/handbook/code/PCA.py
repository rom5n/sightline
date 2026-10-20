# @en PCA: rotate the data onto the directions of largest spread and keep the first few.
# @ru PCA: поворачиваем данные на направления наибольшего разброса и оставляем первые.
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# @en Worked example: five points, covariance matrix, eigenvalues 4.75 and 0.25.
# @ru Пример: пять точек, ковариационная матрица, собственные значения 4.75 и 0.25.
P = np.array([[1, 2], [2, 3], [3, 5], [4, 4], [5, 6]], dtype=float)
C = np.cov(P.T)
values, vectors = np.linalg.eigh(C)
print("covariance:", C.tolist())  # [[2.5, 2.25], [2.25, 2.5]]
print("eigenvalues:", values.round(3), "explained share:", (values[::-1] / values.sum()).round(3))  # 0.95, 0.05
pca = PCA(n_components=2).fit(P)
print("components:", pca.components_.round(3).tolist())
print("first coordinate of each point:", pca.transform(P)[:, 0].round(3))  # -2.828 -1.414 0.707 0.707 2.828

# @en Real data: 4 flower measurements reduced to 2 coordinates.
# @ru Реальные данные: 4 измерения цветков сжимаются до 2 координат.
X = StandardScaler().fit_transform(load_iris().data)
# @en Defaults: n_components=None (keep all), whiten=False, svd_solver='auto'. Sightline keeps 2 for the plot.
# @ru По умолчанию: n_components=None (оставить все), whiten=False, svd_solver='auto'. Sightline оставляет 2 для графика.
full = PCA().fit(X)
print("explained variance ratio:", full.explained_variance_ratio_.round(3))
print("cumulative:", np.cumsum(full.explained_variance_ratio_).round(3))
two = PCA(n_components=2).fit(X)
Z = two.transform(X)
print("2-D shape:", Z.shape, "kept:", round(float(two.explained_variance_ratio_.sum()), 3))
back = two.inverse_transform(Z)
print("mean squared reconstruction error:", round(float(np.mean((X - back) ** 2)), 4))
print("loadings of PC1:", two.components_[0].round(3))
