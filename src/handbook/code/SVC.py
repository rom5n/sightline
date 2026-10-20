# @en SVC: the widest-margin border; only the support vectors decide, a kernel lets it bend.
# @ru SVC: граница с самым широким зазором; решают только опорные векторы, ядро позволяет ей изгибаться.
import numpy as np
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# @en Worked example: six points, a linear kernel and a large C (almost a hard margin).
# @ru Пример: шесть точек, линейное ядро и большое C (почти жёсткий зазор).
X6 = np.array([[1, 2], [2, 3], [2, 1], [4, 3], [5, 4], [4, 1]])
y6 = np.array([0, 0, 0, 1, 1, 1])
lin = SVC(kernel="linear", C=1000).fit(X6, y6)
print("w =", lin.coef_.round(3), "b =", lin.intercept_.round(3))  # w = [1, 0], b = -3
print("support vectors:", lin.support_vectors_.tolist(), "alpha*y:", lin.dual_coef_.round(3))
print("margin width:", round(2 / np.linalg.norm(lin.coef_), 3))  # 2.0

# @en The decision value from the support vectors alone equals decision_function.
# @ru Значение решения только по опорным векторам совпадает с decision_function.
q = np.array([2.5, 2.0])
by_hand = sum(a * (sv @ q) for a, sv in zip(lin.dual_coef_[0], lin.support_vectors_)) + lin.intercept_[0]
print("f(q) by hand:", round(by_hand, 3), " sklearn:", lin.decision_function([q]).round(3))  # -0.5

# @en Curved data: the RBF kernel. Defaults: C=1.0, kernel='rbf', gamma='scale', degree=3 (poly only).
# @ru Изогнутые данные: ядро RBF. По умолчанию: C=1.0, kernel='rbf', gamma='scale', degree=3 (только для poly).
X, y = make_circles(n_samples=600, noise=0.1, factor=0.5, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
for kernel in ("linear", "rbf"):
    model = make_pipeline(StandardScaler(), SVC(kernel=kernel)).fit(X_train, y_train)
    print(f"{kernel}: test accuracy {accuracy_score(y_test, model.predict(X_test)):.3f}, "
          f"support vectors {model[-1].n_support_.sum()}")

# @en C trades margin width for training errors.
# @ru C выбирает между шириной зазора и ошибками на обучении.
for C in (0.1, 1, 100):
    m = make_pipeline(StandardScaler(), SVC(C=C)).fit(X_train, y_train)
    print(f"C={C}: test accuracy {accuracy_score(y_test, m.predict(X_test)):.3f}, support vectors {m[-1].n_support_.sum()}")
print("decision value for one row:", model.decision_function(X_test[:1]).round(3))
