# @en ElasticNet: L1 and L2 penalties together; l1_ratio sets the mix.
# @ru ElasticNet: штрафы L1 и L2 вместе; l1_ratio задаёт пропорцию.
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# @en Worked example: alpha = 1, l1_ratio = 0.5 gives w = 0.44.
# @ru Пример: alpha = 1, l1_ratio = 0.5 даёт w = 0.44.
x = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 6])
m = ElasticNet(alpha=1.0, l1_ratio=0.5).fit(x, y)
print(f"w = {m.coef_[0]:.3f}, b = {m.intercept_:.3f}")  # 0.440, 2.880

# @en A group of 3 correlated useful features and 5 noise features.
# @ru Группа из 3 коррелирующих полезных признаков и 5 шумовых.
rng = np.random.default_rng(0)
base = rng.normal(size=(400, 1))
useful = base + rng.normal(scale=0.1, size=(400, 3))
noise = rng.normal(size=(400, 5))
X = np.hstack([useful, noise])
y = useful.sum(axis=1) * 2 + rng.normal(scale=0.5, size=400)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# @en l1_ratio=1 is Lasso, closer to 0 is Ridge-like. Defaults: alpha=1.0, l1_ratio=0.5, max_iter=1000.
# @ru l1_ratio=1 — это Lasso, ближе к 0 — похоже на Ridge. По умолчанию: alpha=1.0, l1_ratio=0.5, max_iter=1000.
for ratio in (1.0, 0.5, 0.1):
    model = ElasticNet(alpha=0.1, l1_ratio=ratio).fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    print(f"l1_ratio={ratio}: weights {model.coef_.round(2)}  test R2 {r2:.3f}")
print("prediction:", model.predict(X_test[:1]).round(2))
