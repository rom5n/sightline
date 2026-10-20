# @en SGDClassifier: a linear model trained by stochastic gradient descent, one small step per point.
# @ru SGDClassifier: линейная модель, обучаемая стохастическим градиентным спуском, маленький шаг на каждую точку.
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# @en The hand-computed epoch from this chapter (hinge loss, fixed step 0.1, no penalty):
# @en a point inside the margin (label * score < 1) moves the weights towards its own side.
# @ru Эпоха из этой главы, посчитанная вручную (hinge loss, постоянный шаг 0.1, без штрафа):
# @ru точка внутри зазора (метка * оценка < 1) сдвигает веса в свою сторону.
pts = [((1, 2), 1), ((2, 0), -1), ((0, 1), 1)]
w, b, eta = np.zeros(2), 0.0, 0.1
for x, label in pts:
    if label * (w @ np.array(x) + b) < 1:
        w, b = w + eta * label * np.array(x), b + eta * label
print("after one epoch: w =", w.round(2), "b =", round(b, 2))  # [-0.1, 0.3], 0.1

# @en Real use: scale the features first, SGD is sensitive to their units.
# @ru На практике сначала масштабируем признаки: SGD чувствителен к их единицам.
X, y = make_classification(n_samples=2000, n_features=6, n_informative=4, n_clusters_per_class=1, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# @en Defaults: loss='hinge' (a linear SVM), penalty='l2', alpha=0.0001, max_iter=1000.
# @ru По умолчанию: loss='hinge' (линейный SVM), penalty='l2', alpha=0.0001, max_iter=1000.
model = make_pipeline(StandardScaler(), SGDClassifier(random_state=0))
model.fit(X_train, y_train)
print("test accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 3))

# @en Online learning: partial_fit feeds the data epoch by epoch (this is what Sightline animates).
# @ru Онлайн-обучение: partial_fit подаёт данные эпоха за эпохой (это и анимирует Sightline).
scaler = StandardScaler().fit(X_train)
online = SGDClassifier(loss="log_loss", random_state=0)
for epoch in range(1, 6):
    online.partial_fit(scaler.transform(X_train), y_train, classes=[0, 1])
    acc = accuracy_score(y_test, online.predict(scaler.transform(X_test)))
    print(f"epoch {epoch}: test accuracy {acc:.3f}")
print("P(class) for one row:", online.predict_proba(scaler.transform(X_test[:1])).round(3))
