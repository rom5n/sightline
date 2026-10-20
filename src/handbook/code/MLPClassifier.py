# @en A multi-layer perceptron: layers of weighted sums and nonlinearities, trained by backpropagation.
# @ru Многослойный перцептрон: слои взвешенных сумм и нелинейностей, обучение обратным распространением.
import numpy as np
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# @en Worked example: the 2-2-1 network from this chapter, forward pass and one gradient.
# @ru Пример: сеть 2-2-1 из этой главы, прямой проход и один градиент.
x = np.array([1.0, 2.0])
# @en W1: rows = inputs, columns = hidden neurons.
# @ru W1: строки — входы, столбцы — скрытые нейроны.
W1 = np.array([[0.5, -1.0], [1.0, 0.5]])
b1 = np.array([-0.5, 0.2])
v, c = np.array([1.2, -0.8]), 0.1
h = np.maximum(0, x @ W1 + b1)
z = h @ v + c
p = 1 / (1 + np.exp(-z))
print("hidden:", h, "z:", round(z, 3), "p:", round(p, 3))  # [2.0, 0.2], 2.34, 0.912
print("gradient of the loss w.r.t. v1 (true class 1):", round((p - 1) * h[0], 4))  # -0.1757

X, y = make_circles(n_samples=800, noise=0.1, factor=0.5, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
scaler = StandardScaler().fit(X_train)

# @en Defaults: hidden_layer_sizes=(100,), activation='relu', learning_rate_init=0.001, alpha=0.0001, max_iter=200.
# @ru По умолчанию: hidden_layer_sizes=(100,), activation='relu', learning_rate_init=0.001, alpha=0.0001, max_iter=200.
model = MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=500, random_state=0)
model.fit(scaler.transform(X_train), y_train)
print("test accuracy:", round(accuracy_score(y_test, model.predict(scaler.transform(X_test))), 3))
print("epochs:", model.n_iter_, "final training loss:", round(model.loss_, 4))

# @en The forward pass by hand with the trained coefs_ and intercepts_ equals predict_proba.
# @ru Прямой проход вручную с обученными coefs_ и intercepts_ совпадает с predict_proba.
a = scaler.transform(X_test[:1])
for W, b in zip(model.coefs_[:-1], model.intercepts_[:-1]):
    a = np.maximum(0, a @ W + b)
out = 1 / (1 + np.exp(-(a @ model.coefs_[-1] + model.intercepts_[-1])))
print("by hand:", out.ravel().round(4), " sklearn P(class 1):", model.predict_proba(scaler.transform(X_test[:1]))[:, 1].round(4))
