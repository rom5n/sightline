# @en Training, loss, overfitting and the train/test split in one small script.
# @ru Обучение, функция потерь, переобучение и разбиение на обучение/тест в одном скрипте.
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# @en 1. Loss by hand: the worked example from this chapter.
# @ru 1. Функция потерь вручную: пример из этой главы.
y_true = [1, 0, 1, 1]
p_class1 = [0.9, 0.2, 0.6, 0.3]
print("log-loss:", round(log_loss(y_true, p_class1), 3))  # 0.511
print("MSE:", round(mean_squared_error([3, 5, 7], [2.5, 5, 8]), 3))  # 0.417

# @en 2. Data: 400 noisy points in two interlocking moons.
# @ru 2. Данные: 400 зашумлённых точек в виде двух «лун».
X, y = make_moons(n_samples=400, noise=0.35, random_state=0)

# @en 3. Hold out 25 % for testing; keep the class shares (stratify) and fix the shuffle (random_state).
# @ru 3. Откладываем 25 % на тест, сохраняем доли классов (stratify) и фиксируем перемешивание (random_state).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# @en 4. Train two trees: unlimited depth (memorises) and depth 3 (generalises).
# @ru 4. Обучаем два дерева: без ограничения глубины (запоминает) и глубины 3 (обобщает).
for depth in (None, 3):
    model = DecisionTreeClassifier(max_depth=depth, random_state=0)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"max_depth={depth}: train {train_acc:.2f}, test {test_acc:.2f}")
# @en Expected: max_depth=None -> train 1.00, test 0.82 (overfitting); max_depth=3 -> train 0.89, test 0.88.
# @ru Ожидаем: max_depth=None -> обучение 1.00, тест 0.82 (переобучение); max_depth=3 -> обучение 0.89, тест 0.88.

# @en 5. Predict a new point.
# @ru 5. Предсказание для новой точки.
new_point = np.array([[0.5, 0.25]])
print("class:", model.predict(new_point)[0], "probabilities:", model.predict_proba(new_point).round(2))
