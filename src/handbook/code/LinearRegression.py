# @en Linear regression: fit, evaluate on a held-out test set, predict.
# @ru Линейная регрессия: обучение, проверка на отложенном тесте, предсказание.
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# @en The five points from the worked example: the line should be y = 0.8x + 1.8.
# @ru Пять точек из примера: прямая должна получиться y = 0.8x + 1.8.
x = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 6])
small = LinearRegression().fit(x, y)
print("w =", round(small.coef_[0], 3), " b =", round(small.intercept_, 3))  # 0.8, 1.8

# @en A bigger problem: 300 rows, 3 features, some noise.
# @ru Задача побольше: 300 строк, 3 признака, немного шума.
X, y = make_regression(n_samples=300, n_features=3, noise=10.0, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# @en Defaults: fit_intercept=True, positive=False.
# @ru По умолчанию: fit_intercept=True, positive=False.
model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("weights:", model.coef_.round(2), "intercept:", round(model.intercept_, 2))
print("test R2:", round(r2_score(y_test, pred), 3))
print("test MSE:", round(mean_squared_error(y_test, pred), 1))

# @en Predict for a new row.
# @ru Предсказание для новой строки.
print("prediction:", model.predict([[0.5, -1.0, 2.0]]).round(2))
