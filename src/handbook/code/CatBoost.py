# @en CatBoost: boosting on symmetric trees that reads categorical columns directly (cat_features).
# @ru CatBoost: бустинг на симметричных деревьях, который читает категориальные столбцы напрямую (cat_features).
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split


# @en Worked example: ordered target statistics, each row sees only the rows before it.
# @ru Пример: упорядоченные целевые статистики, каждая строка видит только строки до неё.
def ordered_ts(colors, labels, prior=0.5):
    out = []
    for i, c in enumerate(colors):
        seen = [labels[j] for j in range(i) if colors[j] == c]
        out.append((sum(seen) + prior) / (len(seen) + 1))
    return out


colors = ["red", "blue", "red", "red", "blue", "red"]
labels = [1, 0, 1, 0, 1, 1]
print("ordered TS:", [round(v, 3) for v in ordered_ts(colors, labels)])  # 0.5 0.5 0.75 0.833 0.25 0.625

# @en A table with a text column: the label depends on the colour and on x.
# @ru Таблица с текстовым столбцом: метка зависит от цвета и от x.
rng = np.random.default_rng(0)
n = 1500
df = pd.DataFrame({
    "x": rng.normal(size=n),
    "size": rng.uniform(0, 10, size=n),
    "color": rng.choice(["red", "green", "blue", "black"], size=n),
})
logit = 1.5 * df["x"] + df["color"].map({"red": 2.0, "green": -1.0, "blue": 0.5, "black": -2.0})
df["label"] = (logit + rng.normal(scale=0.7, size=n) > 0).astype(int)
X, y = df[["x", "size", "color"]], df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# @en No one-hot encoding: just name the categorical columns.
# @en Defaults: iterations=1000, learning_rate=auto, depth=6, l2_leaf_reg=3.
# @ru Никакого one-hot: достаточно назвать категориальные столбцы.
# @ru По умолчанию: iterations=1000, learning_rate=auto, depth=6, l2_leaf_reg=3.
model = CatBoostClassifier(iterations=300, random_seed=0, verbose=0, allow_writing_files=False)
model.fit(X_train, y_train, cat_features=["color"])
print("learning rate chosen:", round(model.learning_rate_, 4))
print("test accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 3))

# @en Loss per tree: CatBoost records the training loss; the test loss is computed afterwards,
# @en so the test rows never influence training.
# @ru Потери по деревьям: CatBoost записывает потери на обучении; потери на тесте считаются после,
# @ru поэтому тестовые строки никак не влияют на обучение.
train_loss = model.get_evals_result()["learn"]["Logloss"]
test_loss = model.eval_metrics(Pool(X_test, y_test, cat_features=["color"]), ["Logloss"])["Logloss"]
for k in (1, 50, 300):
    print(f"after {k:>3} trees: train log-loss {train_loss[k - 1]:.3f}, test log-loss {test_loss[k - 1]:.3f}")
print("importance:", {c: round(float(v), 1) for c, v in zip(X.columns, model.get_feature_importance())})

new = pd.DataFrame({"x": [0.3], "size": [5.0], "color": ["black"]})
print("P(class) for a new row:", model.predict_proba(new).round(3))

# @en The same for numbers: CatBoostRegressor (RMSE loss by default).
# @ru То же для чисел: CatBoostRegressor (по умолчанию функция потерь RMSE).
target = 3 * df["x"] + df["color"].map({"red": 4.0, "green": 0.0, "blue": 1.0, "black": -3.0}) + rng.normal(size=n)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X, target, test_size=0.25, random_state=42)
reg = CatBoostRegressor(iterations=300, random_seed=0, verbose=0, allow_writing_files=False)
reg.fit(Xr_train, yr_train, cat_features=["color"])
print("regressor test R2:", round(r2_score(yr_test, reg.predict(Xr_test)), 3))
