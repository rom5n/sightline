# @en Extra trees: like a random forest, but every split threshold is drawn at random.
# @ru Сверхслучайные деревья: как случайный лес, но каждый порог разбиения выбирается случайно.
import time

from sklearn.datasets import make_moons
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=1500, noise=0.35, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# @en Defaults: n_estimators=100, max_depth=None, max_features='sqrt', min_samples_leaf=1, bootstrap=False.
# @ru По умолчанию: n_estimators=100, max_depth=None, max_features='sqrt', min_samples_leaf=1, bootstrap=False.
for model in (ExtraTreesClassifier(random_state=0), RandomForestClassifier(random_state=0)):
    t = time.perf_counter()
    model.fit(X_train, y_train)
    ms = (time.perf_counter() - t) * 1000
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{type(model).__name__}: test accuracy {acc:.3f}, fit {ms:.0f} ms")

# @en Bigger leaves smooth the border further.
# @ru Более крупные листья ещё сильнее сглаживают границу.
for leaf in (1, 5, 20):
    m = ExtraTreesClassifier(min_samples_leaf=leaf, random_state=0).fit(X_train, y_train)
    print(f"min_samples_leaf={leaf}: test accuracy {accuracy_score(y_test, m.predict(X_test)):.3f}")
print("P(class) for one row:", m.predict_proba(X_test[:1]).round(3))
