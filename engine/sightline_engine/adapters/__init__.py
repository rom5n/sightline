"""Model registry: one adapter per model, in Home-screen order."""
from __future__ import annotations

from .base import Adapter
from .catboost_models import ADAPTERS as _CB
from .ensembles import ADAPTERS as _ENS
from .linear import ADAPTERS as _LIN
from .other import ADAPTERS as _OTH
from .trees import ADAPTERS as _TREE
from .unsupervised import ADAPTERS as _UNS

_ORDER = [
    "LinearRegression", "LogisticRegression", "Ridge", "Lasso", "ElasticNet", "SGDClassifier",
    "DecisionTreeClassifier", "RandomForestClassifier", "ExtraTreesClassifier", "GradientBoostingClassifier",
    "HistGradientBoostingClassifier", "AdaBoostClassifier", "BaggingClassifier", "CatBoostClassifier", "CatBoostRegressor",
    "SVC", "KNeighborsClassifier", "GaussianNB", "MLPClassifier",
    "KMeans", "DBSCAN", "PCA",
]

_ALL = {a.name: a for a in _LIN + _TREE + _ENS + _CB + _OTH + _UNS}
REGISTRY: dict[str, type[Adapter]] = {name: _ALL[name] for name in _ORDER}

# per-model replay rate at 1x speed (steps per second)
RATE: dict[str, float] = {
    "LogisticRegression": 20, "SGDClassifier": 20, "MLPClassifier": 30, "Lasso": 30, "ElasticNet": 30,
    "KMeans": 2.5, "DecisionTreeClassifier": 1.5, "RandomForestClassifier": 8, "ExtraTreesClassifier": 8,
    "BaggingClassifier": 6, "GradientBoostingClassifier": 25, "HistGradientBoostingClassifier": 25,
    "AdaBoostClassifier": 15, "CatBoostClassifier": 25, "CatBoostRegressor": 25,
}


def schemas() -> list[dict[str, object]]:
    return [a.schema() for a in REGISTRY.values()]


__all__ = ["REGISTRY", "RATE", "schemas", "Adapter"]
