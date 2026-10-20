"""Other classifiers: SVC, KNeighborsClassifier, GaussianNB, MLPClassifier."""
from __future__ import annotations

import threading
import time
import warnings
from typing import Any, Callable, ClassVar, Iterator

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from . import common_params as cp
from .base import Adapter, Grid, P, StepInfo, full_proba, proba_viz, scores_to_proba


def _true(prep: Any, row: int) -> int | None:
    return int(prep.y[row]) if prep.y is not None else None


class SVCA(Adapter):
    name = "SVC"
    family = "other"
    group = "other"
    task = "classification"
    est = SVC
    loss_kind = "onestep"
    step_unit = "none"
    explain_family = "svc"
    expensive = True
    main = [
        P("C", "float", 1.0, ("Regularization", "Регуляризация"),
          ("Small C allows a wider margin with some mistakes. Big C squeezes the margin to get every point right.",
           "Маленький C допускает ошибки ради широкого зазора. Большой C сужает зазор, чтобы угадать каждую точку."),
          lo=0.01, hi=100, log=True),
        P("kernel", "enum", "rbf", ("Kernel", "Ядро"),
          ("How 'similar' is measured. 'linear' draws straight borders, 'rbf' can bend around groups.",
           "Как измеряется «похожесть». «linear» даёт прямые границы, «rbf» умеет огибать группы."),
          options=["rbf", "linear", "poly", "sigmoid"]),
        P("gamma", "mixed", "scale", ("Kernel width", "Ширина ядра"),
          ("How far one point's influence reaches. Bigger gamma → tighter, wigglier borders.",
           "Насколько далеко дотягивается влияние точки. Больше gamma — граница плотнее и извилистее."),
          options=["scale", "auto"], lo=0.001, hi=100, log=True, enabled_if={"kernel": ["rbf", "poly", "sigmoid"]},
          only_tag=("not for 'linear'", "не для «linear»")),
        P("degree", "int", 3, ("Polynomial degree", "Степень многочлена"),
          ("Degree of the polynomial kernel. Higher → more curved borders.", "Степень полиномиального ядра. Выше — более изогнутые границы."),
          lo=1, hi=8, step=1, visible_if={"kernel": ["poly"]}, enabled_if={"kernel": ["poly"]}),
    ]
    advanced = [
        P("coef0", "float", 0.0, ("Kernel offset", "Сдвиг ядра"),
          ("Constant term inside the poly and sigmoid kernels.", "Свободный член внутри ядер poly и sigmoid."),
          lo=-5.0, hi=5.0, step=0.1, section="solver", enabled_if={"kernel": ["poly", "sigmoid"]},
          only_tag=("poly / sigmoid only", "только poly / sigmoid")),
        P("shrinking", "bool", True, ("Shrinking heuristic", "Эвристика сжатия"),
          ("Skips points that clearly won't become support vectors. Faster, same result.",
           "Пропускает точки, которые точно не станут опорными. Быстрее, результат тот же."), section="solver"),
        cp.tol(1e-3),
        P("max_iter", "int", -1, ("Iteration limit", "Предел итераций"),
          ("Hard limit on solver iterations. -1 = no limit.", "Жёсткий предел итераций. -1 — без предела."),
          lo=-1, hi=100000, step=1, section="stop"),
        P("cache_size", "float", 200, ("Kernel cache, MB", "Кэш ядра, МБ"),
          ("Memory for kernel values. Only affects speed.", "Память для значений ядра. Влияет только на скорость."),
          lo=50, hi=2000, step=50, section="other"),
        cp.class_weight("other"),
        P("decision_function_shape", "enum", "ovr", ("Multiclass scores", "Оценки для многих классов"),
          ("Shape of the scores for more than two classes: one-vs-rest or one-vs-one.",
           "Вид оценок для более чем двух классов: «один против всех» или «один против одного»."),
          options=["ovr", "ovo"], section="other"),
        P("break_ties", "bool", False, ("Break ties", "Разрешать ничьи"),
          ("For 3+ classes, settle votes that end in a tie by the scores.", "Для 3+ классов разрешать ничьи в голосовании по оценкам."),
          section="other", enabled_if={"decision_function_shape": ["ovr"]}, only_tag=("ovr only", "только ovr")),
        P("probability", "fixed", "deprecated", ("Probability estimates", "Оценки вероятностей"),
          ("Deprecated in this scikit-learn version. Shading shows the squashed decision value instead.",
           "Устарел в этой версии scikit-learn. Заливка показывает сжатое значение решающей функции."),
          section="other", disabled=("Deprecated in scikit-learn 1.9", "Устарел в scikit-learn 1.9")),
        cp.random_state(),
    ]

    def proba(self, X: np.ndarray) -> np.ndarray:
        return scores_to_proba(self.model.decision_function(X), len(self.prep.classes))

    def fit_stepwise(self) -> Iterator[StepInfo]:
        p = self.prep
        n = len(p.y_train)
        est_s = 0.02
        if n > 600:
            sub = np.random.default_rng(0).choice(n, 300, replace=False)
            t = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.build().fit(p.X_train[sub], p.y_train[sub])
            est_s = max(0.02, (time.perf_counter() - t) * (n / 300) ** 2)
        yield StepInfo(0, 1, blocking=0.0, extra={"estimate_s": est_s})
        box: dict[str, Any] = {}
        done = threading.Event()

        def run() -> None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    box["m"] = self.build().fit(p.X_train, p.y_train)
            except BaseException as e:  # noqa: BLE001 - re-raised below in the worker thread
                box["e"] = e
            finally:
                done.set()

        t0 = time.perf_counter()
        th = threading.Thread(target=run, daemon=True)
        th.start()
        while not done.wait(0.05):
            el = time.perf_counter() - t0
            self.ctx.progress(min(0.95, el / est_s))
            if self.ctx.cancelled():
                return
        if "e" in box:
            raise box["e"]
        self.model = box["m"]
        self.t_fit = (time.perf_counter() - t0) * 1000
        tr, te = self.acc_pair()
        yield StepInfo(1, 1, None, None, tr, te, extra={"n_support": self.model.n_support_.tolist()}, done="fitted")

    def visualize(self, grid: Grid) -> dict[str, Any]:
        v = proba_viz(self.prep, grid, self.proba, score=True, chunk=4000, progress=self.ctx.progress)
        v["support"] = self.prep.ids[self.prep.train_idx[self.model.support_]].astype(np.int32)
        return v

    def summary(self) -> dict[str, Any]:
        return {"n_support": self.model.n_support_.tolist(), "total": int(self.model.support_.size),
                "n_train": int(len(self.prep.train_idx)), "fit_ms": self.t_fit,
                "support": self.prep.ids[self.prep.train_idx[self.model.support_]].astype(np.int32)}

    def _gamma(self) -> float:
        g = getattr(self.model, "_gamma", None)
        if g is not None:
            return float(g)
        X = self.prep.X_train
        gv = self.value("gamma")
        if gv == "scale":
            return float(1.0 / (X.shape[1] * X.var()))
        if gv == "auto":
            return float(1.0 / X.shape[1])
        return float(gv)

    def explain(self, row: int) -> dict[str, Any]:
        p = self.prep
        x = p.Xm[row:row + 1]
        m = self.model
        dec = np.atleast_1d(np.ravel(m.decision_function(x)))
        pred = int(m.predict(x)[0])
        sv_ids = p.ids[p.train_idx[m.support_]]
        out: dict[str, Any] = {"decision": dec.tolist(), "pred": pred, "true": _true(p, row), "classes": p.classes,
                               "n_sv": int(m.support_.size), "n_train": int(len(p.train_idx)), "kernel": m.kernel,
                               "binary": len(p.classes) == 2}
        if len(p.classes) == 2:
            kw: dict[str, Any] = {"gamma": self._gamma()} if m.kernel != "linear" else {}
            if m.kernel in ("poly",):
                kw.update(degree=m.degree, coef0=m.coef0)
            if m.kernel == "sigmoid":
                kw.update(coef0=m.coef0)
            K = pairwise_kernels(x, m.support_vectors_, metric=m.kernel, **kw)[0]
            contrib = m.dual_coef_[0] * K
            order = np.argsort(-np.abs(contrib))[:8]
            out.update({
                "intercept": float(m.intercept_[0]), "sum": float(contrib.sum()),
                "top": [{"id": int(sv_ids[i]), "alpha": float(m.dual_coef_[0][i]), "k": float(K[i]), "contrib": float(contrib[i]),
                         "xy": [float(v) for v in p.model_to_plane(m.support_vectors_[i:i + 1])[0]]} for i in order],
                "rest": float(contrib.sum() - contrib[order].sum()),
            })
        return out


class KNeighborsA(Adapter):
    name = "KNeighborsClassifier"
    family = "other"
    group = "other"
    task = "classification"
    est = KNeighborsClassifier
    loss_kind = "knn"
    step_unit = "none"
    explain_family = "knn"
    main = [
        P("n_neighbors", "int", 5, ("Neighbours (k)", "Соседей (k)"),
          ("How many nearby points vote. Small k follows single points; big k smooths the border.",
           "Сколько ближайших точек голосует. Маленькое k следует за отдельными точками, большое — сглаживает границу."),
          lo=1, hi=50, step=1),
        P("weights", "enum", "uniform", ("Vote weights", "Вес голосов"),
          ("'uniform': every neighbour counts the same. 'distance': closer neighbours count more.",
           "«uniform»: все соседи равны. «distance»: ближние соседи весят больше."), options=["uniform", "distance"]),
        P("metric", "enum", "minkowski", ("Distance", "Расстояние"),
          ("How distance is measured. minkowski with p = 2 is the usual straight-line distance.",
           "Как измеряется расстояние. minkowski при p = 2 — обычное расстояние по прямой."),
          options=["minkowski", "euclidean", "manhattan", "chebyshev"]),
        P("p", "float", 2, ("Minkowski power", "Степень Минковского"),
          ("p = 1 → city-block distance, p = 2 → straight line.", "p = 1 — «городское» расстояние, p = 2 — по прямой."),
          lo=1, hi=5, step=0.5, enabled_if={"metric": ["minkowski"]}, only_tag=("metric='minkowski' only", "только metric='minkowski'")),
    ]
    advanced = [
        P("algorithm", "enum", "auto", ("Search method", "Метод поиска"),
          ("How neighbours are found. Only affects speed.", "Как ищутся соседи. Влияет только на скорость."),
          options=["auto", "ball_tree", "kd_tree", "brute"], section="solver"),
        P("leaf_size", "int", 30, ("Leaf size", "Размер листа"),
          ("Tuning for tree search. Only affects speed.", "Настройка поиска по дереву. Влияет только на скорость."),
          lo=5, hi=200, step=1, section="solver", enabled_if={"algorithm": ["auto", "ball_tree", "kd_tree"]}),
        P("metric_params", "fixed", None, ("Metric options", "Параметры метрики"),
          ("Extra arguments for the distance function.", "Дополнительные аргументы функции расстояния."),
          section="solver", disabled=("Not needed for the metrics offered here", "Не нужны для предложенных метрик")),
    ]

    def proba(self, X: np.ndarray) -> np.ndarray:
        return full_proba(self.model, X, len(self.prep.classes))

    def fit_stepwise(self) -> Iterator[StepInfo]:
        p = self.prep
        t = time.perf_counter()
        self.model = self.build().fit(p.X_train, p.y_train)
        self.t_fit = (time.perf_counter() - t) * 1000
        kmax = int(min(30, len(p.y_train)))
        curve = []
        for k in range(1, kmax + 1):
            m = self.build(n_neighbors=k).fit(p.X_train, p.y_train)
            te = float(accuracy_score(p.y_test, m.predict(p.X_test))) if len(p.test_idx) else None
            curve.append({"k": k, "test": te})
            self.ctx.progress(k / kmax)
        tr, te = self.acc_pair()
        best = max((c for c in curve if c["test"] is not None), key=lambda c: c["test"] or 0, default=None)
        self.curve = curve
        yield StepInfo(1, 1, None, None, tr, te, extra={"curve": curve, "best_k": best["k"] if best else None,
                                                         "k": int(self.value("n_neighbors"))}, done="fitted")

    def visualize(self, grid: Grid) -> dict[str, Any]:
        return proba_viz(self.prep, grid, self.proba, progress=self.ctx.progress)

    def summary(self) -> dict[str, Any]:
        return {"curve": self.curve, "k": int(self.value("n_neighbors"))}

    def explain(self, row: int) -> dict[str, Any]:
        p = self.prep
        x = p.Xm[row:row + 1]
        dist, idx = self.model.kneighbors(x)
        dist, idx = dist[0], idx[0]
        labels = p.y_train[idx]
        k = len(p.classes)
        if self.value("weights") == "distance":
            with np.errstate(divide="ignore"):
                w = 1.0 / dist
            if np.isinf(w).any():
                w = np.isinf(w).astype(float)
        else:
            w = np.ones_like(dist)
        votes = np.zeros(k)
        for lab, wi in zip(labels, w):
            votes[int(lab)] += wi
        proba = self.proba(x)[0]
        pred = int(self.model.predict(x)[0])
        rows = p.train_idx[idx]
        return {"neighbors": [{"id": int(p.ids[r]), "dist": float(d), "cls": int(c), "w": float(wi),
                               "xy": [float(p.xy[r, 0]), float(p.xy[r, 1])]} for r, d, c, wi in zip(rows, dist, labels, w)],
                "votes": votes.tolist(), "proba": proba.tolist(), "pred": pred, "true": _true(p, row), "classes": p.classes,
                "k": int(len(idx)), "weights": self.value("weights"), "point": [float(p.xy[row, 0]), float(p.xy[row, 1])],
                "is_train": bool(row in set(p.train_idx.tolist()))}


class GaussianNBA(Adapter):
    name = "GaussianNB"
    family = "other"
    group = "other"
    task = "classification"
    est = GaussianNB
    loss_kind = "onestep"
    step_unit = "none"
    explain_family = "nb"
    main = [
        P("var_smoothing", "float", 1e-9, ("Variance smoothing", "Сглаживание дисперсии"),
          ("Adds a little spread to every bell curve so none is too narrow. Raise it if borders look jumpy.",
           "Добавляет немного ширины каждой колоколообразной кривой, чтобы ни одна не была слишком узкой."),
          lo=1e-12, hi=1.0, log=True),
        P("priors", "priors", None, ("Class shares (priors)", "Доли классов (априорные)"),
          ("How common each class is before looking at the features. None = learn from the data.",
           "Насколько часто встречается каждый класс до взгляда на признаки. None — взять из данных."),
          none_label=("From the data (None)", "Из данных (None)")),
    ]
    advanced: ClassVar[list[P]] = []

    def proba(self, X: np.ndarray) -> np.ndarray:
        return full_proba(self.model, X, len(self.prep.classes))

    def sk_params(self, **override: Any) -> dict[str, Any]:
        out = super().sk_params(**override)
        pr = out.get("priors")
        k = len(self.prep.classes)
        if pr is not None:
            a = np.clip(np.asarray(pr, dtype=float)[:k], 1e-6, None)
            if len(a) != k:
                a = np.ones(k)
            out["priors"] = (a / a.sum()).tolist()
        return out

    def fit_stepwise(self) -> Iterator[StepInfo]:
        p = self.prep
        t = time.perf_counter()
        self.model = self.build().fit(p.X_train, p.y_train)
        self.t_fit = (time.perf_counter() - t) * 1000
        tr, te = self.acc_pair()
        yield StepInfo(1, 1, None, None, tr, te, done="fitted")

    def visualize(self, grid: Grid) -> dict[str, Any]:
        return proba_viz(self.prep, grid, self.proba)

    def summary(self) -> dict[str, Any]:
        m = self.model
        return {"priors": m.class_prior_.tolist(), "means": m.theta_.tolist(), "vars": m.var_.tolist(),
                "features": self.prep.feature_names, "fit_ms": self.t_fit}

    def explain(self, row: int) -> dict[str, Any]:
        p = self.prep
        m = self.model
        x = p.Xm[row]
        theta, var = m.theta_, m.var_
        lik = np.exp(-0.5 * (x - theta) ** 2 / var) / np.sqrt(2 * np.pi * var)  # (k, d)
        logj = np.log(m.class_prior_) + np.sum(-0.5 * np.log(2 * np.pi * var) - 0.5 * (x - theta) ** 2 / var, axis=1)
        post = np.exp(logj - logj.max())
        post /= post.sum()
        proba = self.proba(p.Xm[row:row + 1])[0]
        pred = int(m.predict(p.Xm[row:row + 1])[0])
        return {"priors": m.class_prior_.tolist(), "features": p.feature_names, "x": x.tolist(),
                "means": theta.tolist(), "stds": np.sqrt(var).tolist(), "lik": lik.tolist(),
                "joint_log": logj.tolist(), "joint": np.exp(logj).tolist(), "posterior": post.tolist(), "proba": proba.tolist(),
                "pred": pred, "true": _true(p, row), "classes": p.classes}


ACT: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "relu": lambda z: np.maximum(z, 0),
    "tanh": np.tanh,
    "logistic": lambda z: 1 / (1 + np.exp(-z)),
    "identity": lambda z: z,
}


class MLPA(Adapter):
    name = "MLPClassifier"
    family = "other"
    group = "other"
    task = "classification"
    est = MLPClassifier
    loss_kind = "loss"
    iterative = True
    step_unit = "epoch"
    lock_while_training = True
    managed = {"warm_start"}
    explain_family = "mlp"
    main = [
        P("hidden_layer_sizes", "layers", (100,), ("Hidden layers", "Скрытые слои"),
          ("Neurons per hidden layer. Layers over 16 neurons are drawn compressed.",
           "Число нейронов в каждом скрытом слое. Слои больше 16 нейронов рисуются сжато."), lo=1, hi=256),
        P("activation", "enum", "relu", ("Activation", "Функция активации"),
          ("How a neuron turns its input sum into an output. relu bends; identity can only draw straight lines.",
           "Как нейрон превращает сумму входов в выход. relu изгибает; identity умеет только прямые линии."),
          options=["relu", "tanh", "logistic", "identity"]),
        P("learning_rate_init", "float", 0.001, ("Learning rate", "Скорость обучения"),
          ("Step size for each weight update. Too big → loss jumps around. Too small → very slow.",
           "Размер шага при обновлении весов. Слишком большой — ошибка скачет. Слишком маленький — очень медленно."),
          lo=1e-5, hi=1.0, log=True),
        P("alpha", "float", 0.0001, ("L2 penalty", "Штраф L2"),
          ("Keeps weights small. Raise it if the border looks wiggly around single points.",
           "Держит веса маленькими. Увеличьте, если граница извивается вокруг отдельных точек."),
          lo=1e-6, hi=10.0, log=True),
        cp.max_iter(200, lo=10, hi=1000, label=("Max epochs", "Макс. эпох"),
                    hint=("How many passes over the data, at most.", "Максимальное число проходов по данным.")),
    ]
    advanced = [
        P("solver", "enum", "adam", ("Solver", "Метод оптимизации"),
          ("The method that updates the weights. lbfgs suits small datasets but has no per-epoch loss curve.",
           "Метод обновления весов. lbfgs подходит для маленьких данных, но у него нет кривой ошибки по эпохам."),
          options=["adam", "sgd", "lbfgs"], section="optimizer"),
        P("batch_size", "mixed", "auto", ("Batch size", "Размер пакета"),
          ("Points per weight update. 'auto' = min(200, number of rows).", "Точек на одно обновление весов. «auto» = min(200, число строк)."),
          options=["auto"], lo=1, hi=1000, step=1, section="optimizer", enabled_if={"solver": ["adam", "sgd"]}),
        P("beta_1", "float", 0.9, ("Adam β1", "Adam β1"),
          ("How much the last updates keep pushing in the same direction. Default 0.9.",
           "Насколько последние обновления продолжают толкать в ту же сторону. По умолчанию 0.9."),
          lo=0.0, hi=0.999, step=0.01, section="optimizer", enabled_if={"solver": ["adam"]}, only_tag=("solver='adam' only", "только solver='adam'")),
        P("beta_2", "float", 0.999, ("Adam β2", "Adam β2"),
          ("Memory for how big recent updates were. Leave as is.", "Память о размере недавних обновлений. Лучше не трогать."),
          lo=0.0, hi=0.9999, step=0.0001, section="optimizer", enabled_if={"solver": ["adam"]}, only_tag=("solver='adam' only", "только solver='adam'")),
        P("epsilon", "float", 1e-8, ("Adam ε", "Adam ε"),
          ("Tiny number that prevents dividing by zero.", "Крошечное число, чтобы не делить на ноль."),
          lo=1e-12, hi=1e-3, log=True, section="optimizer", enabled_if={"solver": ["adam"]}, only_tag=("solver='adam' only", "только solver='adam'")),
        P("learning_rate", "enum", "constant", ("Learning-rate schedule", "График скорости обучения"),
          ("How the step size changes over time.", "Как меняется размер шага со временем."),
          options=["constant", "invscaling", "adaptive"], section="optimizer", enabled_if={"solver": ["sgd"]},
          only_tag=("solver='sgd' only", "только solver='sgd'")),
        P("momentum", "float", 0.9, ("Momentum", "Инерция"),
          ("Keeps moving in the previous direction, like a rolling ball.", "Продолжает двигаться в прежнем направлении, как катящийся шар."),
          lo=0.0, hi=0.99, step=0.01, section="optimizer", enabled_if={"solver": ["sgd"]}, only_tag=("solver='sgd' only", "только solver='sgd'")),
        P("nesterovs_momentum", "bool", True, ("Nesterov momentum", "Инерция Нестерова"),
          ("A slightly smarter momentum that looks one step ahead.", "Немного более умная инерция, которая заглядывает на шаг вперёд."),
          section="optimizer", enabled_if={"solver": ["sgd"]}, only_tag=("solver='sgd' only", "только solver='sgd'")),
        P("power_t", "float", 0.5, ("Decay power", "Степень затухания"),
          ("How fast the step size shrinks.", "Как быстро уменьшается шаг."),
          lo=0.0, hi=2.0, step=0.05, section="optimizer", enabled_if={"solver": ["sgd"], "learning_rate": ["invscaling"]},
          only_tag=("sgd + invscaling only", "только sgd + invscaling")),
        P("early_stopping", "bool", False, ("Early stopping", "Ранняя остановка"),
          ("Hold back part of the training data and stop when its score stops improving.",
           "Отложить часть обучающих данных и остановиться, когда оценка на них перестанет расти."), section="stop"),
        P("validation_fraction", "float", 0.1, ("Validation share", "Доля для проверки"),
          ("Share of training rows held back for early stopping.", "Доля обучающих строк, отложенных для ранней остановки."),
          lo=0.05, hi=0.5, step=0.05, section="stop", enabled_if={"early_stopping": [True]}, only_tag=("early_stopping only", "только при early_stopping")),
        P("n_iter_no_change", "int", 10, ("Patience", "Терпение"),
          ("Epochs without improvement before it stops.", "Сколько эпох без улучшения ждать перед остановкой."), lo=1, hi=100, step=1, section="stop"),
        cp.tol(1e-4),
        P("max_fun", "int", 15000, ("Max function calls", "Макс. вызовов функции"),
          ("Upper limit on loss evaluations.", "Верхний предел числа вычислений ошибки."),
          lo=100, hi=100000, step=100, section="stop", enabled_if={"solver": ["lbfgs"]}, only_tag=("solver='lbfgs' only", "только solver='lbfgs'")),
        cp.random_state(),
        P("shuffle", "bool", True, ("Shuffle each epoch", "Перемешивать каждую эпоху"),
          ("Visit the points in a new random order every epoch.", "Обходить точки в новом случайном порядке каждую эпоху."),
          section="repro", enabled_if={"solver": ["adam", "sgd"]}),
        cp.warm_start(),
    ]

    def __init__(self, ctx: Any) -> None:
        super().__init__(ctx)
        self.probe_row: int | None = None

    def proba(self, X: np.ndarray) -> np.ndarray:
        return full_proba(self.model, X, len(self.prep.classes))

    def fit_stepwise(self) -> Iterator[StepInfo]:
        p = self.prep
        k = len(p.classes)
        total = int(self.value("max_iter"))
        solver = self.value("solver")
        tol = float(self.value("tol"))
        patience = int(self.value("n_iter_no_change"))
        early = bool(self.value("early_stopping"))
        Xf, yf = p.X_train, p.y_train
        Xv = yv = None
        if early and len(yf) >= 20:
            Xf, Xv, yf, yv = train_test_split(Xf, yf, test_size=float(self.value("validation_fraction")),
                                              random_state=self.ctx.seed, stratify=yf)
        classes = np.arange(k)
        if solver == "lbfgs":
            model = self.build(max_iter=1, warm_start=True, early_stopping=False)
        else:
            model = self.build(early_stopping=False)
        best = np.inf
        bad = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for ep in range(1, total + 1):
                if solver == "lbfgs":
                    model.fit(Xf, yf)
                else:
                    model.partial_fit(Xf, yf, classes=classes)
                self.model = model
                loss = float(model.loss_)
                loss_te = float(log_loss(p.y_test, np.clip(self.proba(p.X_test), 1e-15, 1), labels=list(classes))) if len(p.test_idx) else None
                tr, te = self.acc_pair()
                score = -float(np.mean(model.predict(Xv) == yv)) if Xv is not None else loss
                if score > best - tol:
                    bad += 1
                else:
                    bad = 0
                best = min(best, score)
                stop = bad >= patience or (solver == "lbfgs" and int(model.n_iter_) == 0)
                done = ("early_stop" if Xv is not None else "converged") if stop else ("max_iter" if ep == total else None)
                yield StepInfo(ep, total, loss, loss_te, tr, te, done=done)
                if done:
                    return

    def network(self) -> dict[str, Any]:
        m = self.model
        return {"sizes": [m.coefs_[0].shape[0]] + [w.shape[1] for w in m.coefs_],
                "coefs": [w.astype(np.float32) for w in m.coefs_],
                "intercepts": [b.astype(np.float32) for b in m.intercepts_],
                "activation": m.activation, "out_activation": m.out_activation_}

    def forward(self, x: np.ndarray) -> list[dict[str, Any]]:
        m = self.model
        f = ACT[m.activation]
        a = x
        layers = [{"act": a.tolist()}]
        n = len(m.coefs_)
        for i, (W, b) in enumerate(zip(m.coefs_, m.intercepts_)):
            z = a @ W + b
            if i == n - 1:
                if m.out_activation_ == "softmax":
                    e = np.exp(z - z.max())
                    a = e / e.sum()
                else:
                    a = ACT[m.out_activation_](z)
            else:
                a = f(z)
            layers.append({"pre": z.tolist(), "act": a.tolist()})
        return layers

    def probe(self, row: int) -> dict[str, Any] | None:
        if self.model is None or not hasattr(self.model, "coefs_"):
            return None
        return {"row_id": int(self.prep.ids[row]), "layers": self.forward(self.prep.Xm[row])}

    def visualize(self, grid: Grid) -> dict[str, Any]:
        v = proba_viz(self.prep, grid, self.proba)
        v["network"] = self.network()
        if self.probe_row is not None:
            v["probe"] = self.probe(self.probe_row)
        return v

    def explain(self, row: int) -> dict[str, Any]:
        p = self.prep
        layers = self.forward(p.Xm[row])
        proba = self.proba(p.Xm[row:row + 1])[0]
        pred = int(self.model.predict(p.Xm[row:row + 1])[0])
        return {"layers": layers, "network": self.network(), "proba": proba.tolist(), "pred": pred,
                "true": _true(p, row), "classes": p.classes, "features": p.feature_names,
                "activation": self.model.activation, "out_activation": self.model.out_activation_}


ADAPTERS: list[type[Adapter]] = [SVCA, KNeighborsA, GaussianNBA, MLPA]
