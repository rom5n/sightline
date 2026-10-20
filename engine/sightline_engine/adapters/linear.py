"""Linear family: LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, SGDClassifier."""
from __future__ import annotations

import warnings
from typing import Any, Iterator

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from . import common_params as cp
from .base import Adapter, Grid, P, StepInfo, finite, full_proba, proba_viz, scores_to_proba, sigmoid, softmax, value_viz


def _linear_terms(prep: Any, row: int, coef: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for j, name in enumerate(prep.feature_names):
        xm = float(prep.Xm[row, j])
        out.append({"name": name, "x": float(prep.X[row, j]), "xm": xm, "w": float(coef[j]), "contrib": float(coef[j] * xm)})
    return out


def _true_label(prep: Any, row: int) -> Any:
    if prep.y is None:
        return None
    return int(prep.y[row]) if prep.task == "classification" else float(prep.y[row])


class _Regressor(Adapter):
    group = "linear"
    family = "linear"
    task = "regression"
    explain_family = "linear_reg"

    def visualize(self, grid: Grid) -> dict[str, Any]:
        return value_viz(self.prep, grid, self.model.predict)

    def explain(self, row: int) -> dict[str, Any]:
        p = self.prep
        coef = np.ravel(self.model.coef_)
        b = float(np.ravel([self.model.intercept_])[0])
        terms = _linear_terms(p, row, coef)
        pred = float(self.model.predict(p.Xm[row:row + 1])[0])
        return {"terms": terms, "intercept": b, "prediction": pred, "true": _true_label(p, row),
                "target": p.target_name, "scaled": p.scaler is not None}

    def summary(self) -> dict[str, Any]:
        p = self.prep
        idx = p.test_idx if len(p.test_idx) >= 2 else p.train_idx
        assert p.y is not None
        res = p.y[idx] - self.model.predict(p.Xm[idx])
        lim = float(np.max(np.abs(res))) or 1.0
        hist, edges = np.histogram(res, bins=10, range=(-lim, lim))
        tr, te = self.r2_pair()
        return {"r2": te if te is not None else tr, "r2_train": tr, "residuals": {"counts": hist.tolist(), "lo": -lim, "hi": lim},
                "coef": np.ravel(self.model.coef_).tolist(), "intercept": float(np.ravel([self.model.intercept_])[0]),
                "features": p.feature_names, "fit_ms": self.t_fit}


class LinearRegressionA(_Regressor):
    name = "LinearRegression"
    est = LinearRegression
    loss_kind = "onestep_reg"
    step_unit = "none"
    main = [cp.fit_intercept(), cp.positive()]
    advanced = [cp.tol(1e-6), cp.copy("copy_X")]

    def fit_stepwise(self) -> Iterator[StepInfo]:
        import time
        t = time.perf_counter()
        self.model = self.build().fit(self.prep.X_train, self.prep.y_train)
        self.t_fit = (time.perf_counter() - t) * 1000
        tr, te = self.r2_pair()
        yield StepInfo(1, 1, None, None, tr, te, done="fitted")


class RidgeA(_Regressor):
    name = "Ridge"
    est = Ridge
    loss_kind = "onestep_reg"
    step_unit = "none"
    main = [
        P("alpha", "float", 1.0, ("Regularization strength", "Сила регуляризации"),
          ("Bigger alpha keeps the weights smaller, so the line reacts less to single points.",
           "Чем больше alpha, тем меньше веса и тем слабее линия реагирует на отдельные точки."),
          lo=0.001, hi=1000, log=True),
        cp.fit_intercept(),
    ]
    advanced = [
        P("solver", "enum", "auto", ("Solver", "Метод решения"),
          ("The maths used to find the weights. 'auto' picks one for you. 'lbfgs' needs positive=True.",
           "Каким способом считать веса. «auto» выберет сам. Для «lbfgs» нужно positive=True."),
          options=["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"], section="solver"),
        cp.positive("solver"),
        P("max_iter", "int", None, ("Max iterations", "Макс. итераций"),
          ("Only for iterative solvers. None = solver's own default.", "Только для итеративных методов. None — по умолчанию метода."),
          lo=1, hi=5000, step=1, none_label=("Solver default", "По умолчанию метода"), section="solver"),
        cp.tol(1e-4),
        cp.random_state(),
        cp.copy("copy_X"),
    ]

    def fit_stepwise(self) -> Iterator[StepInfo]:
        import time
        t = time.perf_counter()
        self.model = self.build().fit(self.prep.X_train, self.prep.y_train)
        self.t_fit = (time.perf_counter() - t) * 1000
        tr, te = self.r2_pair()
        yield StepInfo(1, 1, None, None, tr, te, done="fitted")


class _CoordDescent(_Regressor):
    loss_kind = "loss"
    iterative = True
    managed = {"warm_start"}

    def objective(self, X: np.ndarray, y: np.ndarray) -> float:
        a = float(self.value("alpha"))
        l1 = float(self.value("l1_ratio")) if self.name == "ElasticNet" else 1.0
        w = np.ravel(self.model.coef_)
        r = y - self.model.predict(X)
        return float(0.5 * np.mean(r * r) + a * l1 * np.abs(w).sum() + 0.5 * a * (1 - l1) * (w @ w))

    def fit_stepwise(self) -> Iterator[StepInfo]:
        p = self.prep
        total = int(self.value("max_iter"))
        tol = float(self.value("tol"))
        model = self.build(max_iter=1, warm_start=True)
        y = p.y_train
        yc = y - y.mean() if self.value("fit_intercept") else y
        gap_tol = tol * float(yc @ yc) / max(1, len(y))
        prev: np.ndarray | None = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for it in range(1, total + 1):
                model.fit(p.X_train, y)
                self.model = model
                w = np.ravel(model.coef_).copy()
                moved = float(np.max(np.abs(w - prev))) if prev is not None else float("inf")
                prev = w
                loss = self.objective(p.X_train, y)
                loss_te = self.objective(p.X_test, p.y_test) if len(p.test_idx) >= 2 else None
                tr, te = self.r2_pair()
                gap = float(getattr(model, "dual_gap_", np.inf))
                w_max = float(np.max(np.abs(w))) if w.size else 0.0
                # same test as sklearn's coordinate descent: weights settled, then duality gap small
                settled = w_max == 0.0 or moved / w_max < tol
                converged = settled and gap <= gap_tol
                done = "converged" if converged else ("max_iter" if it == total else None)
                yield StepInfo(it, total, loss, loss_te, tr, te,
                               extra={"nonzero": int(np.sum(np.abs(w) > 1e-12)), "coef": w.tolist()}, done=done)
                if done:
                    return


_ALPHA_L1 = P("alpha", "float", 1.0, ("Penalty strength", "Сила штрафа"),
              ("Bigger alpha pushes more weights to exactly 0, switching features off.",
               "Чем больше alpha, тем больше весов становятся ровно 0 — признаки выключаются."),
              lo=0.001, hi=100, log=True)


def _cd_advanced(warm: bool = True) -> list[P]:
    return [
        P("selection", "enum", "cyclic", ("Update order", "Порядок обновления"),
          ("'cyclic' updates weights one after another; 'random' picks them at random, often faster.",
           "«cyclic» обновляет веса по очереди; «random» — в случайном порядке, часто быстрее."),
          options=["cyclic", "random"], section="solver"),
        P("precompute", "bool", False, ("Precompute Gram matrix", "Предрасчёт матрицы Грама"),
          ("Speeds things up for many rows and few features.", "Ускоряет при большом числе строк и малом числе признаков."),
          section="solver"),
        cp.positive("solver"),
        cp.tol(1e-4),
        cp.random_state(),
        cp.warm_start(),
        cp.copy("copy_X"),
    ]


class LassoA(_CoordDescent):
    name = "Lasso"
    est = Lasso
    main = [_ALPHA_L1, cp.max_iter(1000, hi=5000, hint=("Passes of coordinate descent. Stops early when the weights settle.",
                                                       "Проходы покоординатного спуска. Остановится раньше, когда веса перестанут меняться.")),
            cp.fit_intercept()]
    advanced = _cd_advanced()


class ElasticNetA(_CoordDescent):
    name = "ElasticNet"
    est = ElasticNet
    main = [
        P("alpha", "float", 1.0, ("Penalty strength", "Сила штрафа"),
          ("Overall strength of the penalty on the weights.", "Общая сила штрафа на веса."), lo=0.001, hi=100, log=True),
        P("l1_ratio", "float", 0.5, ("L1 share", "Доля L1"),
          ("0 = only shrink weights (Ridge), 1 = drop features (Lasso). In between mixes both.",
           "0 — только уменьшать веса (Ridge), 1 — выключать признаки (Lasso). Между ними — смесь."),
          lo=0.0, hi=1.0, step=0.01),
        cp.max_iter(1000, hi=5000, hint=("Passes of coordinate descent. Stops early when the weights settle.",
                                         "Проходы покоординатного спуска. Остановится раньше, когда веса перестанут меняться.")),
        cp.fit_intercept(),
    ]
    advanced = _cd_advanced()


# ------------------------------------------------------------------ classifiers


class _LinearClassifier(Adapter):
    group = "linear"
    family = "linear"
    task = "classification"

    def proba(self, X: np.ndarray) -> np.ndarray:
        return full_proba(self.model, X, len(self.prep.classes))

    def visualize(self, grid: Grid) -> dict[str, Any]:
        v = proba_viz(self.prep, grid, self.proba)
        v["line"] = self._line()
        return v

    def _line(self) -> dict[str, Any] | None:
        """Border z = 0 in plane coords for binary raw-plane models (exact straight line)."""
        p = self.prep
        if p.plane.mode != "raw" or len(p.classes) != 2:
            return None
        w = np.ravel(self.model.coef_)
        b = float(np.ravel([self.model.intercept_])[0])
        if p.scaler is not None:
            s, m = p.scaler.scale_, p.scaler.mean_
            b = b - float(np.sum(w * m / s))
            w = w / s
        return {"w": [float(w[0]), float(w[1])], "b": b}


class LogisticRegressionA(_LinearClassifier):
    name = "LogisticRegression"
    est = LogisticRegression
    loss_kind = "loss"
    iterative = True
    managed = {"warm_start"}
    explain_family = "logistic"
    main = [
        P("C", "float", 1.0, ("Regularization", "Регуляризация"),
          ("Smaller C → simpler, smoother border. Bigger C → follows the points more tightly.",
           "Меньше C → граница проще и плавнее. Больше C → точнее подстраивается под точки."),
          lo=0.01, hi=100, log=True),
        cp.max_iter(100, lo=10, hi=1000),
        cp.fit_intercept(),
        cp.class_weight(),
    ]
    advanced = [
        P("solver", "enum", "lbfgs", ("Solver", "Метод оптимизации"),
          ("The method that searches for the weights. 'saga' is needed for an L1 share above 0.",
           "Метод, который ищет веса. Для доли L1 больше 0 нужен «saga»."),
          options=["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga", "liblinear"], section="solver"),
        P("l1_ratio", "float", 0.0, ("L1 share", "Доля L1"),
          ("0 = classic L2 penalty, 1 = L1 (can switch features off). Replaces the old penalty parameter.",
           "0 — обычный штраф L2, 1 — L1 (может выключать признаки). Заменяет устаревший параметр penalty."),
          lo=0.0, hi=1.0, step=0.05, section="solver", enabled_if={"solver": ["saga"]},
          only_tag=("solver='saga' only", "только solver='saga'")),
        cp.tol(1e-4),
        P("dual", "bool", False, ("Dual formulation", "Двойственная задача"),
          ("A different way to solve the same problem; only for liblinear with L2.",
           "Другой способ решить ту же задачу; только для liblinear с L2."),
          section="solver", enabled_if={"solver": ["liblinear"]}, only_tag=("solver='liblinear' only", "только solver='liblinear'")),
        P("intercept_scaling", "float", 1, ("Intercept scaling", "Масштаб свободного члена"),
          ("liblinear adds a fake feature for the intercept; this is its value.",
           "liblinear добавляет искусственный признак для свободного члена; это его значение."),
          lo=0.1, hi=100, log=True, section="solver", enabled_if={"solver": ["liblinear"]},
          only_tag=("solver='liblinear' only", "только solver='liblinear'")),
        cp.random_state(),
        cp.warm_start(),
    ]

    def fit_stepwise(self) -> Iterator[StepInfo]:
        p = self.prep
        total = int(self.value("max_iter"))
        warm = self.value("solver") != "liblinear"
        model = self.build(max_iter=1, warm_start=True) if warm else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            for it in range(1, total + 1):
                if warm:
                    assert model is not None
                    model.fit(p.X_train, p.y_train)
                else:
                    model = self.build(max_iter=it).fit(p.X_train, p.y_train)
                self.model = model
                loss, loss_te = self.logloss_pair(self.proba(p.X_train), self.proba(p.X_test) if len(p.test_idx) else None)
                tr, te = self.acc_pair()
                n_it = int(np.max(model.n_iter_))
                converged = (n_it == 0) if warm else (n_it < it)
                done = "converged" if converged else ("max_iter" if it == total else None)
                yield StepInfo(it, total, loss, loss_te, tr, te, done=done)
                if done:
                    return

    def explain(self, row: int) -> dict[str, Any]:
        p = self.prep
        x = p.Xm[row]
        k = len(p.classes)
        if k == 2:
            coef = np.ravel(self.model.coef_)
            b = float(self.model.intercept_[0])
            z = float(coef @ x + b)
            prob = float(sigmoid(np.array([z]))[0])
            pred = int(prob >= 0.5)
            return {"binary": True, "terms": _linear_terms(p, row, coef), "intercept": b, "z": z, "prob": prob,
                    "pred": pred, "true": _true_label(p, row), "classes": p.classes, "scaled": p.scaler is not None,
                    "proba": self.proba(p.Xm[row:row + 1])[0].tolist()}
        Z = self.model.coef_ @ x + self.model.intercept_
        pr = softmax(Z.reshape(1, -1))[0]
        per = [{"cls": int(c), "terms": _linear_terms(p, row, self.model.coef_[i]), "intercept": float(self.model.intercept_[i]),
                "z": float(Z[i])} for i, c in enumerate(self.model.classes_)]
        pred = int(self.model.classes_[int(np.argmax(pr))])
        return {"binary": False, "per_class": per, "proba": pr.tolist(), "pred": pred, "true": _true_label(p, row),
                "classes": p.classes, "scaled": p.scaler is not None}


SGD_LOSSES = ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron", "squared_error", "huber",
              "epsilon_insensitive", "squared_epsilon_insensitive"]


def sgd_loss_value(loss: str, scores: np.ndarray, y: np.ndarray, k: int, eps: float) -> float:
    S = scores.reshape(len(y), -1)
    if k == 2:
        Y = np.where(y == 1, 1.0, -1.0).reshape(-1, 1)
    else:
        Y = np.where(y.reshape(-1, 1) == np.arange(k).reshape(1, -1), 1.0, -1.0)
    m = Y * S
    r = Y - S
    if loss == "hinge":
        L = np.maximum(0, 1 - m)
    elif loss == "squared_hinge":
        L = np.maximum(0, 1 - m) ** 2
    elif loss == "perceptron":
        L = np.maximum(0, -m)
    elif loss == "log_loss":
        L = np.logaddexp(0, -m)
    elif loss == "modified_huber":
        L = np.where(m >= -1, np.maximum(0, 1 - m) ** 2, -4 * m)
    elif loss == "squared_error":
        L = 0.5 * r ** 2
    elif loss == "huber":
        a = np.abs(r)
        L = np.where(a <= eps, 0.5 * r ** 2, eps * a - 0.5 * eps ** 2)
    elif loss == "epsilon_insensitive":
        L = np.maximum(0, np.abs(r) - eps)
    else:
        L = np.maximum(0, np.abs(r) - eps) ** 2
    return float(L.mean())


class SGDClassifierA(_LinearClassifier):
    name = "SGDClassifier"
    est = SGDClassifier
    loss_kind = "loss"
    iterative = True
    step_unit = "epoch"
    managed = {"warm_start"}
    explain_family = "sgd"
    main = [
        P("loss", "enum", "hinge", ("Loss", "Функция потерь"),
          ("What counts as a mistake. 'hinge' = linear SVM, 'log_loss' = logistic regression.",
           "Что считается ошибкой. «hinge» — линейный SVM, «log_loss» — логистическая регрессия."),
          options=["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"]),
        P("penalty", "enum", "l2", ("Penalty", "Штраф"),
          ("How large weights are punished. 'l1' can switch features off; None = no penalty.",
           "Как штрафуются большие веса. «l1» может выключать признаки; None — без штрафа."),
          options=["l2", "l1", "elasticnet", None]),
        P("alpha", "float", 0.0001, ("Penalty strength", "Сила штрафа"),
          ("Bigger alpha → smaller weights and a calmer border.", "Больше alpha — меньше веса и спокойнее граница."),
          lo=1e-6, hi=1.0, log=True),
        cp.max_iter(1000, hi=2000, label=("Max epochs", "Макс. эпох"),
                    hint=("Passes over the data, at most. Usually stops much earlier.", "Максимум проходов по данным. Обычно останавливается гораздо раньше.")),
    ]
    advanced = [
        P("learning_rate", "enum", "optimal", ("Step-size schedule", "График шага"),
          ("How the step size changes over time. 'optimal' works well by default.",
           "Как меняется размер шага со временем. «optimal» хорошо работает по умолчанию."),
          options=["optimal", "constant", "invscaling", "adaptive", "pa1", "pa2"], section="solver"),
        P("eta0", "float", 0.01, ("Initial step size", "Начальный шаг"),
          ("Starting step size for 'constant', 'invscaling' and 'adaptive'.", "Начальный шаг для «constant», «invscaling» и «adaptive»."),
          lo=1e-5, hi=1.0, log=True, section="solver", enabled_if={"learning_rate": ["constant", "invscaling", "adaptive", "pa1", "pa2"]},
          only_tag=("not with 'optimal'", "не для «optimal»")),
        P("power_t", "float", 0.5, ("Decay power", "Степень затухания"),
          ("How fast the step size shrinks with 'invscaling'.", "Как быстро уменьшается шаг при «invscaling»."),
          lo=0.0, hi=2.0, step=0.05, section="solver", enabled_if={"learning_rate": ["invscaling"]},
          only_tag=("learning_rate='invscaling' only", "только learning_rate='invscaling'")),
        P("l1_ratio", "float", 0.15, ("L1 share", "Доля L1"),
          ("Mix of L1 and L2 for the elastic-net penalty.", "Смесь L1 и L2 для штрафа elasticnet."),
          lo=0.0, hi=1.0, step=0.05, section="solver", enabled_if={"penalty": ["elasticnet"]},
          only_tag=("penalty='elasticnet' only", "только penalty='elasticnet'")),
        P("epsilon", "float", 0.1, ("Epsilon", "Эпсилон"),
          ("Width of the 'no penalty' zone for huber and epsilon losses.", "Ширина зоны без штрафа для функций huber и epsilon."),
          lo=0.0, hi=2.0, step=0.01, section="solver", enabled_if={"loss": ["huber", "epsilon_insensitive", "squared_epsilon_insensitive"]},
          only_tag=("huber / epsilon losses only", "только huber / epsilon")),
        P("average", "bool", False, ("Average weights", "Усреднять веса"),
          ("Report the average of all weight versions seen; smoother result.", "Брать среднее всех версий весов; результат глаже."),
          section="solver"),
        cp.fit_intercept(),
        cp.class_weight("solver"),
        P("early_stopping", "bool", False, ("Early stopping", "Ранняя остановка"),
          ("Hold back part of the training data and stop when its score stops improving.",
           "Отложить часть обучающих данных и остановиться, когда оценка на них перестанет расти."), section="stop"),
        P("validation_fraction", "float", 0.1, ("Validation share", "Доля для проверки"),
          ("Share of training rows held back for early stopping.", "Доля обучающих строк, отложенных для ранней остановки."),
          lo=0.05, hi=0.5, step=0.05, section="stop", enabled_if={"early_stopping": [True]},
          only_tag=("early_stopping only", "только при early_stopping")),
        P("n_iter_no_change", "int", 5, ("Patience", "Терпение"),
          ("Epochs without improvement before it stops.", "Сколько эпох без улучшения ждать перед остановкой."),
          lo=1, hi=50, step=1, section="stop"),
        cp.tol(1e-3),
        P("shuffle", "bool", True, ("Shuffle each epoch", "Перемешивать каждую эпоху"),
          ("Visit the points in a new random order every epoch.", "Обходить точки в новом случайном порядке каждую эпоху."), section="repro"),
        cp.random_state(),
        cp.warm_start(),
    ]

    def proba(self, X: np.ndarray) -> np.ndarray:
        k = len(self.prep.classes)
        if self.value("loss") in ("log_loss", "modified_huber"):
            return full_proba(self.model, X, k)
        return scores_to_proba(self.model.decision_function(X), k)

    def visualize(self, grid: Grid) -> dict[str, Any]:
        v = proba_viz(self.prep, grid, self.proba, score=self.value("loss") not in ("log_loss", "modified_huber"))
        v["line"] = self._line()
        return v

    def fit_stepwise(self) -> Iterator[StepInfo]:
        p = self.prep
        k = len(p.classes)
        total = int(self.value("max_iter"))
        tol = self.value("tol")
        patience = int(self.value("n_iter_no_change"))
        loss_name = str(self.value("loss"))
        eps = float(self.value("epsilon"))
        cw = self.value("class_weight")
        model = self.build(class_weight=None, early_stopping=False)
        Xf, yf = p.X_train, p.y_train
        Xv = yv = None
        if self.value("early_stopping") and len(yf) >= 20:
            Xf, Xv, yf, yv = train_test_split(Xf, yf, test_size=float(self.value("validation_fraction")),
                                              random_state=self.ctx.seed, stratify=yf)
        sw = None
        if cw == "balanced":
            w = compute_class_weight("balanced", classes=np.arange(k), y=yf)
            sw = w[yf]
        best = np.inf
        bad = 0
        classes = np.arange(k)
        for ep in range(1, total + 1):
            model.partial_fit(Xf, yf, classes=classes, sample_weight=sw)
            self.model = model
            loss = sgd_loss_value(loss_name, model.decision_function(p.X_train), p.y_train, k, eps)
            loss_te = sgd_loss_value(loss_name, model.decision_function(p.X_test), p.y_test, k, eps) if len(p.test_idx) else None
            tr, te = self.acc_pair()
            if Xv is not None and yv is not None:
                score = -float(np.mean(model.predict(Xv) == yv))
            else:
                score = loss
            if tol is not None and score > best - float(tol):
                bad += 1
            else:
                bad = 0
            best = min(best, score)
            stop = bad >= patience
            done = ("early_stop" if Xv is not None else "converged") if stop else ("max_iter" if ep == total else None)
            yield StepInfo(ep, total, finite(loss), finite(loss_te), tr, te, done=done)
            if done:
                return

    def explain(self, row: int) -> dict[str, Any]:
        p = self.prep
        x = p.Xm[row]
        k = len(p.classes)
        loss = str(self.value("loss"))
        if k == 2:
            coef = np.ravel(self.model.coef_)
            b = float(self.model.intercept_[0])
            z = float(coef @ x + b)
            pred = int(self.model.predict(p.Xm[row:row + 1])[0])
            out: dict[str, Any] = {"binary": True, "terms": _linear_terms(p, row, coef), "intercept": b, "z": z, "pred": pred,
                                   "true": _true_label(p, row), "classes": p.classes, "loss": loss, "scaled": p.scaler is not None}
            if loss in ("log_loss", "modified_huber"):
                out["prob"] = float(self.proba(p.Xm[row:row + 1])[0][1])
            return out
        Z = self.model.coef_ @ x + self.model.intercept_
        per = [{"cls": int(c), "terms": _linear_terms(p, row, self.model.coef_[i]), "intercept": float(self.model.intercept_[i]),
                "z": float(Z[i])} for i, c in enumerate(self.model.classes_)]
        return {"binary": False, "per_class": per, "pred": int(self.model.classes_[int(np.argmax(Z))]),
                "true": _true_label(p, row), "classes": p.classes, "loss": loss, "scaled": p.scaler is not None}


ADAPTERS: list[type[Adapter]] = [LinearRegressionA, LogisticRegressionA, RidgeA, LassoA, ElasticNetA, SGDClassifierA]
