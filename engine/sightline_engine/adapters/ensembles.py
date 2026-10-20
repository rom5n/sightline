"""Ensembles: RandomForest, ExtraTrees, Bagging (warm-start batches) and
GradientBoosting, HistGradientBoosting, AdaBoost (fit once, replay stages)."""
from __future__ import annotations

import math
import time
import warnings
from typing import Any, Iterator

import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier

from . import common_params as cp
from .base import Adapter, Grid, P, StepInfo, full_proba, proba_viz


def _n_estimators(default: int, hi: int = 500) -> P:
    return P("n_estimators", "int", default, ("Number of trees", "Число деревьев"),
             ("How many trees vote. More trees → steadier answers, but slower.",
              "Сколько деревьев голосует. Больше — ответы стабильнее, но медленнее."),
             lo=1, hi=hi, step=1)


MAX_FEATURES = P("max_features", "mixed", "sqrt", ("Features per split", "Признаков на разбиение"),
                 ("How many features each split may look at. Fewer → more varied trees.",
                  "Сколько признаков смотреть при разбиении. Меньше — деревья разнообразнее."),
                 options=["sqrt", "log2", None], lo=0.05, hi=1.0, step=0.05)


def _forest_advanced(bootstrap_default: bool) -> list[P]:
    return [
        P("criterion", "enum", "gini", ("Split quality", "Качество разбиения"), cp.TREE_CRITERION_HINT,
          options=["gini", "entropy", "log_loss"], section="shape"),
        cp.min_samples_split(section="shape"),
        P("max_leaf_nodes", "int", None, ("Max leaves per tree", "Макс. листьев в дереве"),
          ("Upper limit on leaves in each tree. None = no limit.", "Верхний предел листьев в каждом дереве. None — без предела."),
          lo=2, hi=200, step=1, none_label=("No limit", "Без предела"), section="shape"),
        P("min_impurity_decrease", "float", 0.0, ("Min. impurity drop", "Мин. снижение неоднородности"),
          ("A split must reduce impurity by at least this much.", "Разбиение должно снижать неоднородность хотя бы на столько."),
          lo=0.0, hi=0.2, step=0.001, section="shape"),
        P("min_weight_fraction_leaf", "float", 0.0, ("Min. leaf weight share", "Мин. доля веса в листе"),
          ("Like min per leaf, but as a share of all points.", "Как «мин. в листе», но в долях от всех точек."),
          lo=0.0, hi=0.5, step=0.01, section="shape"),
        P("ccp_alpha", "float", 0.0, ("Pruning strength", "Сила обрезки"),
          ("Cuts back branches that help too little. 0 = no pruning.", "Срезает ветви, которые помогают слишком мало. 0 — без обрезки."),
          lo=0.0, hi=0.05, step=0.0005, section="shape"),
        P("monotonic_cst", "fixed", None, ("Monotonic constraints", "Монотонные ограничения"),
          ("Forces predictions to only rise (or fall) with a feature.", "Заставляет прогноз только расти (или падать) с признаком."),
          section="shape", disabled=("Needs one value per feature; not editable here", "Нужно значение для каждого признака; здесь не редактируется")),
        P("bootstrap", "bool", bootstrap_default, ("Bootstrap samples", "Бутстрэп-выборки"),
          ("Each tree trains on a random sample drawn with replacement.", "Каждое дерево учится на случайной выборке с повторениями."),
          section="sampling"),
        P("max_samples", "mixed", None, ("Rows per tree", "Строк на дерево"),
          ("Share of rows each tree draws when bootstrapping. None = as many as there are rows.",
           "Доля строк для каждого дерева при бутстрэпе. None — столько же, сколько строк."),
          options=[None], lo=0.05, hi=1.0, step=0.05, section="sampling", enabled_if={"bootstrap": [True]},
          only_tag=("bootstrap=True only", "только bootstrap=True")),
        P("oob_score", "bool", False, ("Out-of-bag score", "Оценка вне выборки (OOB)"),
          ("Score each tree on the rows it did not see. Shown as an extra line on the chart.",
           "Оценивать каждое дерево на строках, которых оно не видело. Показывается на графике отдельной линией."),
          section="sampling", enabled_if={"bootstrap": [True]}, only_tag=("bootstrap=True only", "только bootstrap=True")),
        cp.class_weight("other"),
        cp.random_state(),
        cp.warm_start(),
    ]


class _Averaging(Adapter):
    group = "trees"
    family = "ensemble"
    task = "classification"
    loss_kind = "trees"
    iterative = True
    step_unit = "trees"
    explain_family = "forest"
    managed = {"warm_start"}
    expensive = True

    def proba(self, X: np.ndarray) -> np.ndarray:
        return full_proba(self.model, X, len(self.prep.classes))

    def fit_stepwise(self) -> Iterator[StepInfo]:
        p = self.prep
        total = int(self.value("n_estimators"))
        batch = max(1, math.ceil(total / 25))
        model = self.build(warm_start=True, n_estimators=min(batch, total))
        oob = bool(self.value("oob_score")) and bool(self.value("bootstrap"))
        hist: list[dict[str, float | None]] = []
        k = 0
        while k < total:
            k = min(total, k + batch)
            model.set_params(n_estimators=k)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model.fit(p.X_train, p.y_train)
            self.model = model
            tr, te = self.acc_pair()
            o = float(model.oob_score_) if oob and hasattr(model, "oob_score_") else None
            hist.append({"trees": k, "test": te, "train": tr, "oob": o})
            yield StepInfo(k, total, None, None, tr, te, extra={"oob": o, "trees": k},
                           done="fitted" if k == total else None)

    def visualize(self, grid: Grid) -> dict[str, Any]:
        return proba_viz(self.prep, grid, self.proba, progress=self.ctx.progress)

    def _members(self) -> list[tuple[Any, np.ndarray | None]]:
        feats = getattr(self.model, "estimators_features_", None)
        return [(e, feats[i] if feats is not None else None) for i, e in enumerate(self.model.estimators_)]

    def explain(self, row: int) -> dict[str, Any]:
        p = self.prep
        k = len(p.classes)
        x = p.Xm[row:row + 1]
        per = []
        for i, (e, fidx) in enumerate(self._members()):
            xi = x[:, fidx] if fidx is not None else x
            pr = e.predict_proba(xi)[0]
            full = np.zeros(k)
            for j, c in enumerate(e.classes_):
                full[int(c)] = pr[j]
            per.append({"i": i, "proba": full.tolist()})
        avg = self.proba(x)[0]
        pred = int(np.argmax(avg))
        return {"trees": per, "avg": avg.tolist(), "pred": pred, "true": int(p.y[row]) if p.y is not None else None,
                "classes": p.classes, "n": len(per)}


class RandomForestA(_Averaging):
    name = "RandomForestClassifier"
    est = RandomForestClassifier
    main = [_n_estimators(100), cp.max_depth(), MAX_FEATURES, cp.min_samples_leaf()]
    advanced = _forest_advanced(True)


class ExtraTreesA(_Averaging):
    name = "ExtraTreesClassifier"
    est = ExtraTreesClassifier
    main = [_n_estimators(100), cp.max_depth(), MAX_FEATURES, cp.min_samples_leaf()]
    advanced = _forest_advanced(False)


class BaggingA(_Averaging):
    name = "BaggingClassifier"
    est = BaggingClassifier
    main = [
        P("n_estimators", "int", 10, ("Number of copies", "Число копий"),
          ("How many copies of the base model are trained and vote.", "Сколько копий базовой модели обучается и голосует."),
          lo=1, hi=200, step=1),
        P("max_samples", "mixed", None, ("Rows per copy", "Строк на копию"),
          ("Share of rows each copy trains on. None = as many as there are rows (1.0).",
           "Доля строк для каждой копии. None — столько же, сколько строк (1.0)."),
          options=[None], lo=0.05, hi=1.0, step=0.05, initial=1.0),
        P("max_features", "float", 1.0, ("Features per copy", "Признаков на копию"),
          ("Share of features each copy may use.", "Доля признаков, доступных каждой копии."), lo=0.1, hi=1.0, step=0.05),
    ]
    advanced = [
        P("estimator", "fixed", None, ("Base model", "Базовая модель"),
          ("The model that is copied. None = a decision tree.", "Модель, которую копируют. None — дерево решений."),
          section="sampling", disabled=("Always a decision tree in Sightline", "В Sightline это всегда дерево решений")),
        P("bootstrap", "bool", True, ("Draw rows with replacement", "Строки с повторениями"),
          ("Each copy samples rows with replacement (classic bagging).", "Каждая копия берёт строки с повторениями (классический бэггинг)."),
          section="sampling"),
        P("bootstrap_features", "bool", False, ("Draw features with replacement", "Признаки с повторениями"),
          ("Sample features with replacement too.", "Выбирать с повторениями и признаки."), section="sampling"),
        P("oob_score", "bool", False, ("Out-of-bag score", "Оценка вне выборки (OOB)"),
          ("Score each copy on the rows it did not see.", "Оценивать каждую копию на строках, которых она не видела."),
          section="sampling", enabled_if={"bootstrap": [True]}, only_tag=("bootstrap=True only", "только bootstrap=True")),
        cp.random_state(),
        cp.warm_start(),
    ]

    def sk_params(self, **override: Any) -> dict[str, Any]:
        out = super().sk_params(**override)
        if out.get("max_samples") is not None:
            out["max_samples"] = float(out["max_samples"])
        return out


# ------------------------------------------------------------------ boosting


class _Staged:
    """Advances a staged_predict_proba generator over a fixed set of rows."""

    def __init__(self, model: Any, X: np.ndarray) -> None:
        self.gen = model.staged_predict_proba(X)
        self.stage = 0
        self.last: np.ndarray | None = None

    def at(self, stage: int) -> np.ndarray:
        while self.stage < stage:
            try:
                self.last = next(self.gen)
            except StopIteration:
                break
            self.stage += 1
        assert self.last is not None
        return self.last


class _Boosting(Adapter):
    group = "trees"
    family = "boost"
    task = "classification"
    loss_kind = "boost"
    iterative = True
    step_unit = "stage"
    explain_family = "boost"

    def __init__(self, ctx: Any) -> None:
        super().__init__(ctx)
        self.stage = 0
        self.n_stages = 0
        self.cache: dict[tuple[float, ...], _Staged] = {}
        self.curve: list[dict[str, float | None]] = []
        self.early_stop: int | None = None

    def proba_full(self, X: np.ndarray) -> np.ndarray:
        return full_proba(self.model, X, len(self.prep.classes))

    def loss_of(self, y: np.ndarray, pr: np.ndarray) -> float:
        return float(log_loss(y, np.clip(pr, 1e-15, 1), labels=list(range(len(self.prep.classes)))))

    def stages_total(self) -> int:
        return int(self.model.n_estimators_) if hasattr(self.model, "n_estimators_") else len(self.model.estimators_)

    def fit_stepwise(self) -> Iterator[StepInfo]:
        p = self.prep
        t = time.perf_counter()
        yield StepInfo(0, 1, blocking=0.0)
        self.model = self.build().fit(p.X_train, p.y_train)
        self.t_fit = (time.perf_counter() - t) * 1000
        n = self.stages_total()
        self.n_stages = n
        self.cache = {}
        tr_gen = _Staged(self.model, p.X_train)
        te_gen = _Staged(self.model, p.X_test) if len(p.test_idx) else None
        best = (np.inf, 0)
        k = len(p.classes)
        labels = np.arange(k)
        for s in range(1, n + 1):
            self.stage = s
            ptr = tr_gen.at(s)
            pte = te_gen.at(s) if te_gen else None
            ptr_full = _expand(ptr, self.model.classes_, k)
            pte_full = _expand(pte, self.model.classes_, k) if pte is not None else None
            loss = self.loss_of(p.y_train, ptr_full)
            loss_te = self.loss_of(p.y_test, pte_full) if pte_full is not None else None
            acc_tr = float(np.mean(labels[np.argmax(ptr_full, 1)] == p.y_train))
            acc_te = float(np.mean(labels[np.argmax(pte_full, 1)] == p.y_test)) if pte_full is not None else None
            if loss_te is not None and loss_te < best[0]:
                best = (loss_te, s)
            self.curve.append({"stage": s, "train": loss, "test": loss_te})
            yield StepInfo(s, n, loss, loss_te, acc_tr, acc_te,
                           extra={"best_stage": best[1], "early_stop": self.early_stop, "metric_name": self.metric_name()},
                           done="fitted" if s == n else None)

    def metric_name(self) -> str:
        return "log_loss"

    def visualize(self, grid: Grid) -> dict[str, Any]:
        p = self.prep
        key = (grid.x0, grid.x1, grid.y0, grid.y1, float(grid.w), float(grid.h))
        k = len(p.classes)
        if self.stage >= self.n_stages:
            return proba_viz(p, grid, self.proba_full)
        st = self.cache.get(key)
        if st is None or st.stage > self.stage:
            self.cache = {}
            mesh = p.model_input(p.mesh(grid.x0, grid.x1, grid.y0, grid.y1, grid.w, grid.h))
            st = _Staged(self.model, mesh)
            self.cache[key] = st
        pr = _expand(st.at(self.stage), self.model.classes_, k)
        data = pr.T.reshape(k, grid.h, grid.w).astype(np.float32)
        return {"kind": "proba", "grid": {"bounds": grid.bounds, "w": grid.w, "h": grid.h, "k": k, "data": data}, "score": False}

    def staged_scores(self, x: np.ndarray) -> list[np.ndarray]:
        return [np.atleast_1d(np.ravel(d)) for d in self.model.staged_decision_function(x)]

    def baseline(self, x: np.ndarray, first: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def explain(self, row: int) -> dict[str, Any]:
        p = self.prep
        x = p.Xm[row:row + 1]
        staged = self.staged_scores(x)
        base = self.baseline(x, staged[0])
        contribs = [staged[0] - base] + [staged[i] - staged[i - 1] for i in range(1, len(staged))]
        score = staged[-1]
        k = len(p.classes)
        proba = self.proba_full(x)[0]
        pred = int(np.argmax(proba))
        return {"kind": "additive", "base": base.tolist(), "contribs": [c.tolist() for c in contribs],
                "score": score.tolist(), "proba": proba.tolist(), "pred": pred, "binary": k == 2,
                "true": int(p.y[row]) if p.y is not None else None, "classes": p.classes, "n": len(contribs)}


def _expand(pr: np.ndarray, classes: np.ndarray, k: int) -> np.ndarray:
    if pr.shape[1] == k:
        return pr
    out = np.zeros((pr.shape[0], k))
    for j, c in enumerate(classes):
        out[:, int(c)] = pr[:, j]
    return out


class GradientBoostingA(_Boosting):
    name = "GradientBoostingClassifier"
    est = GradientBoostingClassifier
    main = [
        P("n_estimators", "int", 100, ("Number of stages", "Число этапов"),
          ("How many small trees are added one after another.", "Сколько маленьких деревьев добавляется одно за другим."),
          lo=1, hi=500, step=1),
        P("learning_rate", "float", 0.1, ("Learning rate", "Скорость обучения"),
          ("How much each new tree may correct. Smaller → needs more trees, but generalises better.",
           "Насколько сильно каждое дерево исправляет ошибки. Меньше — нужно больше деревьев, но обобщает лучше."),
          lo=0.001, hi=1.0, log=True),
        P("max_depth", "int", 3, ("Tree depth", "Глубина дерева"),
          ("Depth of each small tree. 1–3 is typical for boosting.", "Глубина каждого маленького дерева. Для бустинга обычно 1–3."),
          lo=1, hi=10, step=1, none_label=("No limit", "Без ограничения")),
        P("subsample", "float", 1.0, ("Row share per stage", "Доля строк на этап"),
          ("Below 1.0 each tree sees a random part of the rows (stochastic boosting).",
           "Меньше 1.0 — каждое дерево видит случайную часть строк (стохастический бустинг)."),
          lo=0.1, hi=1.0, step=0.05),
    ]
    advanced = [
        P("loss", "enum", "log_loss", ("Loss", "Функция потерь"),
          ("'log_loss' gives probabilities; 'exponential' turns it into AdaBoost (2 classes only).",
           "«log_loss» даёт вероятности; «exponential» превращает его в AdaBoost (только 2 класса)."),
          options=["log_loss", "exponential"], section="solver"),
        P("criterion", "fixed", "deprecated", ("Split quality", "Качество разбиения"),
          ("Deprecated in this scikit-learn version; trees always use friedman_mse.",
           "Устарел в этой версии scikit-learn; деревья всегда используют friedman_mse."),
          section="shape", disabled=("Deprecated in scikit-learn 1.9", "Устарел в scikit-learn 1.9")),
        cp.min_samples_split(section="shape"),
        cp.min_samples_leaf(section="shape"),
        P("max_features", "mixed", None, ("Features per split", "Признаков на разбиение"),
          ("How many features each split may look at. None = all.", "Сколько признаков смотреть при разбиении. None — все."),
          options=[None, "sqrt", "log2"], lo=0.05, hi=1.0, step=0.05, section="shape"),
        P("max_leaf_nodes", "int", None, ("Max leaves per tree", "Макс. листьев в дереве"),
          ("Upper limit on leaves in each tree. None = no limit.", "Верхний предел листьев. None — без предела."),
          lo=2, hi=100, step=1, none_label=("No limit", "Без предела"), section="shape"),
        P("min_impurity_decrease", "float", 0.0, ("Min. impurity drop", "Мин. снижение неоднородности"),
          ("A split must reduce impurity by at least this much.", "Разбиение должно снижать неоднородность хотя бы на столько."),
          lo=0.0, hi=0.2, step=0.001, section="shape"),
        P("min_weight_fraction_leaf", "float", 0.0, ("Min. leaf weight share", "Мин. доля веса в листе"),
          ("Like min per leaf, but as a share of all points.", "Как «мин. в листе», но в долях от всех точек."),
          lo=0.0, hi=0.5, step=0.01, section="shape"),
        P("ccp_alpha", "float", 0.0, ("Pruning strength", "Сила обрезки"),
          ("Cuts back branches that help too little.", "Срезает ветви, которые помогают слишком мало."),
          lo=0.0, hi=0.05, step=0.0005, section="shape"),
        P("n_iter_no_change", "int", None, ("Early-stopping patience", "Терпение ранней остановки"),
          ("Stop when the validation score has not improved for this many stages. None = never stop early.",
           "Остановиться, если оценка на проверке не растёт столько этапов. None — без ранней остановки."),
          lo=1, hi=50, step=1, none_label=("Off", "Выкл."), section="stop"),
        P("validation_fraction", "float", 0.1, ("Validation share", "Доля для проверки"),
          ("Share of training rows held back for early stopping.", "Доля обучающих строк для ранней остановки."),
          lo=0.05, hi=0.5, step=0.05, section="stop"),
        cp.tol(1e-4),
        P("init", "fixed", None, ("Initial model", "Начальная модель"),
          ("The model that makes the base guess. None = class shares.", "Модель для начальной догадки. None — доли классов."),
          section="solver", disabled=("Always the class shares in Sightline", "В Sightline это всегда доли классов")),
        cp.random_state(),
        cp.warm_start(),
    ]

    def stages_total(self) -> int:
        return int(self.model.n_estimators_)

    def baseline(self, x: np.ndarray, first: np.ndarray) -> np.ndarray:
        lr = float(self.model.learning_rate)
        tree1 = np.array([e.predict(x)[0] for e in self.model.estimators_[0]])
        return np.asarray(first - lr * tree1)


class HistGradientBoostingA(_Boosting):
    name = "HistGradientBoostingClassifier"
    est = HistGradientBoostingClassifier
    main = [
        P("max_iter", "int", 100, ("Number of stages", "Число этапов"),
          ("How many boosting stages (trees) at most.", "Максимальное число этапов (деревьев)."), lo=1, hi=500, step=1),
        P("learning_rate", "float", 0.1, ("Learning rate", "Скорость обучения"),
          ("How much each new tree may correct.", "Насколько сильно каждое новое дерево исправляет ошибки."),
          lo=0.001, hi=1.0, log=True),
        P("max_leaf_nodes", "int", 31, ("Max leaves per tree", "Макс. листьев в дереве"),
          ("Size of each tree. Fewer leaves → simpler steps.", "Размер каждого дерева. Меньше листьев — проще шаги."),
          lo=2, hi=128, step=1, none_label=("No limit", "Без предела")),
        P("max_depth", "int", None, ("Max depth", "Макс. глубина"),
          ("Depth limit for each tree. None = only the leaf limit applies.", "Предел глубины дерева. None — действует только предел листьев."),
          lo=1, hi=20, step=1, none_label=("No limit (None, the default)", "Без ограничения (None, по умолчанию)")),
        P("l2_regularization", "float", 0.0, ("L2 penalty", "Штраф L2"),
          ("Keeps leaf values small. Raise it for smoother borders.", "Держит значения листьев маленькими. Увеличьте для более гладких границ."),
          lo=0.0, hi=10.0, step=0.1),
    ]
    advanced = [
        P("loss", "enum", "log_loss", ("Loss", "Функция потерь"), ("What is minimised: log-loss.", "Что минимизируется: log-loss."),
          options=["log_loss"], section="solver"),
        P("min_samples_leaf", "int", 20, ("Min. per leaf", "Мин. в листе"),
          ("Every leaf keeps at least this many rows.", "В каждом листе не меньше столько строк."), lo=1, hi=100, step=1, section="shape"),
        P("max_features", "float", 1.0, ("Feature share per split", "Доля признаков на разбиение"),
          ("Share of features considered at each split.", "Доля признаков для каждого разбиения."), lo=0.1, hi=1.0, step=0.05, section="shape"),
        P("max_bins", "int", 255, ("Bins per feature", "Корзин на признак"),
          ("Values are sorted into this many bins before training.", "Перед обучением значения раскладываются по столько корзинам."),
          lo=2, hi=255, step=1, section="shape"),
        P("early_stopping", "mixed", "auto", ("Early stopping", "Ранняя остановка"),
          ("'auto' turns it on only for more than 10 000 rows. Turn it on to see the stopping point.",
           "«auto» включает её только при числе строк больше 10 000. Включите, чтобы увидеть точку остановки."),
          options=["auto", True, False], section="stop"),
        P("scoring", "enum", "loss", ("Early-stopping score", "Оценка для остановки"),
          ("What early stopping watches.", "За чем следит ранняя остановка."), options=["loss", "accuracy"], section="stop"),
        P("validation_fraction", "float", 0.1, ("Validation share", "Доля для проверки"),
          ("Share of training rows held back for early stopping.", "Доля обучающих строк для ранней остановки."),
          lo=0.05, hi=0.5, step=0.05, section="stop"),
        P("n_iter_no_change", "int", 10, ("Patience", "Терпение"),
          ("Stages without improvement before it stops.", "Сколько этапов без улучшения ждать."), lo=1, hi=50, step=1, section="stop"),
        cp.tol(1e-7),
        cp.class_weight("other"),
        P("categorical_features", "fixed", "from_dtype", ("Categorical features", "Категориальные признаки"),
          ("Which features are categories.", "Какие признаки — категории."), section="other",
          disabled=("Sightline one-hot encodes categories itself", "Sightline сам кодирует категории (one-hot)")),
        P("interaction_cst", "fixed", None, ("Interaction constraints", "Ограничения взаимодействий"),
          ("Which features may be combined in one tree.", "Какие признаки можно сочетать в одном дереве."), section="other",
          disabled=("Not editable here", "Здесь не редактируется")),
        P("monotonic_cst", "fixed", None, ("Monotonic constraints", "Монотонные ограничения"),
          ("Forces predictions to only rise (or fall) with a feature.", "Заставляет прогноз только расти (или падать)."), section="other",
          disabled=("Needs one value per feature; not editable here", "Нужно значение для каждого признака; здесь не редактируется")),
        cp.random_state(),
        cp.warm_start(),
    ]

    def stages_total(self) -> int:
        return int(self.model.n_iter_)

    def fit_stepwise(self) -> Iterator[StepInfo]:
        for st in super().fit_stepwise():
            if self.model is not None and self.early_stop is None and len(getattr(self.model, "validation_score_", [])) > 0:
                if int(self.model.n_iter_) < int(self.value("max_iter")):
                    self.early_stop = int(self.model.n_iter_)
                    st.extra["early_stop"] = self.early_stop
            if st.done:
                st.extra["validation_score"] = [float(v) for v in getattr(self.model, "validation_score_", [])]
                st.extra["train_score"] = [float(v) for v in getattr(self.model, "train_score_", [])]
                if self.early_stop:
                    st.done = "fitted"
            yield st

    def baseline(self, x: np.ndarray, first: np.ndarray) -> np.ndarray:
        b = getattr(self.model, "_baseline_prediction", None)
        if b is None:
            return np.zeros_like(first)
        return np.asarray(np.atleast_1d(np.ravel(np.asarray(b, dtype=float)))[: len(first)] * np.ones_like(first))


class AdaBoostA(_Boosting):
    name = "AdaBoostClassifier"
    est = AdaBoostClassifier
    explain_family = "ada"
    main = [
        P("n_estimators", "int", 50, ("Number of models", "Число моделей"),
          ("How many weak models are added one after another.", "Сколько слабых моделей добавляется одна за другой."), lo=1, hi=500, step=1),
        P("learning_rate", "float", 1.0, ("Learning rate", "Скорость обучения"),
          ("Shrinks each model's say. Smaller → needs more models.", "Уменьшает голос каждой модели. Меньше — нужно больше моделей."),
          lo=0.01, hi=2.0, log=True),
        P("estimator_depth", "int", 1, ("Base tree depth", "Глубина базового дерева"),
          ("Depth of each weak tree. 1 = a stump: one question only.", "Глубина каждого слабого дерева. 1 — пень: всего один вопрос."),
          lo=1, hi=6, step=1, sk=False, sk_name="estimator__max_depth"),
    ]
    advanced = [
        P("estimator", "fixed", None, ("Base model", "Базовая модель"),
          ("The weak model that is boosted. None = a depth-1 tree.", "Слабая модель, которую усиливают. None — дерево глубины 1."),
          section="solver", disabled=("A decision tree; set its depth in the main panel", "Дерево решений; глубина задаётся в основной панели")),
        cp.random_state(),
    ]

    def sk_params(self, **override: Any) -> dict[str, Any]:
        out = super().sk_params(**override)
        out["estimator"] = DecisionTreeClassifier(max_depth=int(self.value("estimator_depth")))
        return out

    def stages_total(self) -> int:
        return len(self.model.estimators_)

    def metric_name(self) -> str:
        return "error"

    def loss_of(self, y: np.ndarray, pr: np.ndarray) -> float:
        return float(np.mean(np.argmax(pr, 1) != y))

    def explain(self, row: int) -> dict[str, Any]:
        p = self.prep
        x = p.Xm[row:row + 1]
        k = len(p.classes)
        w = np.asarray(self.model.estimator_weights_[: len(self.model.estimators_)], dtype=float)
        votes = []
        share = np.zeros(k)
        for i, e in enumerate(self.model.estimators_):
            c = int(e.predict(x)[0])
            votes.append({"i": i, "cls": c, "weight": float(w[i])})
            share[c] += w[i]
        share = share / max(1e-12, w.sum())
        proba = self.proba_full(x)[0]
        pred = int(self.model.predict(x)[0])
        return {"kind": "votes", "votes": votes, "share": share.tolist(), "proba": proba.tolist(), "pred": pred,
                "true": int(p.y[row]) if p.y is not None else None, "classes": p.classes, "n": len(votes), "binary": k == 2}

    def baseline(self, x: np.ndarray, first: np.ndarray) -> np.ndarray:
        return np.zeros_like(first)


ADAPTERS: list[type[Adapter]] = [RandomForestA, ExtraTreesA, GradientBoostingA, HistGradientBoostingA, AdaBoostA, BaggingA]

