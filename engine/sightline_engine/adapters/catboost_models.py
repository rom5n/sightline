"""CatBoost: gradient boosting on oblivious (symmetric) trees with native categorical features.

Categorical columns are handed to CatBoost as they are (`cat_features`), never one-hot encoded:
CatBoost turns each category into an ordered target statistic (CTR) itself. Prepared keeps the
encoded matrix for the other models; `Prepared.raw()` rebuilds the original columns from it.

Training is one native fit (with real progress and cancel through a CatBoost callback); the frames
then replay it iteration by iteration: the train loss CatBoost recorded per iteration
(`get_evals_result`), the test loss computed afterwards (`eval_metrics`, so the test rows never
steer training), and the model cut at that many trees (`ntree_end`).
"""
from __future__ import annotations

import math
import re
import time
from typing import Any, Iterator

import numpy as np

from .base import Adapter, Grid, P, StepInfo, softmax, sigmoid

MAX_FRAMES = 200


def _cb() -> Any:
    import catboost  # imported lazily: ~1 s and a large native library only when CatBoost is used

    return catboost


AUTO = ("Auto", "Авто")

_MAIN = [
    P("iterations", "int", 1000, ("Number of trees", "Число деревьев"),
      ("How many trees are added one after another; each fixes part of the remaining error. CatBoost's default is 1000; "
       "Sightline starts at 300 so a run takes seconds.",
       "Сколько деревьев добавляется одно за другим; каждое исправляет часть оставшейся ошибки. По умолчанию в CatBoost — 1000; "
       "Sightline начинает с 300, чтобы обучение занимало секунды."),
      lo=10, hi=2000, step=10, initial=300),
    P("learning_rate", "float", None, ("Learning rate", "Скорость обучения"),
      ("How much each tree may correct. Auto = CatBoost picks it from the data size and the number of trees.",
       "Насколько сильно каждое дерево исправляет ошибку. Авто — CatBoost подбирает по размеру данных и числу деревьев."),
      lo=0.001, hi=1.0, log=True, none_label=AUTO),
    P("depth", "int", 6, ("Tree depth", "Глубина дерева"),
      ("Levels in each symmetric tree; a tree of depth d has 2^d leaves. 4–8 is typical.",
       "Уровней в каждом симметричном дереве; у дерева глубины d — 2^d листьев. Обычно 4–8."),
      lo=1, hi=10, step=1),
    P("l2_leaf_reg", "float", 3.0, ("L2 leaf regularisation", "L2-регуляризация листьев"),
      ("Pulls leaf values towards zero. Higher → smoother, less overfitting.",
       "Тянет значения листьев к нулю. Больше — глаже и меньше переобучения."),
      lo=0.0, hi=30.0, step=0.5),
]


def _advanced(classifier: bool) -> list[P]:
    adv = [
        P("border_count", "int", 254, ("Number of borders", "Число границ"),
          ("Numbers are cut into at most this many bins before splitting. Fewer → faster, coarser.",
           "Числа делятся максимум на столько корзин перед разбиением. Меньше — быстрее, грубее."),
          lo=1, hi=1024, step=1, section="shape"),
        P("random_strength", "float", 1.0, ("Split randomness", "Случайность разбиений"),
          ("Noise added to split scores so that trees differ. 0 = always the best split.",
           "Шум к оценкам разбиений, чтобы деревья различались. 0 — всегда лучшее разбиение."),
          lo=0.0, hi=10.0, step=0.1, section="shape"),
        P("bootstrap_type", "enum", None, ("Row sampling", "Выборка строк"),
          ("How rows are sampled for each tree. Auto: MVS for 2 classes and regression, Bayesian for more classes.",
           "Как выбираются строки для каждого дерева. Авто: MVS для 2 классов и регрессии, Bayesian для большего числа классов."),
          options=[None, "Bayesian", "Bernoulli", "MVS", "No"], none_label=AUTO, section="sampling"),
        P("bagging_temperature", "float", 1.0, ("Bagging temperature", "Температура бэггинга"),
          ("Bayesian sampling only: 0 = every row weight 1, higher = more random weights.",
           "Только для Bayesian: 0 — у всех строк вес 1, больше — веса случайнее."),
          lo=0.0, hi=10.0, step=0.1, section="sampling", enabled_if={"bootstrap_type": ["Bayesian"]},
          only_tag=("bootstrap_type='Bayesian' only", "только при bootstrap_type='Bayesian'")),
        P("subsample", "float", None, ("Row share per tree", "Доля строк на дерево"),
          ("Share of rows each tree sees (Bernoulli and MVS sampling). Auto = 0.8.",
           "Доля строк для каждого дерева (выборки Bernoulli и MVS). Авто — 0.8."),
          lo=0.1, hi=1.0, step=0.05, none_label=AUTO, section="sampling",
          enabled_if={"bootstrap_type": ["Bernoulli", "MVS", None]}),
        P("rsm", "float", 1.0, ("Feature share per split", "Доля признаков на разбиение"),
          ("Random share of features each split may use. 1 = all.", "Случайная доля признаков для разбиения. 1 — все."),
          lo=0.1, hi=1.0, step=0.05, section="sampling"),
        P("one_hot_max_size", "int", None, ("One-hot up to N values", "One-hot до N значений"),
          ("Categories with at most this many values are one-hot encoded inside CatBoost; the rest use target statistics. Auto = 2.",
           "Категории с не больше чем N значениями CatBoost кодирует one-hot сам; остальные — целевыми статистиками. Авто — 2."),
          lo=1, hi=255, step=1, none_label=AUTO, section="cat"),
        P("max_ctr_complexity", "int", None, ("Category combinations", "Комбинации категорий"),
          ("How many categorical columns may be combined into one feature. Auto = 4.",
           "Сколько категориальных столбцов можно объединить в один признак. Авто — 4."),
          lo=1, hi=6, step=1, none_label=AUTO, section="cat"),
        P("boosting_type", "enum", None, ("Boosting scheme", "Схема бустинга"),
          ("'Ordered' fights target leakage on small data but is slower; 'Plain' is classic boosting. Auto picks by data size.",
           "«Ordered» борется с утечкой целевой переменной на малых данных, но медленнее; «Plain» — классический бустинг. Авто выбирает по размеру данных."),
          options=[None, "Plain", "Ordered"], none_label=AUTO, section="solver"),
        P("leaf_estimation_iterations", "int", None, ("Leaf refinement steps", "Шагов уточнения листьев"),
          ("Newton steps used to compute each leaf value. Auto depends on the loss.",
           "Шагов Ньютона для расчёта значения листа. Авто зависит от функции потерь."),
          lo=1, hi=20, step=1, none_label=AUTO, section="solver"),
        P("model_size_reg", "float", 0.5, ("Model size regularisation", "Регуляризация размера"),
          ("Penalises large models built from many category combinations.",
           "Штраф за большие модели из многих комбинаций категорий."),
          lo=0.0, hi=10.0, step=0.1, section="cat"),
        P("nan_mode", "enum", "Min", ("Missing values", "Пропуски"),
          ("Where missing numbers go: below every value ('Min'), above ('Max') or forbidden.",
           "Куда идут пропущенные числа: ниже всех значений («Min»), выше («Max») или запрещены."),
          options=["Min", "Max", "Forbidden"], section="solver"),
        P("grow_policy", "fixed", "SymmetricTree", ("Tree shape", "Форма дерева"),
          ("Symmetric (oblivious) trees ask the same question on a whole level — CatBoost's default and what Sightline explains.",
           "Симметричные (oblivious) деревья задают один вопрос на весь уровень — по умолчанию в CatBoost; их и объясняет Sightline."),
          section="shape", disabled=("Sightline explains symmetric trees", "Sightline объясняет симметричные деревья")),
        P("random_seed", "int", 0, ("Random seed", "Зерно случайности"),
          ("Same seed + same data = the same model.", "Одно зерно и те же данные — та же модель."),
          lo=0, hi=9999, step=1, section="repro"),
    ]
    if classifier:
        adv.insert(0, P("auto_class_weights", "enum", None, ("Class weights", "Веса классов"),
                        ("'Balanced' counts rare classes more, so the model can't ignore them.",
                         "«Balanced» сильнее учитывает редкие классы, чтобы модель их не игнорировала."),
                        options=[None, "Balanced", "SqrtBalanced"], none_label=("None", "Нет"), section="solver"))
    return adv


class _CatBoost(Adapter):
    group = "trees"
    family = "boost"
    iterative = True
    step_unit = "trees"
    explain_family = "catboost"
    lock_while_training = True
    expensive = True
    est = type(None)  # the class lives in catboost; see build()
    main = _MAIN

    def __init__(self, ctx: Any) -> None:
        super().__init__(ctx)
        self.stage = 0
        self.n_trees = 0
        self._mesh: tuple[tuple[float, ...], Any] | None = None

    # ---------------------------------------------------------------- data

    def pool(self, X: np.ndarray, y: np.ndarray | None = None) -> Any:
        p = self.prep
        return _cb().Pool(p.raw(X), y, cat_features=p.cat_features, feature_names=[c.name for c in p.columns])

    def build(self, **override: Any) -> Any:
        cls = getattr(_cb(), self.name)
        params = {k: v for k, v in self.sk_params(**override).items() if v is not None}
        return cls(**params, logging_level="Silent", allow_writing_files=False, thread_count=-1)

    def sk_params(self, **override: Any) -> dict[str, Any]:
        out = super().sk_params(**override)
        out.pop("grow_policy", None)  # fixed: always SymmetricTree
        return out

    # ---------------------------------------------------------------- fitting

    def fit_stepwise(self) -> Iterator[StepInfo]:
        p = self.prep
        t = time.perf_counter()
        yield StepInfo(0, 1, blocking=0.0)
        train = self.pool(p.X[p.train_idx], p.y[p.train_idx] if p.y is not None else None)
        test = self.pool(p.X[p.test_idx], p.y[p.test_idx]) if len(p.test_idx) and p.y is not None else None
        n_iter = int(self.value("iterations"))
        ctx = self.ctx

        class Progress:  # CatBoost calls this after every tree; False stops the fit (cancel)
            def after_iteration(self, info: Any) -> bool:
                ctx.progress(min(1.0, (info.iteration + 1) / n_iter))
                return not ctx.cancelled()

        model = self.build()
        model.fit(train, callbacks=[Progress()])
        self.model = model
        self.t_fit = (time.perf_counter() - t) * 1000
        n = int(model.tree_count_)
        self.n_trees = n
        metric = self.loss_metric()
        learn = model.get_evals_result().get("learn", {}).get(metric, [])
        test_loss = model.eval_metrics(test, [metric], eval_period=1)[metric] if test is not None else []
        stride = max(1, math.ceil(n / MAX_FRAMES))
        stops = list(range(stride, n + 1, stride))
        if not stops or stops[-1] != n:
            stops.append(n)
        tr_pred = self.staged(train, stops)
        te_pred = self.staged(test, stops) if test is not None else None
        for i, s in enumerate(stops):
            self.stage = s
            m_tr = self.metric(p.y[p.train_idx], tr_pred[i]) if p.y is not None else None
            m_te = self.metric(p.y[p.test_idx], te_pred[i]) if te_pred is not None and p.y is not None else None
            lt = float(test_loss[s - 1]) if len(test_loss) >= s else None
            yield StepInfo(s, n, float(learn[s - 1]) if len(learn) >= s else None, lt, m_tr, m_te,
                           extra={"metric_name": metric, "cat_features": [p.columns[j].name for j in p.cat_features],
                                  "learning_rate": float(model.learning_rate_), "stride": stride},
                           done="fitted" if s == n else None)

    def staged(self, pool: Any, stops: list[int]) -> list[np.ndarray]:
        """Predictions after each count of trees in `stops` (one incremental pass over the trees)."""
        out: list[np.ndarray] = []
        want = set(stops)
        for k, pred in enumerate(self.model.staged_predict(pool, prediction_type=self.pred_type(), eval_period=1), start=1):
            if k in want:
                out.append(np.asarray(pred))
        return out

    def loss_metric(self) -> str:
        raise NotImplementedError

    def pred_type(self) -> str:
        raise NotImplementedError

    def metric(self, y: np.ndarray, pred: np.ndarray) -> float:
        raise NotImplementedError

    # ---------------------------------------------------------------- views

    def mesh_pool(self, grid: Grid) -> Any:
        key = (grid.x0, grid.x1, grid.y0, grid.y1, float(grid.w), float(grid.h))
        if self._mesh is None or self._mesh[0] != key:
            p = self.prep
            feats = p.plane.to_features(p.mesh(grid.x0, grid.x1, grid.y0, grid.y1, grid.w, grid.h))
            self._mesh = (key, self.pool(feats))
        return self._mesh[1]

    def predict_row(self, row: int) -> Any:
        return self.model.predict(self.pool(self.prep.X[row:row + 1]))[0]

    def raw_score(self, row: int) -> tuple[np.ndarray, list[np.ndarray]]:
        """Bias and every tree's contribution to the raw score of one row (exact staged differences)."""
        pool = self.pool(self.prep.X[row:row + 1])
        staged = [np.atleast_1d(np.ravel(s)) for s in self.model.staged_predict(pool, prediction_type="RawFormulaVal", eval_period=1)]
        scale, bias = self.model.get_scale_and_bias()
        base = np.atleast_1d(np.asarray(bias, dtype=float)) * np.ones_like(staged[0])
        contribs = [staged[0] - base] + [staged[i] - staged[i - 1] for i in range(1, len(staged))]
        return base, contribs

    def tree_paths(self, row: int, n: int = 3) -> list[dict[str, Any]]:
        """The first trees as CatBoost stores them: one question per level, the row's answers, its leaf."""
        p = self.prep
        pool = self.pool(p.X[row:row + 1])
        n = min(n, self.n_trees)
        leaves = self.model.calc_leaf_indexes(pool, ntree_end=n)[0]
        counts = list(self.model.get_tree_leaf_counts())
        values = np.asarray(self.model.get_leaf_values())
        dims = values.size // int(sum(counts)) if sum(counts) else 1
        names = [c.name for c in p.columns]
        raw = p.raw(p.X[row:row + 1])[0]
        out = []
        start = 0
        for t in range(n):
            leaf = int(leaves[t])
            levels = []
            for d, s in enumerate(self.model._get_tree_splits(t, pool)):
                levels.append({**_parse_split(str(s), names, raw), "yes": bool((leaf >> d) & 1)})
            v = values[(start + leaf) * dims:(start + leaf + 1) * dims]
            out.append({"levels": levels, "leaf": leaf, "leaves": int(counts[t]), "value": [float(x) for x in v]})
            start += int(counts[t])
        return out

    def explain(self, row: int) -> dict[str, Any]:
        p = self.prep
        base, contribs = self.raw_score(row)
        score = base + np.sum(contribs, axis=0)
        raw = p.raw(p.X[row:row + 1])[0]
        cols = [{"name": c.name, "value": (str(raw[j]) if c.kind == "category" else float(raw[j])), "cat": c.kind == "category"}
                for j, c in enumerate(p.columns)]
        d: dict[str, Any] = {"kind": "catboost", "base": base.tolist(), "contribs": [c.tolist() for c in contribs],
                             "score": score.tolist(), "trees": self.tree_paths(row), "columns": cols,
                             "n": len(contribs), "depth": int(self.value("depth")),
                             "learning_rate": float(self.model.learning_rate_)}
        d.update(self.final(row, score))
        return d

    def final(self, row: int, score: np.ndarray) -> dict[str, Any]:
        raise NotImplementedError

    def summary(self) -> dict[str, Any]:
        return {"trees": self.n_trees, "learning_rate": float(self.model.learning_rate_),
                "cat_features": [self.prep.columns[j].name for j in self.prep.cat_features]}


def _parse_split(s: str, names: list[str], raw: np.ndarray) -> dict[str, Any]:
    """'x1, bin=0.048' (number > border) or '{color} counter_type=Counter ..., border=14' (a split on the
    target statistic CatBoost computed for a category, or for a combination '{a, b}')."""
    m = re.match(r"^(.+?), bin=([-+0-9.eE]+)$", s)
    if m and m.group(1) in names:
        j = names.index(m.group(1))
        return {"feature": names[j], "kind": "num", "border": float(m.group(2)), "value": float(raw[j])}
    m = re.match(r"^\{(.+?)\}\s*counter_type=(\w+).*?border=(\d+)", s)
    if m:
        cols = [c.strip() for c in m.group(1).split(",")]
        vals = [str(raw[names.index(c)]) if c in names else "?" for c in cols]
        return {"feature": " × ".join(cols), "kind": "ctr", "ctr": m.group(2), "bin": int(m.group(3)),
                "value": " × ".join(vals)}
    return {"feature": s, "kind": "other"}


class CatBoostClassifierA(_CatBoost):
    name = "CatBoostClassifier"
    task = "classification"
    loss_kind = "boost"
    advanced = _advanced(True)

    def loss_metric(self) -> str:
        return "Logloss" if len(self.prep.classes) == 2 else "MultiClass"

    def pred_type(self) -> str:
        return "Probability"

    def metric(self, y: np.ndarray, pred: np.ndarray) -> float:
        return float(np.mean(self.labels(pred) == y))

    def labels(self, pr: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.classes_, dtype=int)[np.argmax(pr, axis=1)]

    def proba_full(self, X: np.ndarray) -> np.ndarray:
        return self.expand(np.asarray(self.model.predict_proba(self.pool(X), ntree_end=self.stage or 0)))

    def expand(self, pr: np.ndarray) -> np.ndarray:
        k = len(self.prep.classes)
        if pr.shape[1] == k:
            return pr
        out = np.zeros((pr.shape[0], k))
        for j, c in enumerate(self.model.classes_):
            out[:, int(c)] = pr[:, j]
        return out

    def visualize(self, grid: Grid) -> dict[str, Any]:
        k = len(self.prep.classes)
        pr = self.expand(np.asarray(self.model.predict_proba(self.mesh_pool(grid), ntree_end=self.stage)))
        data = pr.T.reshape(k, grid.h, grid.w).astype(np.float32)
        return {"kind": "proba", "grid": {"bounds": grid.bounds, "w": grid.w, "h": grid.h, "k": k, "data": data}, "score": False}

    def point_predictions(self) -> np.ndarray | None:
        return np.asarray(self.proba_full(self.prep.X), dtype=np.float32)

    def predict_row(self, row: int) -> Any:
        return int(np.ravel(self.model.predict(self.pool(self.prep.X[row:row + 1])))[0])

    def final(self, row: int, score: np.ndarray) -> dict[str, Any]:
        p = self.prep
        k = len(p.classes)
        if score.size == 1:  # binary: raw score is the log-odds of the second class
            p1 = float(sigmoid(score.reshape(1))[0])
            proba = [1 - p1, p1]
        else:
            proba = softmax(score.reshape(1, -1))[0].tolist()
        full = np.zeros(k)
        for j, c in enumerate(self.model.classes_):
            full[int(c)] = proba[j]
        return {"proba": full.tolist(), "pred": int(np.argmax(full)), "binary": k == 2, "classes": p.classes,
                "true": int(p.y[row]) if p.y is not None else None, "task": "classification"}


class CatBoostRegressorA(_CatBoost):
    name = "CatBoostRegressor"
    task = "regression"
    loss_kind = "boost"
    advanced = _advanced(False)

    def loss_metric(self) -> str:
        return "RMSE"

    def pred_type(self) -> str:
        return "RawFormulaVal"

    def metric(self, y: np.ndarray, pred: np.ndarray) -> float:
        pred = np.ravel(pred)
        ss = float(np.sum((y - np.mean(y)) ** 2))
        return float(1 - np.sum((y - pred) ** 2) / ss) if ss > 0 else 0.0

    def visualize(self, grid: Grid) -> dict[str, Any]:
        p = self.prep
        assert p.y is not None
        if p.plane.mode == "xy":
            xs = np.linspace(grid.x0, grid.x1, max(64, grid.w))
            feats = p.plane.to_features(np.c_[xs, np.zeros_like(xs)])
            ys = self.model.predict(self.pool(feats), ntree_end=self.stage)
            return {"kind": "curve", "x": xs.astype(np.float32), "y": np.asarray(ys, dtype=np.float32)}
        v = np.asarray(self.model.predict(self.mesh_pool(grid), ntree_end=self.stage), dtype=np.float64)
        return {"kind": "value", "grid": {"bounds": grid.bounds, "w": grid.w, "h": grid.h, "k": 1,
                                          "data": v.reshape(1, grid.h, grid.w).astype(np.float32)},
                "range": [float(np.min(p.y)), float(np.max(p.y))]}

    def point_predictions(self) -> np.ndarray | None:
        return np.asarray(self.model.predict(self.pool(self.prep.X)), dtype=np.float32)

    def predict_row(self, row: int) -> Any:
        return float(np.ravel(self.model.predict(self.pool(self.prep.X[row:row + 1])))[0])

    def final(self, row: int, score: np.ndarray) -> dict[str, Any]:
        p = self.prep
        return {"value": float(score[0]), "true": float(p.y[row]) if p.y is not None else None, "task": "regression",
                "target": p.target_name}


ADAPTERS: list[type[Adapter]] = [CatBoostClassifierA, CatBoostRegressorA]
