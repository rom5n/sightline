"""Adapter base: parameter schema, stepwise fitting contract, shared visual helpers."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Iterator, Literal

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, r2_score

from ..prep import Prepared

Bi = tuple[str, str]  # (en, ru)
MISSING: Any = object()


@dataclass
class P:
    """One parameter. `default` must equal sklearn's default (checked by tests)."""

    name: str
    kind: Literal["float", "int", "bool", "enum", "mixed", "layers", "priors", "fixed"]
    default: Any
    label: Bi
    hint: Bi
    lo: float | None = None
    hi: float | None = None
    log: bool = False
    step: float | None = None
    options: list[Any] | None = None  # enum literals (str/int/None/float)
    none_label: Bi | None = None  # int|None with a "No limit" toggle
    initial: Any = MISSING  # Sightline's starting value when it differs from sklearn's
    section: str | None = None
    enabled_if: dict[str, list[Any]] | None = None
    only_tag: Bi | None = None  # e.g. ("solver='sgd' only", "только при solver='sgd'")
    visible_if: dict[str, list[Any]] | None = None
    disabled: Bi | None = None  # permanently disabled, with the reason
    sk: bool = True  # real sklearn parameter
    sk_name: str | None = None  # for nested params, e.g. estimator__max_depth

    def start(self) -> Any:
        return self.default if self.initial is MISSING else self.initial

    def schema(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name, "kind": self.kind, "default": _jsonable(self.default),
            "initial": _jsonable(self.start()), "label": {"en": self.label[0], "ru": self.label[1]},
            "hint": {"en": self.hint[0], "ru": self.hint[1]},
        }
        for k in ("lo", "hi", "step", "section"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.log:
            d["log"] = True
        if self.options is not None:
            d["options"] = [_jsonable(o) for o in self.options]
        if self.none_label:
            d["noneLabel"] = {"en": self.none_label[0], "ru": self.none_label[1]}
        if self.enabled_if:
            d["enabledIf"] = {k: [_jsonable(x) for x in v] for k, v in self.enabled_if.items()}
        if self.visible_if:
            d["visibleIf"] = {k: [_jsonable(x) for x in v] for k, v in self.visible_if.items()}
        if self.only_tag:
            d["onlyTag"] = {"en": self.only_tag[0], "ru": self.only_tag[1]}
        if self.disabled:
            d["disabled"] = {"en": self.disabled[0], "ru": self.disabled[1]}
        return d


def _jsonable(v: Any) -> Any:
    if isinstance(v, tuple):
        return list(v)
    return v


@dataclass
class Grid:
    x0: float
    x1: float
    y0: float
    y1: float
    w: int
    h: int

    @property
    def bounds(self) -> list[float]:
        return [self.x0, self.x1, self.y0, self.y1]


@dataclass
class StepInfo:
    iteration: int
    total: int
    loss: float | None = None
    loss_test: float | None = None
    metric_train: float | None = None
    metric_test: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    done: str | None = None  # reason when this is the last step
    blocking: float | None = None  # progress 0..1 while inside an uninterruptible fit


@dataclass
class Ctx:
    prep: Prepared
    params: dict[str, Any]
    seed: int
    cancelled: Callable[[], bool] = lambda: False
    progress: Callable[[float], None] = lambda p: None


LossKind = Literal["loss", "inertia", "boost", "trees", "onestep", "onestep_reg", "knn", "pca", "depth"]


class Adapter:
    name: ClassVar[str]
    family: ClassVar[str]  # linear | tree | ensemble | boost | other | cluster | reduction
    group: ClassVar[str]  # home group: linear | trees | other | unsupervised
    task: ClassVar[Literal["classification", "regression", "clustering", "reduction"]]
    est: ClassVar[type]
    main: ClassVar[list[P]]
    advanced: ClassVar[list[P]]
    loss_kind: ClassVar[LossKind]
    iterative: ClassVar[bool] = False
    step_unit: ClassVar[str] = "iteration"  # iteration | epoch | level | trees | stage | none
    lock_while_training: ClassVar[bool] = False
    managed: ClassVar[set[str]] = set()  # sklearn params Sightline sets itself (warm_start ...)
    expensive: ClassVar[bool] = False
    explain_family: ClassVar[str]

    def __init__(self, ctx: Ctx) -> None:
        self.ctx = ctx
        self.model: Any = None
        self.t_fit = 0.0

    # ---------------------------------------------------------------- params

    @classmethod
    def all_params(cls) -> list[P]:
        return cls.main + cls.advanced

    @classmethod
    def schema(cls) -> dict[str, Any]:
        return {
            "name": cls.name, "family": cls.family, "group": cls.group, "task": cls.task,
            "lossKind": cls.loss_kind, "iterative": cls.iterative, "stepUnit": cls.step_unit,
            "lockWhileTraining": cls.lock_while_training, "expensive": cls.expensive,
            "explainFamily": cls.explain_family,
            "main": [p.schema() for p in cls.main],
            "advanced": [p.schema() for p in cls.advanced],
        }

    @classmethod
    def initial_params(cls) -> dict[str, Any]:
        return {p.name: _jsonable(p.start()) for p in cls.all_params()}

    def sk_params(self, **override: Any) -> dict[str, Any]:
        """Values for the sklearn constructor (only enabled, editable, real params)."""
        vals = {**self.initial_params(), **self.ctx.params}
        out: dict[str, Any] = {}
        for p in self.all_params():
            if not p.sk or p.disabled or p.sk_name:
                continue
            if p.enabled_if and not all(vals.get(k) in v for k, v in p.enabled_if.items()):
                continue
            v = vals.get(p.name, p.default)
            if p.kind == "layers" and v is not None:
                v = tuple(int(x) for x in v)
            if p.kind == "priors" and v is not None:
                v = list(v)
            out[p.name] = v
        if "random_state" in out and out["random_state"] is None:
            out["random_state"] = self.ctx.seed
        out.update(override)
        return out

    def value(self, name: str) -> Any:
        vals = {**self.initial_params(), **self.ctx.params}
        return vals.get(name)

    def build(self, **override: Any) -> Any:
        return self.est(**self.sk_params(**override))

    # ---------------------------------------------------------------- contract

    def fit_stepwise(self) -> Iterator[StepInfo]:
        raise NotImplementedError

    def visualize(self, grid: Grid) -> dict[str, Any]:
        raise NotImplementedError

    def explain(self, row: int) -> dict[str, Any]:
        raise NotImplementedError

    def summary(self) -> dict[str, Any]:
        return {}

    def predict_row(self, row: int) -> Any:
        """What `predict` says for one row — used to check explain() consistency."""
        return self.model.predict(self.ctx.prep.Xm[row:row + 1])[0]

    def probe(self, row: int) -> dict[str, Any] | None:
        return None

    def point_predictions(self) -> np.ndarray | None:
        """Per-point model output for hover tooltips (final frames only)."""
        try:
            if self.task == "classification" and hasattr(self, "proba"):
                return np.asarray(getattr(self, "proba")(self.prep.Xm), dtype=np.float32)
            if self.task == "regression":
                return np.asarray(self.model.predict(self.prep.Xm), dtype=np.float32)
        except Exception:  # noqa: BLE001 - tooltips fall back to the grid
            return None
        return None

    def grid_cost_hint(self) -> float:
        """Relative cost of one grid evaluation; runner uses measured time anyway."""
        return 1.0

    # ---------------------------------------------------------------- helpers

    @property
    def prep(self) -> Prepared:
        return self.ctx.prep

    def acc_pair(self, model: Any = None) -> tuple[float, float | None]:
        m = model if model is not None else self.model
        p = self.prep
        tr = float(accuracy_score(p.y_train, m.predict(p.X_train)))
        te = float(accuracy_score(p.y_test, m.predict(p.X_test))) if len(p.test_idx) else None
        return tr, te

    def r2_pair(self, model: Any = None) -> tuple[float, float | None]:
        m = model if model is not None else self.model
        p = self.prep
        tr = float(r2_score(p.y_train, m.predict(p.X_train)))
        te = float(r2_score(p.y_test, m.predict(p.X_test))) if len(p.test_idx) >= 2 else None
        return tr, te

    def logloss_pair(self, proba_tr: np.ndarray, proba_te: np.ndarray | None) -> tuple[float, float | None]:
        p = self.prep
        labels = list(range(len(p.classes)))
        tr = float(log_loss(p.y_train, np.clip(proba_tr, 1e-15, 1), labels=labels))
        te = float(log_loss(p.y_test, np.clip(proba_te, 1e-15, 1), labels=labels)) if proba_te is not None and len(p.test_idx) else None
        return tr, te


# ------------------------------------------------------------------ visual helpers


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return np.asarray(e / e.sum(axis=1, keepdims=True))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(z, -40, 40))))


def scores_to_proba(scores: np.ndarray, k: int) -> np.ndarray:
    """Decision scores -> pseudo-probabilities for shading (binary: sigmoid, multi: softmax)."""
    if scores.ndim == 1 or k == 2:
        s = scores.reshape(-1)
        p1 = sigmoid(s)
        return np.asarray(np.c_[1 - p1, p1])
    return softmax(scores)


def full_proba(model: Any, X: np.ndarray, k: int) -> np.ndarray:
    """predict_proba with columns for all k classes even if the model saw fewer."""
    pr = model.predict_proba(X)
    if pr.shape[1] == k:
        return np.asarray(pr)
    out = np.zeros((X.shape[0], k))
    for j, c in enumerate(model.classes_):
        out[:, int(c)] = pr[:, j]
    return out


def proba_viz(prep: Prepared, grid: Grid, fn: Callable[[np.ndarray], np.ndarray], *, score: bool = False,
              chunk: int = 8000, progress: Callable[[float], None] | None = None) -> dict[str, Any]:
    P = prep.mesh(grid.x0, grid.x1, grid.y0, grid.y1, grid.w, grid.h)
    Xm = prep.model_input(P)
    parts = []
    for i in range(0, len(Xm), chunk):
        parts.append(fn(Xm[i:i + chunk]))
        if progress:
            progress(min(1.0, (i + chunk) / len(Xm)))
    pr = np.vstack(parts)
    k = pr.shape[1]
    data = pr.T.reshape(k, grid.h, grid.w).astype(np.float32)
    return {"kind": "proba", "grid": {"bounds": grid.bounds, "w": grid.w, "h": grid.h, "k": k, "data": data},
            "score": score}


def value_viz(prep: Prepared, grid: Grid, predict: Callable[[np.ndarray], np.ndarray]) -> dict[str, Any]:
    """Regression: a curve for 1 feature, a value heatmap otherwise."""
    if prep.plane.mode == "xy":
        xs = np.linspace(grid.x0, grid.x1, max(64, grid.w))
        Xm = prep.model_input(np.c_[xs, np.zeros_like(xs)])
        ys = predict(Xm)
        return {"kind": "curve", "x": xs.astype(np.float32), "y": np.asarray(ys, dtype=np.float32)}
    P = prep.mesh(grid.x0, grid.x1, grid.y0, grid.y1, grid.w, grid.h)
    v = np.asarray(predict(prep.model_input(P)), dtype=np.float64)
    assert prep.y is not None
    lo, hi = float(np.min(prep.y)), float(np.max(prep.y))
    return {"kind": "value", "grid": {"bounds": grid.bounds, "w": grid.w, "h": grid.h, "k": 1,
                                      "data": v.reshape(1, grid.h, grid.w).astype(np.float32)},
            "range": [lo, hi]}


def finite(x: float | None) -> float | None:
    if x is None:
        return None
    return float(x) if math.isfinite(x) else None


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    @property
    def ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0
