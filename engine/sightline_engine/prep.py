"""Turns a Dataset into model-ready arrays: one-hot encoding, optional scaling,
fixed stratified train/test split, and the 2D plane the UI draws on."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from .lite import PCA2, Scaler, train_test_split_idx

if TYPE_CHECKING:  # the training worker imports this module too; keep pandas out of it
    from .datasets import Dataset

ModelTask = Literal["classification", "regression", "clustering", "reduction"]

PlaneMode = Literal["raw", "pca", "xy", "line"]


@dataclass
class Plane:
    mode: PlaneMode
    axes: tuple[str, str]
    bounds: tuple[float, float, float, float]
    editable: bool
    mean: np.ndarray | None = None  # for pca mode: standardisation of encoded features
    std: np.ndarray | None = None
    pca: PCA2 | None = None
    explained: tuple[float, float] | None = None

    def to_features(self, P: np.ndarray) -> np.ndarray:
        """Plane coordinates (m,2) -> encoded raw features (m,d)."""
        if self.mode == "raw":
            return P.astype(np.float64)
        if self.mode in ("xy", "line"):
            return P[:, :1].astype(np.float64)
        assert self.pca is not None and self.mean is not None and self.std is not None
        Z = self.pca.inverse_transform(P.astype(np.float64))
        return np.asarray(Z * self.std + self.mean)

    def from_features(self, X: np.ndarray) -> np.ndarray:
        """Encoded raw features (m,d) -> plane coordinates (m,2)."""
        if self.mode == "raw":
            return X[:, :2].astype(np.float64)
        if self.mode in ("xy", "line"):
            return np.asarray(np.c_[X[:, 0], np.zeros(len(X))])
        assert self.pca is not None and self.mean is not None and self.std is not None
        return np.asarray(self.pca.transform((X - self.mean) / self.std))

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode, "axes": list(self.axes), "bounds": list(self.bounds),
            "editable": self.editable,
            "explained": list(self.explained) if self.explained else None,
        }


@dataclass
class Column:
    """One original dataset column and where it lives in the encoded matrix."""

    name: str
    kind: Literal["numeric", "category"]
    idx: list[int]  # numeric: [i]; category: its one-hot columns
    cats: list[str]  # category values in one-hot order (empty for numeric)


@dataclass
class Prepared:
    task: ModelTask
    ids: np.ndarray
    X: np.ndarray  # encoded, unscaled
    Xm: np.ndarray  # model input (scaled if requested)
    y: np.ndarray | None  # class index (int) or target (float)
    classes: list[str]
    feature_names: list[str]
    train_idx: np.ndarray
    test_idx: np.ndarray
    scaler: Scaler | None
    plane: Plane
    xy: np.ndarray
    target_name: str | None
    columns: list[Column] = field(default_factory=list)

    @property
    def cat_features(self) -> list[int]:
        """Positions of categorical columns in `raw()` output (CatBoost's cat_features)."""
        return [i for i, c in enumerate(self.columns) if c.kind == "category"]

    def raw(self, X: np.ndarray) -> np.ndarray:
        """Encoded, unscaled rows -> original columns (object array): numbers stay numbers, each
        one-hot group turns back into its category (argmax, so mesh points between categories
        get the nearest one). Lets CatBoost read categories natively instead of one-hot."""
        out = np.empty((X.shape[0], len(self.columns)), dtype=object)
        for j, c in enumerate(self.columns):
            if c.kind == "numeric":
                out[:, j] = X[:, c.idx[0]].astype(np.float64)
            else:
                out[:, j] = np.asarray(c.cats, dtype=object)[np.argmax(X[:, c.idx], axis=1)]
        return out

    @property
    def X_train(self) -> np.ndarray:
        return np.asarray(self.Xm[self.train_idx])

    @property
    def X_test(self) -> np.ndarray:
        return np.asarray(self.Xm[self.test_idx])

    @property
    def y_train(self) -> np.ndarray:
        assert self.y is not None
        return np.asarray(self.y[self.train_idx])

    @property
    def y_test(self) -> np.ndarray:
        assert self.y is not None
        return np.asarray(self.y[self.test_idx])

    def model_input(self, P: np.ndarray) -> np.ndarray:
        X = self.plane.to_features(P)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return np.asarray(X)

    def model_to_plane(self, Xm: np.ndarray) -> np.ndarray:
        X = self.scaler.inverse_transform(Xm) if self.scaler is not None else Xm
        return self.plane.from_features(np.asarray(X))

    def row_of(self, point_id: int) -> int:
        hits = np.nonzero(self.ids == point_id)[0]
        if hits.size == 0:
            raise KeyError(point_id)
        return int(hits[0])

    def mesh(self, x0: float, x1: float, y0: float, y1: float, w: int, h: int) -> np.ndarray:
        xs = x0 + (np.arange(w) + 0.5) / w * (x1 - x0)
        ys = y0 + (np.arange(h) + 0.5) / h * (y1 - y0)
        gx, gy = np.meshgrid(xs, ys)  # row j = ys[j] (bottom to top)
        return np.asarray(np.c_[gx.ravel(), gy.ravel()])


def encode(ds: "Dataset") -> tuple[np.ndarray, list[str], list[Column]]:
    import pandas as pd

    feats = ds.feature_columns
    parts: list[pd.DataFrame] = []
    cols: list[Column] = []
    at = 0
    for c in feats:
        if c.type == "numeric":
            parts.append(ds.df[[c.name]].astype(float))
            cols.append(Column(c.name, "numeric", [at], []))
            at += 1
        else:
            vals = ds.df[c.name].astype(str)
            d = pd.get_dummies(vals, prefix=c.name, prefix_sep="=", dtype=float)
            parts.append(d)
            cats = [str(n)[len(c.name) + 1:] for n in d.columns]
            cols.append(Column(c.name, "category", list(range(at, at + d.shape[1])), cats))
            at += d.shape[1]
    enc = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=ds.df.index)
    return enc.to_numpy(dtype=np.float64), [str(c) for c in enc.columns], cols


def _pad(lo: float, hi: float, frac: float = 0.08) -> tuple[float, float]:
    span = hi - lo
    if span < 1e-9:
        span = 1.0
    return lo - span * frac, hi + span * frac


def prepare(ds: "Dataset", task: ModelTask, scale: bool, split_seed: int) -> Prepared:
    X, names, columns = encode(ds)
    ids = ds.df.index.to_numpy().astype(np.int64)
    n = len(ids)
    classes: list[str] = []
    y: np.ndarray | None = None
    if task == "classification" and ds.target is not None:
        classes = ds.classes()
        lut = {c: i for i, c in enumerate(classes)}
        y = np.array([lut[str(v)] for v in ds.df[ds.target]], dtype=np.int64)
    elif task == "regression" and ds.target is not None:
        y = ds.df[ds.target].astype(float).to_numpy()

    if task in ("classification", "regression") and y is not None and n >= 8:
        strat = y if task == "classification" else None
        tr, te = train_test_split_idx(n, 0.25, split_seed, strat)
        train_idx, test_idx = np.sort(tr), np.sort(te)
    else:
        train_idx, test_idx = np.arange(n), np.arange(0)

    scaler: Scaler | None = None
    Xm = X
    if scale and X.shape[1] > 0:
        scaler = Scaler(X[train_idx])
        Xm = scaler.transform(X)

    d = X.shape[1]
    numeric_only = all(c.type == "numeric" for c in ds.feature_columns)
    if d == 1 and task == "regression" and y is not None:
        xy = np.c_[X[:, 0], y]
        plane = Plane("xy", (names[0], ds.target or "y"), (0, 0, 0, 0), numeric_only)
    elif d == 1:
        jitter = ((ids * 0.6180339887) % 1.0) * 2 - 1
        xy = np.c_[X[:, 0], jitter]
        plane = Plane("line", (names[0], ""), (0, 0, 0, 0), False)
    elif d == 2:
        xy = X.copy()
        plane = Plane("raw", (names[0], names[1]), (0, 0, 0, 0), numeric_only)
    else:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-12] = 1.0
        pca = PCA2((X - mean) / std)
        xy = pca.transform((X - mean) / std)
        ev = pca.explained_variance_ratio_
        plane = Plane("pca", ("PC1", "PC2"), (0, 0, 0, 0), False, mean, std, pca, (float(ev[0]), float(ev[1])))
    x0, x1 = _pad(float(xy[:, 0].min()), float(xy[:, 0].max()))
    y0, y1 = _pad(float(xy[:, 1].min()), float(xy[:, 1].max()))
    if plane.mode == "line":
        y0, y1 = -1.6, 1.6
    plane.bounds = (x0, x1, y0, y1)
    return Prepared(task, ids, X, Xm, y, classes, names, train_idx, test_idx, scaler, plane,
                    xy.astype(np.float64), ds.target, columns)
