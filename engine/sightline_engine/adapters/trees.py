"""DecisionTreeClassifier with a growth replay (max_depth = 1..N)."""
from __future__ import annotations

import time
from typing import Any, Iterator

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from . import common_params as cp
from .base import Adapter, Grid, StepInfo, full_proba, proba_viz


def tree_structure(model: Any, prep: Any, max_nodes: int = 511) -> dict[str, Any]:
    """Nodes with thresholds in raw feature units (undoing the scaler when it is on)."""
    feature_names = prep.feature_names
    sc = prep.scaler
    t = model.tree_
    nodes = []
    depth = np.zeros(t.node_count, dtype=int)
    for i in range(t.node_count):
        for c in (t.children_left[i], t.children_right[i]):
            if c != -1:
                depth[c] = depth[i] + 1
    for i in range(min(t.node_count, max_nodes)):
        leaf = t.children_left[i] == -1
        counts = t.value[i][0] * t.weighted_n_node_samples[i]  # value is a fraction since sklearn 1.4
        nodes.append({
            "id": i, "depth": int(depth[i]), "leaf": bool(leaf),
            "feature": None if leaf else int(t.feature[i]),
            "name": None if leaf else feature_names[int(t.feature[i])],
            "threshold": None if leaf else _raw_thr(sc, int(t.feature[i]), float(t.threshold[i])),
            "left": None if leaf else int(t.children_left[i]), "right": None if leaf else int(t.children_right[i]),
            "n": int(t.n_node_samples[i]), "impurity": float(t.impurity[i]),
            "counts": [float(round(x, 6)) for x in counts], "pred": int(np.argmax(t.value[i][0])),
        })
    return {"nodes": nodes, "depth": int(depth.max()) if t.node_count else 0,
            "leaves": int(np.sum(t.children_left == -1)), "node_count": int(t.node_count), "truncated": t.node_count > max_nodes}


def _raw_thr(sc: Any, f: int, thr: float) -> float:
    if sc is None:
        return thr
    return float(thr * sc.scale_[f] + sc.mean_[f])


class DecisionTreeA(Adapter):
    name = "DecisionTreeClassifier"
    family = "tree"
    group = "trees"
    task = "classification"
    est = DecisionTreeClassifier
    loss_kind = "depth"
    iterative = True
    step_unit = "level"
    explain_family = "tree"
    main = [cp.max_depth(), cp.criterion(), cp.min_samples_split(), cp.min_samples_leaf()]
    advanced = [*cp.tree_shape_advanced(include_splitter=True, include_max_features=True),
                cp.class_weight("other"), cp.random_state()]

    def __init__(self, ctx: Any) -> None:
        super().__init__(ctx)
        self.full: Any = None
        self.by_depth: list[dict[str, float | None]] = []

    def proba(self, X: np.ndarray) -> np.ndarray:
        return full_proba(self.model, X, len(self.prep.classes))

    def fit_stepwise(self) -> Iterator[StepInfo]:
        p = self.prep
        t = time.perf_counter()
        self.full = self.build().fit(p.X_train, p.y_train)
        self.t_fit = (time.perf_counter() - t) * 1000
        n = max(1, int(self.full.get_depth()))
        self.by_depth = []
        for d in range(1, n + 1):
            m = self.full if d == n else self.build(max_depth=d).fit(p.X_train, p.y_train)
            self.model = m
            tr, te = self.acc_pair(m)
            self.by_depth.append({"depth": d, "train": tr, "test": te})
            yield StepInfo(d, n, None, None, tr, te,
                           extra={"depth": d, "by_depth": list(self.by_depth), "leaves": int(m.get_n_leaves()),
                                  "fit_ms": self.t_fit},
                           done="fitted" if d == n else None)

    def visualize(self, grid: Grid) -> dict[str, Any]:
        v = proba_viz(self.prep, grid, self.proba)
        v["tree"] = tree_structure(self.model, self.prep)
        return v

    def explain(self, row: int) -> dict[str, Any]:
        return explain_tree(self.model, self.prep, row)

    def summary(self) -> dict[str, Any]:
        return {"by_depth": self.by_depth, "fit_ms": self.t_fit, "leaves": int(self.model.get_n_leaves()),
                "depth": int(self.model.get_depth())}


def explain_tree(model: Any, prep: Any, row: int) -> dict[str, Any]:
    x = prep.Xm[row:row + 1]
    path = model.decision_path(x).indices.tolist()
    t = model.tree_
    steps = []
    for node in path[:-1]:
        f = int(t.feature[node])
        thr = float(t.threshold[node])
        val = float(x[0, f])
        go_left = val <= thr
        steps.append({"node": int(node), "feature": f, "name": prep.feature_names[f], "threshold": thr,
                      "threshold_raw": _raw_thr(prep.scaler, f, thr),
                      "value": val, "raw": float(prep.X[row, f]), "left": go_left,
                      "next": int(t.children_left[node] if go_left else t.children_right[node]),
                      "n": int(t.n_node_samples[node]), "impurity": float(t.impurity[node]),
                      "counts": (t.value[node][0] * t.weighted_n_node_samples[node]).round(6).tolist()})
    leaf = path[-1]
    counts = (t.value[leaf][0] * t.weighted_n_node_samples[leaf]).round(6)
    proba = full_proba(model, x, len(prep.classes))[0]
    pred = int(np.argmax(proba))
    return {"path": steps, "leaf": {"node": int(leaf), "counts": counts.tolist(), "n": int(t.n_node_samples[leaf]),
                                    "pred": pred, "proba": proba.tolist()},
            "root_n": int(t.n_node_samples[0]), "pred": pred, "true": int(prep.y[row]) if prep.y is not None else None,
            "classes": prep.classes, "tree": tree_structure(model, prep), "scaled": prep.scaler is not None,
            "point": [float(prep.xy[row, 0]), float(prep.xy[row, 1])]}


ADAPTERS: list[type[Adapter]] = [DecisionTreeA]
