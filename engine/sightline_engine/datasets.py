"""Dataset service: presets, rule-based generator, manual tables, CSV, stats, compatibility."""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from .lite import make_blobs, make_circles, make_moons, make_regression

CLASS_NAMES = ["A", "B", "C", "D", "E"]
MAX_CLASSES = 5
N_MAX = 50000
PRESET_SEED = 7

ColType = Literal["numeric", "category"]


@dataclass
class Column:
    name: str
    type: ColType
    cat_kind: Literal["text", "number"] = "text"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.type, "catKind": self.cat_kind}


@dataclass
class Dataset:
    name: str
    kind: str
    df: pd.DataFrame  # index = stable point ids
    columns: list[Column]
    target: str | None
    preset: str | None = None
    settings: dict[str, float] = field(default_factory=dict)

    @property
    def feature_columns(self) -> list[Column]:
        return [c for c in self.columns if c.name != self.target]

    @property
    def task(self) -> Literal["classification", "regression", "none"]:
        if self.target is None:
            return "none"
        col = next(c for c in self.columns if c.name == self.target)
        return "classification" if col.type == "category" else "regression"

    def classes(self) -> list[str]:
        if self.task != "classification" or self.target is None:
            return []
        vals = self.df[self.target].dropna()
        uniq = sorted({str(v) for v in vals}, key=_natural_key)
        return uniq

    def table(self) -> dict[str, Any]:
        rows: list[list[Any]] = []
        for pid, row in zip(self.df.index.tolist(), self.df.itertuples(index=False)):
            rows.append([int(pid)] + [_py(v) for v in row])
        return {"columns": [c.to_dict() for c in self.columns], "rows": rows, "target": self.target}


def _natural_key(s: str) -> tuple[int, float, str]:
    try:
        return (0, float(s), "")
    except ValueError:
        return (1, 0.0, s)


def _py(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


# ------------------------------------------------------------------ presets

PRESETS: dict[str, dict[str, Any]] = {
    "moons": {
        "task": "classification",
        "settings": {
            "n_samples": {"min": 50, "max": 2000, "step": 10, "default": 300},
            "noise": {"min": 0.0, "max": 0.5, "step": 0.01, "default": 0.15},
        },
    },
    "blobs": {
        "task": "classification",
        "settings": {
            "n_samples": {"min": 50, "max": 2000, "step": 10, "default": 400},
            "centers": {"min": 2, "max": 5, "step": 1, "default": 4},
            "cluster_std": {"min": 0.2, "max": 3.0, "step": 0.1, "default": 1.0},
        },
    },
    "circles": {
        "task": "classification",
        "settings": {
            "n_samples": {"min": 50, "max": 2000, "step": 10, "default": 400},
            "noise": {"min": 0.0, "max": 0.3, "step": 0.01, "default": 0.08},
            "factor": {"min": 0.1, "max": 0.9, "step": 0.05, "default": 0.5},
        },
    },
    "linear": {
        "task": "regression",
        "settings": {
            "n_samples": {"min": 20, "max": 2000, "step": 10, "default": 200},
            "noise": {"min": 0.0, "max": 50.0, "step": 1, "default": 15},
        },
    },
    "spirals": {
        "task": "classification",
        "settings": {
            "n_samples": {"min": 50, "max": 2000, "step": 10, "default": 400},
            "noise": {"min": 0.0, "max": 0.6, "step": 0.01, "default": 0.2},
            "arms": {"min": 2, "max": 3, "step": 1, "default": 2},
        },
    },
    "xor": {
        "task": "classification",
        "settings": {
            "n_samples": {"min": 50, "max": 2000, "step": 10, "default": 400},
            "noise": {"min": 0.0, "max": 0.3, "step": 0.01, "default": 0.05},
        },
    },
}


def preset_catalog() -> list[dict[str, Any]]:
    return [{"id": k, **v} for k, v in PRESETS.items()]


def _setting(preset: str, settings: dict[str, float], key: str) -> float:
    spec = PRESETS[preset]["settings"][key]
    v = float(settings.get(key, spec["default"]))
    return float(min(spec["max"], max(spec["min"], v)))


def make_preset_arrays(preset: str, settings: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Returns X (n,2 or n,1) and y. Same seed for every call, so ids keep their identity
    and points move smoothly when a setting changes."""
    if preset not in PRESETS:
        raise KeyError(preset)
    n = int(_setting(preset, settings, "n_samples"))
    if preset == "moons":
        X, y = make_moons(n, _setting(preset, settings, "noise"), PRESET_SEED)
        return X, y
    if preset == "blobs":
        k = int(_setting(preset, settings, "centers"))
        X, y = make_blobs(n, k, _setting(preset, settings, "cluster_std"), PRESET_SEED)
        return X, y
    if preset == "circles":
        X, y = make_circles(n, _setting(preset, settings, "noise"), _setting(preset, settings, "factor"), PRESET_SEED)
        return X, y
    if preset == "linear":
        X, yr = make_regression(n, 1, _setting(preset, settings, "noise"), PRESET_SEED)
        return X, yr
    rng = np.random.default_rng(PRESET_SEED)
    u = rng.random(N_MAX)
    e = rng.standard_normal((N_MAX, 2))
    flip = rng.random(N_MAX)
    noise = _setting(preset, settings, "noise")
    if preset == "spirals":
        arms = int(_setting(preset, settings, "arms"))
        idx = np.arange(n)
        y = idx % arms
        t = 0.25 + 0.75 * np.sqrt(u[:n])
        ang = t * 3.2 * math.pi + y * (2 * math.pi / arms)
        X = np.c_[t * np.cos(ang), t * np.sin(ang)] + e[:n] * noise * 0.25
        return X, y
    # xor
    pos = rng.random((N_MAX, 2)) * 2 - 1
    X = pos[:n]
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    flipped = flip[:n] < noise
    y = np.where(flipped, 1 - y, y)
    return X, y


def dataset_from_preset(preset: str, settings: dict[str, float]) -> Dataset:
    X, y = make_preset_arrays(preset, settings)
    task = PRESETS[preset]["task"]
    full = {k: float(settings.get(k, s["default"])) for k, s in PRESETS[preset]["settings"].items()}
    if task == "regression":
        df = pd.DataFrame({"x1": X[:, 0], "y": y})
        cols = [Column("x1", "numeric"), Column("y", "numeric")]
        return Dataset(preset, "preset", df, cols, "y", preset, full)
    df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "label": [CLASS_NAMES[int(v)] for v in y]})
    cols = [Column("x1", "numeric"), Column("x2", "numeric"), Column("label", "category")]
    return Dataset(preset, "preset", df, cols, "label", preset, full)


# ------------------------------------------------------------------ generator


def generate(spec: dict[str, Any]) -> Dataset:
    rows = int(min(N_MAX, max(10, int(spec.get("rows", 500)))))
    seed = int(spec.get("seed", 42))
    mode = spec.get("mode", "rules")
    rng = np.random.default_rng(seed)
    data: dict[str, Any] = {}
    cols: list[Column] = []
    centers: dict[str, tuple[float, float]] = {}
    for c in spec.get("columns", []):
        name = str(c.get("name") or f"col{len(cols) + 1}")
        if c.get("type", "numeric") == "numeric":
            if mode == "random":
                v = rng.standard_normal(rows)
                centers[name] = (0.0, 1.0)
            elif c.get("dist", "normal") == "uniform":
                lo, hi = float(c.get("min", -3)), float(c.get("max", 3))
                if hi < lo:
                    lo, hi = hi, lo
                v = rng.uniform(lo, hi, rows)
                centers[name] = ((lo + hi) / 2, max(1e-9, (hi - lo) / 2))
            else:
                m, s = float(c.get("mean", 0)), abs(float(c.get("std", 1)))
                v = rng.normal(m, s, rows)
                centers[name] = (m, max(1e-9, s))
            data[name] = v
            cols.append(Column(name, "numeric"))
        else:
            values = c.get("values") or [{"v": "a", "w": 1}, {"v": "b", "w": 1}]
            labels = [str(x["v"]) for x in values]
            w = np.array([max(0.0, float(x.get("w", 1))) for x in values])
            p = w / w.sum() if mode == "rules" and w.sum() > 0 else np.full(len(labels), 1 / len(labels))
            idx = rng.choice(len(labels), size=rows, p=p)
            cat_kind: Literal["text", "number"] = "number" if c.get("catKind") == "number" else "text"
            arr: list[Any] = [labels[i] for i in idx]
            if cat_kind == "number":
                arr = [_maybe_number(s) for s in arr]
            data[name] = arr
            cols.append(Column(name, "category", cat_kind))
    tgt = spec.get("target")
    target_name: str | None = None
    if tgt:
        target_name = str(tgt.get("name") or "label")
        classes = [str(x) for x in (tgt.get("classes") or ["A", "B"])][:MAX_CLASSES]
        if len(classes) < 2:
            classes = ["A", "B"]
        on = [n for n in tgt.get("on", []) if n in centers]
        nums = [c.name for c in cols if c.type == "numeric"]
        while len(on) < 2 and len(nums) > len(on):
            cand = next(n for n in nums if n not in on)
            on.append(cand)
        if mode == "random" or tgt.get("source") == "random" or len(on) < 2:
            lab = rng.integers(0, len(classes), rows)
        else:
            a = (np.asarray(data[on[0]]) - centers[on[0]][0]) / centers[on[0]][1]
            b = (np.asarray(data[on[1]]) - centers[on[1]][0]) / centers[on[1]][1]
            rule = tgt.get("rule", "line")
            if rule == "circle":
                r = np.sqrt(a * a + b * b)
                lab = (r < np.median(r)).astype(int)
            elif rule == "xor":
                lab = (a * b > 0).astype(int)
            else:
                lab = (a + b > 0).astype(int)
            noise = float(min(0.5, max(0.0, float(tgt.get("noise", 0.1)))))
            flip = rng.random(rows) < noise
            lab = np.where(flip, 1 - lab, lab)
        data[target_name] = [classes[int(i)] for i in lab]
        cols.append(Column(target_name, "category"))
    df = pd.DataFrame(data)
    return Dataset(str(spec.get("name") or "Generated"), "generator", df, cols, target_name)


def _maybe_number(s: Any) -> Any:
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return s


# ------------------------------------------------------------------ tables & CSV


def dataset_from_table(tbl: dict[str, Any], name: str | None) -> Dataset:
    cols = [
        Column(str(c["name"]), "category" if c.get("type") == "category" else "numeric",
               "number" if c.get("catKind") == "number" else "text")
        for c in tbl["columns"]
    ]
    names = [c.name for c in cols]
    if len(set(names)) != len(names):
        raise ValueError("duplicate_columns")
    ids: list[int] = []
    rows: list[list[Any]] = []
    for i, r in enumerate(tbl.get("rows", [])):
        # rows may be [id, v1, v2, ...] (as sent by us) or plain [v1, v2, ...]
        if len(r) == len(cols) + 1:
            ids.append(int(r[0]))
            rows.append(list(r[1:]))
        else:
            ids.append(i)
            rows.append(list(r))
    df = pd.DataFrame(rows, columns=names, index=pd.Index(ids))
    for c in cols:
        if c.type == "numeric":
            df[c.name] = pd.to_numeric(df[c.name], errors="coerce").astype(float)
        else:
            df[c.name] = [None if (v is None or v == "") else (_maybe_number(v) if c.cat_kind == "number" else str(v))
                          for v in df[c.name]]
    target = tbl.get("target")
    if target not in names:
        target = None
    return Dataset(name or "Table", "table", df, cols, target)


def parse_csv(text: str) -> dict[str, Any]:
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    if df.shape[1] == 0:
        raise ValueError("empty_csv")
    cols: list[dict[str, Any]] = []
    for name in df.columns:
        s = df[name]
        num = pd.to_numeric(s, errors="coerce")
        is_num = bool(num.notna().sum() == s.notna().sum() and s.notna().sum() > 0)
        cols.append({"name": str(name), "type": "numeric" if is_num else "category", "catKind": "text"})
        if is_num:
            df[name] = num
    target = cols[-1]["name"]
    last = df[target]
    if cols[-1]["type"] == "numeric" and last.dropna().nunique() <= 5 and (last.dropna() % 1 == 0).all():
        cols[-1]["type"] = "category"
        cols[-1]["catKind"] = "number"
    rows = [[_py(v) for v in r] for r in df.itertuples(index=False)]
    return {"columns": cols, "rows": rows, "target": target}


def to_csv(ds: Dataset) -> str:
    return str(ds.df.to_csv(index=False))


# ------------------------------------------------------------------ compatibility

ModelTask = Literal["classification", "regression", "clustering", "reduction"]

CLASSIFIERS_SUGGEST = "LogisticRegression"
REGRESSOR_SUGGEST = "LinearRegression"


def compat(ds: Dataset, model: str, model_task: ModelTask) -> dict[str, Any]:
    """Can `model` train on `ds`? Returns {ok, code, params, fixes}."""
    feats = ds.feature_columns
    numeric_targets = [c.name for c in ds.columns if c.type == "numeric" and c.name != ds.target]
    cat_targets = [c.name for c in ds.columns if c.type == "category" and c.name != ds.target]

    def bad(code: str, fixes: list[dict[str, Any]], **params: Any) -> dict[str, Any]:
        return {"ok": False, "code": code, "params": params, "fixes": fixes}

    if not feats:
        return bad("needs_features", [{"kind": "preset", "preset": "moons"}])
    cols = feats + ([c for c in ds.columns if c.name == ds.target] if model_task in ("classification", "regression") else [])
    missing = {c.name: int(ds.df[c.name].isna().sum()) for c in cols if ds.df[c.name].isna().any()}
    if missing:
        name, cnt = next(iter(missing.items()))
        return bad("missing_values", [{"kind": "edit_table"}], column=name, count=cnt)
    if len(ds.df) < 10:
        return bad("too_few_rows", [{"kind": "preset", "preset": "moons"}], rows=len(ds.df))
    if model_task == "reduction" and sum(1 for c in feats if c.type == "numeric") + sum(1 for c in feats if c.type == "category") < 2:
        return bad("needs_two_features", [{"kind": "preset", "preset": "blobs"}])
    if model_task == "classification":
        if ds.target is None:
            return bad("no_target", [{"kind": "change_target", "column": cat_targets[0] if cat_targets else None},
                                     {"kind": "preset", "preset": "moons"}])
        if ds.task == "regression":
            return bad("needs_category_target",
                       [{"kind": "switch_model", "model": REGRESSOR_SUGGEST},
                        {"kind": "change_target", "column": cat_targets[0] if cat_targets else None},
                        {"kind": "preset", "preset": "moons"}],
                       model=model, target=ds.target)
        classes = ds.classes()
        if len(classes) < 2:
            return bad("one_class", [{"kind": "edit_table"}, {"kind": "preset", "preset": "moons"}], target=ds.target)
        if len(classes) > MAX_CLASSES:
            return bad("too_many_classes", [{"kind": "change_target", "column": None},
                                            {"kind": "preset", "preset": "blobs"}], count=len(classes), max=MAX_CLASSES)
        counts = ds.df[ds.target].astype(str).value_counts()
        if int(counts.min()) < 2:
            return bad("class_too_small", [{"kind": "edit_table"}, {"kind": "preset", "preset": "moons"}],
                       cls=str(counts.idxmin()))
    if model_task == "regression":
        if ds.target is None:
            return bad("no_target", [{"kind": "change_target", "column": numeric_targets[0] if numeric_targets else None},
                                     {"kind": "preset", "preset": "linear"}])
        if ds.task == "classification":
            values = ds.classes()[:3]
            return bad("needs_numeric_target",
                       [{"kind": "switch_model", "model": CLASSIFIERS_SUGGEST},
                        {"kind": "change_target", "column": numeric_targets[0] if numeric_targets else None},
                        {"kind": "preset", "preset": "linear"}],
                       model=model, target=ds.target, values=values)
    return {"ok": True, "code": "ok", "params": {}, "fixes": []}


# ------------------------------------------------------------------ insights


def stats(ds: Dataset) -> dict[str, Any]:
    df = ds.df
    num = [c.name for c in ds.feature_columns if c.type == "numeric"]
    cats = [c.name for c in ds.feature_columns if c.type == "category"]
    classes = ds.classes()
    cls_idx: np.ndarray | None = None
    if classes and ds.target is not None:
        lut = {c: i for i, c in enumerate(classes)}
        cls_idx = np.array([lut.get(str(v), -1) for v in df[ds.target]])
    out: dict[str, Any] = {"features": [], "classes": classes, "n": len(df), "categorical": []}
    sub = df[num].astype(float) if num else pd.DataFrame(index=df.index)
    outlier_rows: dict[int, list[dict[str, Any]]] = {}
    for name in num:
        v = sub[name].to_numpy()
        ok = ~np.isnan(v)
        vv = v[ok]
        if vv.size == 0:
            continue
        q1, med, q3 = (float(x) for x in np.percentile(vv, [25, 50, 75]))
        iqr = q3 - q1
        lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        wlo = float(vv[vv >= lo_f].min()) if (vv >= lo_f).any() else float(vv.min())
        whi = float(vv[vv <= hi_f].max()) if (vv <= hi_f).any() else float(vv.max())
        lo_b, hi_b = float(vv.min()), float(vv.max())
        if hi_b - lo_b < 1e-12:
            hi_b = lo_b + 1.0
        bins = np.linspace(lo_b, hi_b, 17)
        hist: list[list[int]] = []
        if cls_idx is not None:
            for k in range(len(classes)):
                h, _ = np.histogram(v[ok & (cls_idx == k)], bins=bins)
                hist.append([int(x) for x in h])
        else:
            h, _ = np.histogram(vv, bins=bins)
            hist.append([int(x) for x in h])
        out["features"].append({
            "name": name, "mean": float(vv.mean()), "std": float(vv.std()), "min": float(vv.min()),
            "max": float(vv.max()), "q1": q1, "median": med, "q3": q3, "iqr": iqr,
            "whisker_lo": wlo, "whisker_hi": whi, "bins": [float(b) for b in bins], "hist": hist,
        })
        for pid, val in zip(df.index[ok].tolist(), vv.tolist()):
            if val < lo_f or val > hi_f:
                outlier_rows.setdefault(int(pid), []).append({"feature": name, "dir": "low" if val < lo_f else "high"})
    for name in cats:
        vc = df[name].astype(str).value_counts()
        out["categorical"].append({"name": name, "counts": {str(k): int(v) for k, v in vc.items()}})
    if cls_idx is not None:
        out["balance"] = [int((cls_idx == k).sum()) for k in range(len(classes))]
    pair_feats = num[:4]
    # the pair plot draws every point in every cell: keep it to a fixed-seed sample for big data
    take = np.arange(len(df))
    if len(df) > 3000:
        take = np.sort(np.random.default_rng(0).choice(len(df), 3000, replace=False))
    out["pair"] = {
        "features": pair_feats,
        "ids": df.index.to_numpy()[take].astype(np.int32),
        "values": sub[pair_feats].to_numpy(dtype=np.float32)[take] if pair_feats else np.zeros((len(take), 0), np.float32),
        "cls": (cls_idx if cls_idx is not None else np.zeros(len(df)))[take].astype(np.int32),
        "sampled": bool(len(take) < len(df)),
    }
    if len(num) >= 2:
        corr = sub[num].corr().to_numpy()
        out["corr"] = {"features": num, "matrix": [[float(x) if not math.isnan(x) else 0.0 for x in row] for row in corr]}
        stds = [f["std"] for f in out["features"]]
        imax, imin = int(np.argmax(stds)), int(np.argmin(stds))
        out["spread"] = {"wide": num[imax], "narrow": num[imin],
                         "ratio": float(stds[imax] / stds[imin]) if stds[imin] > 0 else None}
    out["outliers"] = [{"id": pid, "values": {n: _py(df.at[pid, n]) for n in pair_feats}, "why": why}
                       for pid, why in sorted(outlier_rows.items())][:50]
    out["outlier_count"] = len(outlier_rows)
    return out
