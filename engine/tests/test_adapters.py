"""Every adapter fits on every compatible preset, yields well-formed frames,
and explain() agrees with predict()/predict_proba()."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sightline_engine import datasets as d
from sightline_engine.adapters import REGISTRY
from sightline_engine.adapters.base import Ctx, Grid, sigmoid, softmax
from sightline_engine.prep import prepare

CLASS_PRESETS = ["moons", "blobs", "circles", "spirals", "xor"]


def presets_for(task: str) -> list[str]:
    if task == "regression":
        return ["linear"]
    if task == "classification":
        return CLASS_PRESETS
    return CLASS_PRESETS + (["linear"] if task == "clustering" else [])


CASES = [(n, p) for n, A in REGISTRY.items() for p in presets_for(A.task)]
SMALL = {"n_samples": 160}


def fit(name: str, preset: str, params: dict[str, Any] | None = None, scale: bool = False) -> Any:
    A = REGISTRY[name]
    ds = d.dataset_from_preset(preset, SMALL)
    assert d.compat(ds, name, A.task)["ok"]
    prep = prepare(ds, A.task, scale, 42)
    a = A(Ctx(prep, params or {}, 42))
    steps = [s for s in a.fit_stepwise() if s.blocking is None]
    return a, prep, steps


def check_viz(v: dict[str, Any], prep: Any, g: Grid) -> None:
    kind = v["kind"]
    assert kind in ("proba", "value", "curve", "clusters", "dbscan", "projection")
    if kind in ("proba", "value", "clusters"):
        grid = v["grid"]
        data = grid["data"]
        assert data.dtype == np.float32
        assert data.shape == (grid["k"], g.h, g.w)
        assert np.isfinite(data).all()
        if kind == "proba":
            assert np.allclose(data.sum(0), 1.0, atol=1e-4)
    if kind == "curve":
        assert v["x"].shape == v["y"].shape
    if kind == "clusters":
        assert v["labels"].shape == (len(prep.ids),)
        assert v["path"].ndim == 3 and v["path"].shape[2] == 2
    if kind == "projection":
        assert v["xy"].shape == (len(prep.ids), 2)


@pytest.mark.parametrize("name,preset", CASES)
def test_fit_frames_explain(name: str, preset: str) -> None:
    a, prep, steps = fit(name, preset)
    assert steps, "no steps yielded"
    assert steps[-1].done is not None
    its = [s.iteration for s in steps]
    assert its == sorted(its) and its[0] >= 1
    for s in steps:
        assert s.total >= s.iteration
        for v in (s.loss, s.loss_test, s.metric_train, s.metric_test):
            assert v is None or np.isfinite(v)
    g = Grid(*prep.plane.bounds, 48, 30)
    check_viz(a.visualize(g), prep, g)
    rows = list(prep.test_idx[:6]) + list(prep.train_idx[:4]) if len(prep.test_idx) else list(range(10))
    for row in rows:
        check_explain(a, prep, int(row))


def check_explain(a: Any, prep: Any, row: int) -> None:
    e = a.explain(row)
    fam = a.explain_family
    x = prep.Xm[row:row + 1]
    m = a.model
    if fam == "catboost":
        check_catboost(a, prep, row, e)
        return
    if a.task == "classification":
        assert e["pred"] == int(m.predict(x)[0]), f"{a.name}: explain pred {e['pred']} != predict"
        if "proba" in e and hasattr(m, "predict_proba"):
            pr = m.predict_proba(x)[0]
            assert np.allclose(np.asarray(e["proba"])[list(m.classes_)], pr, atol=1e-6)
    if fam == "logistic":
        if e["binary"]:
            z = sum(t["contrib"] for t in e["terms"]) + e["intercept"]
            assert np.isclose(z, e["z"]) and np.isclose(sigmoid(np.array([z]))[0], m.predict_proba(x)[0][1])
        else:
            zs = np.array([c["z"] for c in e["per_class"]])
            assert np.allclose(softmax(zs.reshape(1, -1))[0], m.predict_proba(x)[0], atol=1e-6)
    elif fam == "sgd" and e["binary"]:
        z = sum(t["contrib"] for t in e["terms"]) + e["intercept"]
        assert np.isclose(z, m.decision_function(x)[0])
    elif fam == "linear_reg":
        y = sum(t["contrib"] for t in e["terms"]) + e["intercept"]
        assert np.isclose(y, e["prediction"]) and np.isclose(y, m.predict(x)[0])
    elif fam == "tree":
        assert e["leaf"]["node"] == int(m.apply(x)[0])
        for st in e["path"]:
            assert st["left"] == (st["value"] <= st["threshold"])
    elif fam == "forest":
        avg = np.mean([t["proba"] for t in e["trees"]], axis=0)
        assert np.allclose(avg, e["avg"], atol=1e-6)
    elif fam == "boost":
        score = np.asarray(e["base"]) + np.sum(np.asarray(e["contribs"]), axis=0)
        assert np.allclose(score, e["score"], atol=1e-6)
        dec = np.atleast_1d(np.ravel(m.decision_function(x)))
        assert np.allclose(score, dec, atol=1e-6)
    elif fam == "ada":
        assert int(np.argmax(e["share"])) == e["pred"]
    elif fam == "knn":
        assert int(np.argmax(e["votes"])) == e["pred"]
        assert len(e["neighbors"]) == m.n_neighbors
    elif fam == "svc":
        if e["binary"]:
            assert np.isclose(e["sum"] + e["intercept"], m.decision_function(x)[0], atol=1e-6)
    elif fam == "nb":
        assert np.allclose(e["posterior"], m.predict_proba(x)[0], atol=1e-6)
    elif fam == "mlp":
        assert np.allclose(np.ravel(e["layers"][-1]["act"])[-1], m.predict_proba(x)[0][-1], atol=1e-6)
    elif fam == "kmeans":
        assert e["nearest"] == int(m.predict(x)[0])
        assert e["label"] == e["nearest"]
    elif fam == "dbscan":
        core = row in set(m.core_sample_indices_.tolist())
        assert (e["status"] == "core") == core
        assert (e["status"] == "noise") == (m.labels_[row] == -1)
        assert e["count"] >= 1
        if core:
            assert e["count"] >= m.min_samples
    elif fam == "pca":
        assert np.allclose(e["scores"], m.transform(x)[0], atol=1e-8)
    assert e is not None


@pytest.mark.parametrize("name", ["LogisticRegression", "KNeighborsClassifier", "SVC", "MLPClassifier", "KMeans"])
def test_scaling_pipeline(name: str) -> None:
    a, prep, _ = fit(name, "blobs" if name == "KMeans" else "moons", scale=True)
    assert prep.scaler is not None
    assert np.allclose(prep.X_train.mean(0), 0, atol=1e-9)
    check_explain(a, prep, int(prep.ids[0]))


def test_kmeans_matches_sklearn() -> None:
    for init in ("k-means++", "random"):
        for k in (2, 4, 8):
            a, _, steps = fit("KMeans", "blobs", {"n_clusters": k, "init": init})
            v = steps[-1].extra["validated"]
            assert v["ok"], v


def test_logistic_stops_on_convergence() -> None:
    _, _, steps = fit("LogisticRegression", "blobs")
    assert steps[-1].done in ("converged", "max_iter")
    losses = [s.loss for s in steps]
    assert losses[-1] <= losses[0]


def test_tree_growth_replay() -> None:
    a, _, steps = fit("DecisionTreeClassifier", "xor")
    depths = [s.extra["depth"] for s in steps]
    assert depths == list(range(1, len(depths) + 1))
    assert depths[-1] == a.full.get_depth()


def test_hist_gb_early_stopping_point() -> None:
    a, _, steps = fit("HistGradientBoostingClassifier", "moons", {"early_stopping": True, "n_iter_no_change": 3})
    assert steps[-1].extra.get("validation_score")


def test_more_than_two_features_uses_pca_plane() -> None:
    spec = {"rows": 200, "seed": 1, "mode": "rules",
            "columns": [{"name": f"f{i}", "type": "numeric", "dist": "normal", "mean": 0, "std": 1} for i in range(4)],
            "target": {"name": "label", "source": "rule", "rule": "circle", "on": ["f0", "f1"], "noise": 0.05}}
    ds = d.generate(spec)
    prep = prepare(ds, "classification", True, 42)
    assert prep.plane.mode == "pca"
    a = REGISTRY["LogisticRegression"](Ctx(prep, {}, 42))
    list(a.fit_stepwise())
    g = Grid(*prep.plane.bounds, 32, 20)
    check_viz(a.visualize(g), prep, g)


def test_categorical_one_hot() -> None:
    tbl = {"columns": [{"name": "w", "type": "numeric"}, {"name": "color", "type": "category"},
                       {"name": "type", "type": "category"}],
           "rows": [[100 + i * 3, ["red", "green", "yellow"][i % 3], ["apple", "lemon"][(i // 2) % 2]] for i in range(40)],
           "target": "type"}
    ds = d.dataset_from_table(tbl, "fruit")
    prep = prepare(ds, "classification", False, 0)
    assert prep.X.shape[1] == 4  # w + 3 one-hot columns
    assert prep.plane.mode == "pca"


def test_compat_checks() -> None:
    reg = d.dataset_from_preset("linear", {})
    assert not d.compat(reg, "LogisticRegression", "classification")["ok"]
    cls = d.dataset_from_preset("moons", {})
    c = d.compat(cls, "LinearRegression", "regression")
    assert c["code"] == "needs_numeric_target" and {f["kind"] for f in c["fixes"]} >= {"switch_model", "preset"}
    assert d.compat(cls, "KMeans", "clustering")["ok"]


def test_stats_outliers() -> None:
    ds = d.dataset_from_preset("moons", {})
    ds.df.loc[999] = [9.0, 0.0, "A"]
    st = d.stats(ds)
    assert any(o["id"] == 999 for o in st["outliers"])
    assert st["balance"] and len(st["features"]) == 2


def check_catboost(a: Any, prep: Any, row: int, e: dict[str, Any]) -> None:
    """CatBoost reads raw columns (categories as text), so it is checked through its own Pool."""
    pool = a.pool(prep.X[row:row + 1])
    m = a.model
    score = np.asarray(e["base"]) + np.sum(np.asarray(e["contribs"]), axis=0)
    assert np.allclose(score, e["score"], atol=1e-6)
    assert np.allclose(score, np.ravel(m.predict(pool, prediction_type="RawFormulaVal")), atol=1e-6)
    assert len(e["contribs"]) == m.tree_count_
    for t in e["trees"]:
        assert len(t["levels"]) == a.value("depth") or len(t["levels"]) <= a.value("depth")
        for lv in t["levels"]:
            if lv["kind"] == "num":
                assert lv["yes"] == (lv["value"] > lv["border"]), lv
    if a.task == "classification":
        assert e["pred"] == a.predict_row(row)
        pr = np.ravel(m.predict_proba(pool))
        assert np.allclose(np.asarray(e["proba"])[list(m.classes_)], pr, atol=1e-6)
    else:
        assert np.isclose(e["value"], a.predict_row(row))


def test_catboost_reads_categories_natively() -> None:
    """A categorical column reaches CatBoost as text through cat_features, not one-hot."""
    spec = {"rows": 240, "mode": "rules", "seed": 3, "columns": [
        {"name": "x1", "type": "numeric", "dist": "uniform", "min": -3, "max": 3},
        {"name": "x2", "type": "numeric", "dist": "normal", "mean": 0, "std": 1},
        {"name": "color", "type": "category", "values": [{"v": "red", "w": 1}, {"v": "green", "w": 1}, {"v": "blue", "w": 1}]}],
        "target": {"name": "label", "classes": ["A", "B"], "rule": "line", "noise": 0.05}}
    ds = d.generate(spec)
    A = REGISTRY["CatBoostClassifier"]
    prep = prepare(ds, A.task, False, 42)
    assert prep.cat_features == [2]
    a = A(Ctx(prep, {"iterations": 40}, 42))
    steps = [s for s in a.fit_stepwise() if s.blocking is None]
    assert steps[-1].extra["cat_features"] == ["color"]
    assert a.model.get_cat_feature_indices() == [2]
    raw = prep.raw(prep.X[:5])
    assert set(raw[:, 2]) <= {"red", "green", "blue"}
    g = Grid(*prep.plane.bounds, 32, 24)
    check_viz(a.visualize(g), prep, g)
    check_explain(a, prep, int(prep.test_idx[0]))
