"""Adapter schemas must match the installed scikit-learn: names, defaults, valid options."""
from __future__ import annotations

import numpy as np
import pytest

from sightline_engine.adapters import REGISTRY

EXCLUDED = {"verbose", "n_jobs"}  # never shown
# scikit-learn estimators; CatBoost has its own checks below (its get_params() lists only explicit args)
SK = [n for n in REGISTRY if not n.startswith("CatBoost")]
CB = [n for n in REGISTRY if n.startswith("CatBoost")]


def _norm(v: object) -> object:
    if isinstance(v, tuple):
        return list(v)
    return v


@pytest.mark.parametrize("name", SK)
def test_schema_matches_get_params(name: str) -> None:
    A = REGISTRY[name]
    sk = A.est().get_params()
    seen = set()
    for p in A.all_params():
        if not p.sk:
            assert p.sk_name, f"{name}.{p.name}: non-sklearn param needs sk_name"
            assert p.sk_name.split("__")[0] in sk
            continue
        assert p.name in sk, f"{name}.{p.name} is not a parameter of {A.est.__name__}"
        assert _norm(p.default) == _norm(sk[p.name]), f"{name}.{p.name}: default {p.default!r} != sklearn {sk[p.name]!r}"
        seen.add(p.name)
    # every sklearn parameter is either shown (main/advanced) or deliberately excluded
    missing = set(sk) - seen - EXCLUDED - {"penalty"}
    assert not missing, f"{name}: parameters not in schema: {sorted(missing)}"


@pytest.mark.parametrize("name", SK)
def test_main_panel_has_no_deprecated(name: str) -> None:
    A = REGISTRY[name]
    sk = A.est().get_params()
    for p in A.main:
        if p.sk:
            assert sk.get(p.name) != "deprecated", f"{name}.{p.name} is deprecated and must not be in the main panel"
    assert 2 <= len(A.main) <= 5


def test_logistic_penalty_is_hidden() -> None:
    names = {p.name for p in REGISTRY["LogisticRegression"].all_params()}
    assert REGISTRY["LogisticRegression"].est().get_params()["penalty"] == "deprecated"
    assert "penalty" not in names
    assert "l1_ratio" in names


@pytest.mark.parametrize("name", SK)
def test_enum_options_are_accepted(name: str) -> None:
    """Every enum option must pass sklearn's own parameter validation."""
    A = REGISTRY[name]
    for p in A.all_params():
        if not p.sk or p.disabled or p.kind not in ("enum", "mixed") or not p.options:
            continue
        for opt in p.options:
            est = A.est(**{p.name: opt})
            est._validate_params()


@pytest.mark.parametrize("name", list(REGISTRY))
def test_numeric_ranges_contain_default(name: str) -> None:
    for p in REGISTRY[name].all_params():
        v = p.start()
        if p.kind in ("float", "int") and isinstance(v, (int, float)) and not isinstance(v, bool) and p.lo is not None:
            assert p.lo <= v <= (p.hi if p.hi is not None else np.inf), f"{name}.{p.name}={v} outside [{p.lo}, {p.hi}]"


def test_twenty_two_models() -> None:
    assert len(REGISTRY) == 22
    for A in REGISTRY.values():
        for p in A.all_params():
            assert p.label[0] and p.label[1] and p.hint[0] and p.hint[1], f"{A.name}.{p.name} needs EN and RU text"


def _cb_effective(name: str, classes: int = 2, **kw: object) -> dict[str, object]:
    import catboost

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 2))
    y = (X[:, 0] > 0).astype(int) if classes == 2 else X[:, 0] * 2
    m = getattr(catboost, name)(iterations=5, logging_level="Silent", allow_writing_files=False, **kw).fit(X, y)
    return m.get_all_params()


@pytest.mark.parametrize("name", CB)
def test_catboost_schema_matches_library(name: str) -> None:
    """Every parameter is a real constructor argument, and every non-auto default is CatBoost's own
    effective default (from get_all_params on a model fitted without arguments)."""
    import inspect

    import catboost

    A = REGISTRY[name]
    est = getattr(catboost, name)
    args = set(inspect.signature(est.__init__).parameters)
    eff = _cb_effective(name, 2 if A.task == "classification" else 0)
    for p in A.all_params():
        assert p.name in args, f"{name}.{p.name} is not a CatBoost argument"
        if p.default is None:
            continue  # "Auto": CatBoost decides from the data
        cond = {k: next(v for v in vs if v is not None) for k, vs in (p.enabled_if or {}).items()}
        got = (_cb_effective(name, 2 if A.task == "classification" else 0, **cond) if cond else eff).get(p.name)
        if isinstance(p.default, float) and isinstance(got, (int, float)):
            assert np.isclose(p.default, got), f"{name}.{p.name}: {p.default} != CatBoost {got}"
        elif p.name != "iterations":
            assert p.default == got, f"{name}.{p.name}: {p.default!r} != CatBoost {got!r}"
    assert next(p for p in A.main if p.name == "iterations").default == 1000
    assert 2 <= len(A.main) <= 5


@pytest.mark.parametrize("name", CB)
def test_catboost_options_are_accepted(name: str) -> None:
    import catboost

    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 2))
    y = (X[:, 0] > 0).astype(int) if REGISTRY[name].task == "classification" else X[:, 0]
    for p in REGISTRY[name].all_params():
        if p.disabled or p.kind != "enum" or not p.options:
            continue
        for opt in p.options:
            kw = {} if opt is None else {p.name: opt}
            getattr(catboost, name)(iterations=2, logging_level="Silent", allow_writing_files=False, **kw).fit(X, y)
