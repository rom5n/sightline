"""The server process uses NumPy ports instead of importing scikit-learn; they must match it exactly."""
import numpy as np
import pytest
from sklearn import datasets as skd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sightline_engine import lite


@pytest.mark.parametrize("n", [20, 301, 5000])
@pytest.mark.parametrize("seed", [0, 7])
def test_generators_match_sklearn(n: int, seed: int) -> None:
    pairs = [
        (lite.make_moons(n, 0.2, seed), skd.make_moons(n_samples=n, noise=0.2, random_state=seed)),
        (lite.make_circles(n, 0.1, 0.5, seed), skd.make_circles(n_samples=n, noise=0.1, factor=0.5, random_state=seed)),
        (lite.make_blobs(n, 4, 1.3, seed), skd.make_blobs(n_samples=n, centers=4, cluster_std=1.3, random_state=seed)),
        (lite.make_regression(n, 1, 12.0, seed), skd.make_regression(n_samples=n, n_features=1, noise=12.0, random_state=seed)),
    ]
    for (a, b), (c, d) in pairs:
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)


@pytest.mark.parametrize("n", [12, 99, 300, 5000])
@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_split_matches_sklearn(n: int, seed: int) -> None:
    rng = np.random.default_rng(n)
    y = rng.integers(0, 3, n)
    y[:6] = [0, 0, 1, 1, 2, 2]
    for strat in (None, y):
        tr, te = lite.train_test_split_idx(n, 0.25, seed, strat)
        tr2, te2 = train_test_split(np.arange(n), test_size=0.25, random_state=seed, stratify=strat)
        np.testing.assert_array_equal(tr, tr2)
        np.testing.assert_array_equal(te, te2)


def test_split_too_small_raises_like_sklearn() -> None:
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    with pytest.raises(ValueError):
        train_test_split(np.arange(8), test_size=0.25, random_state=0, stratify=y)
    with pytest.raises(ValueError):
        lite.train_test_split_idx(8, 0.25, 0, y)


def test_scaler_and_pca_match_sklearn() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 5)) * [1, 10, 0.1, 5, 2] + [0, 3, -1, 100, 0]
    X[:, 2] = 4.0  # constant column
    s, s2 = lite.Scaler(X), StandardScaler().fit(X)
    np.testing.assert_allclose(s.transform(X), s2.transform(X), atol=1e-12)
    p, p2 = lite.PCA2(s.transform(X)), PCA(n_components=2, random_state=0).fit(s2.transform(X))
    np.testing.assert_allclose(p.transform(s.transform(X)), p2.transform(s2.transform(X)), atol=1e-8)
    np.testing.assert_allclose(p.explained_variance_ratio_, p2.explained_variance_ratio_, atol=1e-10)
    Z = p.transform(s.transform(X))
    np.testing.assert_allclose(p.inverse_transform(Z), p2.inverse_transform(Z), atol=1e-8)
