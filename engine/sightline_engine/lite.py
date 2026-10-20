"""NumPy ports of the few scikit-learn helpers the server process needs.

The server process never trains anything; importing scikit-learn there would cost ~65 MB
(sklearn pulls in scipy.stats and pandas eagerly), on top of the training worker, which
does import it. These ports follow the installed scikit-learn code line by line and draw
from the RNG in the same order, so their output is identical; tests/test_lite.py checks
that against scikit-learn itself.
"""
from __future__ import annotations

import math

import numpy as np


def _shuffle(rng: np.random.RandomState, *arrays: np.ndarray) -> list[np.ndarray]:
    # sklearn.utils.shuffle -> resample(replace=False)
    idx = np.arange(len(arrays[0]))
    rng.shuffle(idx)
    return [a[idx] for a in arrays]


def make_moons(n_samples: int, noise: float | None, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    n_out = n_samples // 2
    n_in = n_samples - n_out
    rng = np.random.RandomState(random_state)
    X = np.vstack([
        np.append(np.cos(np.linspace(0, np.pi, n_out)), 1 - np.cos(np.linspace(0, np.pi, n_in))),
        np.append(np.sin(np.linspace(0, np.pi, n_out)), 1 - np.sin(np.linspace(0, np.pi, n_in)) - 0.5),
    ]).T
    y = np.hstack([np.zeros(n_out, dtype=np.intp), np.ones(n_in, dtype=np.intp)])
    X, y = _shuffle(rng, X, y)
    if noise is not None:
        X += rng.normal(scale=noise, size=X.shape)
    return X, y


def make_circles(n_samples: int, noise: float | None, factor: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    n_out = n_samples // 2
    n_in = n_samples - n_out
    rng = np.random.RandomState(random_state)
    lo = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    li = np.linspace(0, 2 * np.pi, n_in, endpoint=False)
    X = np.vstack([np.append(np.cos(lo), np.cos(li) * factor), np.append(np.sin(lo), np.sin(li) * factor)]).T
    y = np.hstack([np.zeros(n_out, dtype=np.intp), np.ones(n_in, dtype=np.intp)])
    X, y = _shuffle(rng, X, y)
    if noise is not None:
        X += rng.normal(scale=noise, size=X.shape)
    return X, y


def make_blobs(n_samples: int, centers: int, cluster_std: float, random_state: int,
               n_features: int = 2, center_box: tuple[float, float] = (-10.0, 10.0)) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_state)
    c = rng.uniform(center_box[0], center_box[1], size=(centers, n_features))
    per = [n_samples // centers] * centers
    for i in range(n_samples % centers):
        per[i] += 1
    X = np.empty((n_samples, n_features), dtype=np.float64)
    y = np.empty((n_samples,), dtype=int)
    start = 0
    for i, n in enumerate(per):
        X[start:start + n] = rng.normal(loc=c[i], scale=cluster_std, size=(n, n_features))
        y[start:start + n] = i
        start += n
    X, y = _shuffle(rng, X, y)
    return X, y


def make_regression(n_samples: int, n_features: int, noise: float, random_state: int,
                    n_informative: int = 10) -> tuple[np.ndarray, np.ndarray]:
    n_informative = min(n_features, n_informative)
    rng = np.random.RandomState(random_state)
    X = rng.standard_normal(size=(n_samples, n_features))
    truth = np.zeros((n_features, 1))
    truth[:n_informative, :] = 100 * rng.uniform(size=(n_informative, 1))
    y = np.dot(X, truth)
    if noise > 0.0:
        y += rng.normal(scale=noise, size=y.shape)
    X, y = _shuffle(rng, X, y)
    idx = np.arange(n_features)
    rng.shuffle(idx)
    X[:, :] = X[:, idx]
    return X, np.squeeze(y)


def _approximate_mode(counts: np.ndarray, n_draws: int, rng: np.random.RandomState) -> np.ndarray:
    continuous = counts / counts.sum() * n_draws
    floored = np.floor(continuous)
    need = int(n_draws - floored.sum())
    if need > 0:
        rem = continuous - floored
        for value in np.sort(np.unique(rem))[::-1]:
            (inds,) = np.where(rem == value)
            add = min(len(inds), need)
            inds = rng.choice(inds, size=add, replace=False)
            floored[inds] += 1
            need -= add
            if need == 0:
                break
    return np.asarray(floored.astype(int))


def train_test_split_idx(n: int, test_size: float, random_state: int,
                         stratify: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """train_test_split(np.arange(n), test_size=..., random_state=..., stratify=...)."""
    n_test = math.ceil(test_size * n)
    n_train = n - n_test
    if n_train == 0:
        raise ValueError("the train set would be empty")
    rng = np.random.RandomState(random_state)
    if stratify is None:
        perm = rng.permutation(n)
        return perm[n_test:n_test + n_train], perm[:n_test]
    classes, y_idx, counts = np.unique(stratify, return_inverse=True, return_counts=True)
    if counts.min() < 2:
        raise ValueError("The least populated class in y has only 1 member, which is too few.")
    if n_train < len(classes) or n_test < len(classes):
        raise ValueError("train and test sets must each hold at least one sample per class")
    class_indices = np.split(np.argsort(y_idx, kind="stable"), np.cumsum(counts)[:-1])
    n_i = _approximate_mode(counts, n_train, rng)
    t_i = _approximate_mode(counts - n_i, n_test, rng)
    train: list[int] = []
    test: list[int] = []
    for i in range(len(classes)):
        perm_i = class_indices[i].take(rng.permutation(counts[i]), mode="clip")
        train.extend(perm_i[:n_i[i]])
        test.extend(perm_i[n_i[i]:n_i[i] + t_i[i]])
    return rng.permutation(train), rng.permutation(test)


class Scaler:
    """StandardScaler: per-feature mean and population std (zero std -> 1)."""

    def __init__(self, X: np.ndarray) -> None:
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        # sklearn treats variances near float eps as constant features
        eps = 10 * np.finfo(np.float64).eps
        scale[scale < eps * np.maximum(1.0, np.abs(self.mean_))] = 1.0
        self.scale_ = scale

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray((X - self.mean_) / self.scale_)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X * self.scale_ + self.mean_)


class PCA2:
    """PCA(n_components=2): components with sklearn's sign convention (largest |loading| positive)."""

    def __init__(self, X: np.ndarray) -> None:
        self.mean_ = X.mean(axis=0)
        _, s, vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        signs = np.sign(vt[np.arange(vt.shape[0]), np.argmax(np.abs(vt), axis=1)])
        signs[signs == 0] = 1.0
        vt *= signs[:, None]
        var = s**2 / max(len(X) - 1, 1)
        self.components_ = vt[:2]
        total = var.sum()
        self.explained_variance_ratio_ = var[:2] / total if total > 0 else np.zeros(2)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray((X - self.mean_) @ self.components_.T)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(Z @ self.components_ + self.mean_)
