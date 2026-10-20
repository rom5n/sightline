"""Unsupervised: KMeans (manual Lloyd steps, validated against KMeans.fit), DBSCAN, PCA."""
from __future__ import annotations

import time
from typing import Any, Iterator

import numpy as np
from sklearn.cluster import DBSCAN, KMeans, kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

from . import common_params as cp
from .base import Adapter, Grid, P, StepInfo


def _silhouette(X: np.ndarray, labels: np.ndarray, seed: int) -> float | None:
    mask = labels >= 0
    if mask.sum() < 3 or len(np.unique(labels[mask])) < 2 or len(np.unique(labels[mask])) >= mask.sum():
        return None
    n = int(mask.sum())
    return float(silhouette_score(X[mask], labels[mask], sample_size=min(n, 1500), random_state=seed))


def _sq_dist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    return np.asarray(((X[:, None, :] - C[None, :, :]) ** 2).sum(-1))


def region_scores(d2: np.ndarray, w: int, h: int) -> np.ndarray:
    """(m, k) squared distances -> (k, m) float32 scores whose argmax is the nearest centre.
    One-hot labels would turn into cell-sized stairs once the GPU filters the grid; a softmax
    of -d2 with a temperature of ~1.5 grid cells keeps the straight Voronoi edges sub-cell exact."""
    srt = np.sort(d2, axis=1)
    gap = (srt[:, 1] - srt[:, 0]) if d2.shape[1] > 1 else np.zeros(len(d2))
    step = np.abs(np.diff(gap.reshape(h, w), axis=1))
    t = 1.5 * float(np.median(step)) if step.size else 1.0
    t = t if t > 1e-12 else 1.0
    z = np.exp(-(d2 - srt[:, :1]) / t)
    return np.asarray((z / z.sum(axis=1, keepdims=True)).T.astype(np.float32))


class KMeansA(Adapter):
    name = "KMeans"
    family = "cluster"
    group = "unsupervised"
    task = "clustering"
    est = KMeans
    loss_kind = "inertia"
    iterative = True
    explain_family = "kmeans"
    main = [
        P("n_clusters", "int", 8, ("Number of groups", "Число групп"),
          ("How many groups to look for. KMeans always finds exactly this many, even if the data has fewer. Default is 8.",
           "Сколько групп искать. KMeans всегда находит ровно столько, даже если в данных их меньше. По умолчанию 8."),
          lo=2, hi=10, step=1),
        P("init", "enum", "k-means++", ("Starting centers", "Начальные центры"),
          ("Where centers start. k-means++ spreads them apart, so it usually converges faster.",
           "Где стартуют центры. k-means++ разносит их подальше друг от друга, поэтому обычно сходится быстрее."),
          options=["k-means++", "random"]),
        P("n_init", "mixed", "auto", ("Restarts", "Перезапуски"),
          ("Run several times from different starts and keep the best. 'auto' = 1 run with k-means++, 10 with random.",
           "Запустить несколько раз с разных стартов и оставить лучший. «auto» — 1 запуск для k-means++, 10 для random."),
          options=["auto", 1, 5, 10], lo=1, hi=20, step=1),
        cp.max_iter(300, lo=1, hi=500, hint=("Upper limit on center moves. It stops earlier once the centers stop moving.",
                                             "Верхний предел перемещений центров. Остановится раньше, когда центры перестанут двигаться.")),
    ]
    advanced = [
        P("algorithm", "enum", "lloyd", ("Algorithm", "Алгоритм"),
          ("'elkan' skips some distance checks; same result. Sightline animates the classic Lloyd steps either way.",
           "«elkan» пропускает часть вычислений расстояний; результат тот же. Sightline в любом случае показывает шаги Ллойда."),
          options=["lloyd", "elkan"], section="solver"),
        cp.tol(1e-4),
        cp.random_state(),
        cp.copy("copy_x"),
    ]

    def __init__(self, ctx: Any) -> None:
        super().__init__(ctx)
        self.centers: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.path: list[np.ndarray] = []
        self.validated: dict[str, Any] | None = None
        self.converged_at: int | None = None

    def _lloyd(self, X: np.ndarray, C0: np.ndarray, max_iter: int, tol: float) -> Iterator[tuple[np.ndarray, np.ndarray, bool]]:
        """sklearn's _kmeans_single_lloyd, step by step. Yields (centers, labels, converged)."""
        C = C0.copy()
        k = len(C)
        labels_old = np.full(len(X), -1)
        for _ in range(max_iter):
            D = _sq_dist(X, C)
            labels = np.argmin(D, axis=1)
            Cn = np.zeros_like(C)
            counts = np.bincount(labels, minlength=k)
            empty = np.nonzero(counts == 0)[0]
            if empty.size:
                far = np.argsort(-D[np.arange(len(X)), labels], kind="stable")[: empty.size]
                for e, f in zip(empty, far):
                    counts[labels[f]] -= 1
                    labels[f] = e
                    counts[e] = 1
            for j in range(k):
                Cn[j] = X[labels == j].mean(axis=0) if counts[j] else C[j]
            shift = float(((Cn - C) ** 2).sum())
            C = Cn
            strict = bool(np.array_equal(labels, labels_old))
            if strict or shift <= tol:
                if not strict:
                    labels = np.argmin(_sq_dist(X, C), axis=1)
                yield C, labels, True
                return
            labels_old = labels.copy()
            yield C, labels, False

    def _inits(self, X: np.ndarray, k: int) -> list[np.ndarray]:
        rs = check_random_state(self.sk_params()["random_state"])
        init = self.value("init")
        n_init = self.value("n_init")
        runs = (1 if init == "k-means++" else 10) if n_init == "auto" else int(n_init)
        out = []
        for _ in range(runs):
            if init == "k-means++":
                c, _ = kmeans_plusplus(X, k, random_state=rs)
            else:
                seeds = rs.choice(len(X), size=k, replace=False, p=np.full(len(X), 1 / len(X)))
                c = X[seeds]
            out.append(np.asarray(c, dtype=np.float64))
        return out

    def fit_stepwise(self) -> Iterator[StepInfo]:
        p = self.prep
        X0 = p.Xm
        mean = X0.mean(axis=0)
        X = X0 - mean
        k = int(min(self.value("n_clusters"), len(X)))
        max_iter = int(self.value("max_iter"))
        tol = float(np.mean(np.var(X, axis=0)) * float(self.value("tol")))
        inits = self._inits(X, k)
        best_i = 0
        if len(inits) > 1:
            best_inertia = np.inf
            for i, c0 in enumerate(inits):
                last = None
                for last in self._lloyd(X, c0, max_iter, tol):
                    pass
                assert last is not None
                C, lab, _ = last
                inertia = float(((X - C[lab]) ** 2).sum())
                if inertia < best_inertia:
                    best_inertia, best_i = inertia, i
        c0 = inits[best_i]
        self.path = [c0 + mean]
        it = 0
        for C, lab, conv in self._lloyd(X, c0, max_iter, tol):
            it += 1
            self.centers, self.labels = C + mean, lab
            self.path.append(C + mean)
            inertia = float(((X - C[lab]) ** 2).sum())
            sil = _silhouette(X, lab, self.ctx.seed)
            done = "converged" if conv else ("max_iter" if it == max_iter else None)
            extra: dict[str, Any] = {"restarts": len(inits), "best_run": best_i + 1}
            if done:
                self.converged_at = it if conv else None
                ref = KMeans(n_clusters=k, init=c0 + mean, n_init=1, max_iter=max_iter, tol=float(self.value("tol")),
                             algorithm="lloyd").fit(X0)
                diff = float(np.max(np.abs(ref.cluster_centers_ - self.centers)))
                same = bool(np.array_equal(ref.labels_, lab))
                self.validated = {"ok": diff < 1e-6 and same, "max_center_diff": diff, "same_labels": same,
                                  "sk_n_iter": int(ref.n_iter_), "n_iter": it}
                self.model = ref
                extra["validated"] = self.validated
            yield StepInfo(it, max_iter, inertia, None, sil, None, extra=extra, done=done)
            if done:
                return

    def visualize(self, grid: Grid) -> dict[str, Any]:
        p = self.prep
        assert self.centers is not None and self.labels is not None
        P_ = p.mesh(grid.x0, grid.x1, grid.y0, grid.y1, grid.w, grid.h)
        k = len(self.centers)
        soft = region_scores(_sq_dist(p.model_input(P_), self.centers), grid.w, grid.h)
        path = np.stack([p.model_to_plane(c) for c in self.path]).astype(np.float32)  # (iters, k, 2)
        return {"kind": "clusters", "grid": {"bounds": grid.bounds, "w": grid.w, "h": grid.h, "k": k,
                                             "data": soft.reshape(k, grid.h, grid.w)},
                "labels": self.labels.astype(np.int32), "centers": p.model_to_plane(self.centers).astype(np.float32),
                "path": path}

    def predict_row(self, row: int) -> Any:
        return int(self.model.predict(self.prep.Xm[row:row + 1])[0])

    def summary(self) -> dict[str, Any]:
        assert self.labels is not None
        return {"validated": self.validated, "converged_at": self.converged_at,
                "sizes": np.bincount(self.labels, minlength=len(self.centers) if self.centers is not None else 0).tolist()}

    def explain(self, row: int) -> dict[str, Any]:
        p = self.prep
        assert self.centers is not None and self.labels is not None
        x = p.Xm[row]
        d = np.sqrt(((self.centers - x) ** 2).sum(1))
        best = int(np.argmin(d))
        return {"distances": d.tolist(), "nearest": best, "pred": best, "label": int(self.labels[row]),
                "centers": p.model_to_plane(self.centers).tolist(), "point": [float(p.xy[row, 0]), float(p.xy[row, 1])],
                "x": [float(v) for v in p.X[row]], "features": p.feature_names, "rounds": len(self.path) - 1,
                "scaled": p.scaler is not None}


class DBSCANA(Adapter):
    name = "DBSCAN"
    family = "cluster"
    group = "unsupervised"
    task = "clustering"
    est = DBSCAN
    loss_kind = "onestep"
    step_unit = "none"
    explain_family = "dbscan"
    main = [
        P("eps", "float", 0.5, ("Neighbourhood radius", "Радиус соседства"),
          ("How close two points must be to count as neighbours.", "Насколько близко должны быть точки, чтобы считаться соседями."),
          lo=0.01, hi=5.0, log=True),
        P("min_samples", "int", 5, ("Min. neighbours", "Мин. соседей"),
          ("Points needed within the radius (itself included) to be a dense core.",
           "Сколько точек (включая саму) нужно в радиусе, чтобы стать ядром."), lo=1, hi=50, step=1),
        P("metric", "enum", "euclidean", ("Distance", "Расстояние"),
          ("How distance is measured.", "Как измеряется расстояние."),
          options=["euclidean", "manhattan", "chebyshev", "minkowski", "cosine"]),
    ]
    advanced = [
        P("algorithm", "enum", "auto", ("Search method", "Метод поиска"),
          ("How neighbours are found. Only affects speed.", "Как ищутся соседи. Влияет только на скорость."),
          options=["auto", "ball_tree", "kd_tree", "brute"], section="solver"),
        P("leaf_size", "int", 30, ("Leaf size", "Размер листа"),
          ("Tuning for tree search. Only affects speed.", "Настройка поиска по дереву. Влияет только на скорость."),
          lo=5, hi=200, step=1, section="solver"),
        P("p", "float", None, ("Minkowski power", "Степень Минковского"),
          ("Power for the minkowski distance. None = 2.", "Степень для расстояния Минковского. None — 2."),
          lo=1, hi=5, step=0.5, none_label=("Default (2)", "По умолчанию (2)"), section="solver",
          enabled_if={"metric": ["minkowski"]}, only_tag=("metric='minkowski' only", "только metric='minkowski'")),
        P("metric_params", "fixed", None, ("Metric options", "Параметры метрики"),
          ("Extra arguments for the distance function.", "Дополнительные аргументы функции расстояния."), section="solver",
          disabled=("Not needed for the metrics offered here", "Не нужны для предложенных метрик")),
    ]

    def fit_stepwise(self) -> Iterator[StepInfo]:
        p = self.prep
        t = time.perf_counter()
        self.model = self.build().fit(p.Xm)
        self.t_fit = (time.perf_counter() - t) * 1000
        lab = self.model.labels_
        sil = _silhouette(p.Xm, lab, self.ctx.seed)
        yield StepInfo(1, 1, None, None, sil, None, extra=self.summary(), done="fitted")

    def visualize(self, grid: Grid) -> dict[str, Any]:
        core = np.zeros(len(self.prep.ids), np.uint8)
        core[self.model.core_sample_indices_] = 1
        return {"kind": "dbscan", "labels": self.model.labels_.astype(np.int32), "core": core}

    def predict_row(self, row: int) -> Any:
        return int(self.model.labels_[row])

    def summary(self) -> dict[str, Any]:
        lab = self.model.labels_
        cl = sorted(set(lab.tolist()) - {-1})
        return {"clusters": len(cl), "noise": int((lab == -1).sum()), "core": int(len(self.model.core_sample_indices_)),
                "sizes": [int((lab == c).sum()) for c in cl], "fit_ms": self.t_fit}

    def explain(self, row: int) -> dict[str, Any]:
        p = self.prep
        m = self.model
        kw: dict[str, Any] = {"metric": m.metric}
        if m.metric == "minkowski" and m.p is not None:
            kw["p"] = m.p
        nn = NearestNeighbors(radius=m.eps, **kw).fit(p.Xm)
        dist, idx = nn.radius_neighbors(p.Xm[row:row + 1])
        idx, dist = idx[0], dist[0]
        core_set = set(m.core_sample_indices_.tolist())
        is_core = row in core_set
        label = int(m.labels_[row])
        core_nb = [int(i) for i in idx if i in core_set and i != row]
        status = "core" if is_core else ("border" if label != -1 else "noise")
        return {"eps": float(m.eps), "min_samples": int(m.min_samples), "count": int(len(idx)), "status": status,
                "pred": label, "label": label,
                "neighbors": [{"id": int(p.ids[i]), "dist": float(d), "core": bool(i in core_set),
                               "xy": [float(p.xy[i, 0]), float(p.xy[i, 1])]} for i, d in zip(idx, dist)][:200],
                "core_neighbors": [int(p.ids[i]) for i in core_nb][:20],
                "point": [float(p.xy[row, 0]), float(p.xy[row, 1])], "plane_is_model_space": p.plane.mode == "raw" and p.scaler is None}


class PCAA(Adapter):
    name = "PCA"
    family = "reduction"
    group = "unsupervised"
    task = "reduction"
    est = PCA
    loss_kind = "pca"
    step_unit = "none"
    explain_family = "pca"
    main = [
        P("n_components", "int", None, ("Components to keep", "Сколько компонент оставить"),
          ("How many new axes to keep. The plot always shows the first two.", "Сколько новых осей оставить. График всегда показывает первые две."),
          lo=1, hi=10, step=1, initial=2, none_label=("All (None)", "Все (None)")),
        P("whiten", "bool", False, ("Whiten", "Выбеливание"),
          ("Rescale each component to the same spread. Hides how much each one matters.",
           "Привести все компоненты к одному разбросу. Скрывает, насколько важна каждая."),),
        P("svd_solver", "enum", "auto", ("Solver", "Метод"),
          ("The maths used to find the axes. They all give the same answer on small data.",
           "Каким способом искать оси. На маленьких данных все дают одинаковый ответ."),
          options=["auto", "full", "covariance_eigh", "arpack", "randomized"]),
    ]
    advanced = [
        P("tol", "float", 0.0, ("Tolerance", "Допуск"), ("Only for 'arpack'.", "Только для «arpack»."),
          lo=0.0, hi=0.1, step=0.001, section="solver", enabled_if={"svd_solver": ["arpack"]}, only_tag=("arpack only", "только arpack")),
        P("iterated_power", "mixed", "auto", ("Power iterations", "Степенные итерации"), ("Only for 'randomized'.", "Только для «randomized»."),
          options=["auto"], lo=0, hi=20, step=1, section="solver", enabled_if={"svd_solver": ["randomized"]},
          only_tag=("randomized only", "только randomized")),
        P("n_oversamples", "int", 10, ("Oversamples", "Запас выборки"), ("Only for 'randomized'.", "Только для «randomized»."),
          lo=1, hi=50, step=1, section="solver", enabled_if={"svd_solver": ["randomized"]}, only_tag=("randomized only", "только randomized")),
        P("power_iteration_normalizer", "enum", "auto", ("Normalizer", "Нормализатор"), ("Only for 'randomized'.", "Только для «randomized»."),
          options=["auto", "QR", "LU", "none"], section="solver", enabled_if={"svd_solver": ["randomized"]},
          only_tag=("randomized only", "только randomized")),
        cp.random_state(),
        cp.copy("copy"),
    ]

    def sk_params(self, **override: Any) -> dict[str, Any]:
        out = super().sk_params(**override)
        d = self.prep.Xm.shape[1]
        n = out.get("n_components")
        if n is not None:
            out["n_components"] = int(min(max(1, int(n)), d))
        if out.get("svd_solver") == "arpack":
            n2 = out.get("n_components") or d - 1
            out["n_components"] = int(min(n2, d - 1))
        return out

    def fit_stepwise(self) -> Iterator[StepInfo]:
        p = self.prep
        t = time.perf_counter()
        self.model = self.build().fit(p.Xm)
        self.t_fit = (time.perf_counter() - t) * 1000
        self.full = PCA().fit(p.Xm)
        kept = float(np.sum(self.model.explained_variance_ratio_))
        yield StepInfo(1, 1, None, None, kept, None, extra=self.summary(), done="fitted")

    def visualize(self, grid: Grid) -> dict[str, Any]:
        p = self.prep
        Z = self.model.transform(p.Xm)
        xy = np.c_[Z[:, 0], Z[:, 1] if Z.shape[1] > 1 else np.zeros(len(Z))]
        x0, x1 = float(xy[:, 0].min()), float(xy[:, 0].max())
        y0, y1 = float(xy[:, 1].min()), float(xy[:, 1].max())
        pad_x, pad_y = (x1 - x0 or 1) * 0.08, (y1 - y0 or 1) * 0.08
        comps = self.model.components_[:2]
        scale = float(np.max(np.abs(xy))) * 0.8 or 1.0
        arrows = [{"name": n, "dx": float(comps[0, j] * scale), "dy": float(comps[1, j] * scale) if comps.shape[0] > 1 else 0.0}
                  for j, n in enumerate(p.feature_names)]
        return {"kind": "projection", "xy": xy.astype(np.float32), "bounds": [x0 - pad_x, x1 + pad_x, y0 - pad_y, y1 + pad_y],
                "arrows": arrows}

    def predict_row(self, row: int) -> Any:
        return self.model.transform(self.prep.Xm[row:row + 1])[0].tolist()

    def summary(self) -> dict[str, Any]:
        return {"ratio": self.full.explained_variance_ratio_.tolist(), "kept": int(self.model.n_components_),
                "kept_share": float(np.sum(self.model.explained_variance_ratio_)), "fit_ms": self.t_fit,
                "features": self.prep.feature_names}

    def explain(self, row: int) -> dict[str, Any]:
        p = self.prep
        m = self.model
        x = p.Xm[row]
        xc = x - m.mean_
        comps = m.components_
        dots = comps @ xc
        if m.whiten:
            dots = dots / np.sqrt(m.explained_variance_)
        return {"x": x.tolist(), "mean": m.mean_.tolist(), "centered": xc.tolist(), "components": comps[:2].tolist(),
                "scores": dots.tolist(), "pred": dots.tolist(), "features": p.feature_names, "whiten": bool(m.whiten),
                "ratio": m.explained_variance_ratio_.tolist()}


ADAPTERS: list[type[Adapter]] = [KMeansA, DBSCANA, PCAA]
