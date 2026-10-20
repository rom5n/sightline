DBSCAN (*density-based spatial clustering of applications with noise*) finds clusters as areas where points are packed densely, separated by sparser areas. It needs no number of clusters, finds clusters of any shape, and labels isolated points as **noise**. It is the natural partner of [k-means](#hb/KMeans/idea) for shapes k-means cannot handle. Basics: [the intro](#hb/intro/training).

## The idea {#idea}

Look at a city from a plane at night. Neighbourhoods are areas where lights are close together; a lone farmhouse light is just that — not a neighbourhood. You draw the outline of a neighbourhood by starting at a bright, dense spot and spreading as long as the next light is close enough. DBSCAN formalises "close enough" with a radius `eps` and "dense" with a minimum number of neighbours `min_samples`.

## How it works, step by step {#steps}

1. For every point count its neighbours within distance `eps` — the point itself included.
2. A point with at least `min_samples` such neighbours is a **core** point.
3. Start a cluster from any unvisited core point; add all points within `eps` of it; from every core point added, keep adding its neighbours. The cluster stops growing when no more core points are reachable.
4. Non-core points that ended up in a cluster are **border** points (they are near a core but not dense themselves).
5. Points reached by no cluster are **noise** (label −1).

**Worked example.** Points on a line: $1, 2, 3, 8, 9, 10, 20$; `eps` $= 1.5$, `min_samples` $= 3$.

| point | neighbours within 1.5 (incl. itself) | count | type | cluster |
|---|---|---|---|---|
| 1 | 1, 2 | 2 | border | 0 |
| 2 | 1, 2, 3 | 3 | **core** | 0 |
| 3 | 2, 3 | 2 | border | 0 |
| 8 | 8, 9 | 2 | border | 1 |
| 9 | 8, 9, 10 | 3 | **core** | 1 |
| 10 | 9, 10 | 2 | border | 1 |
| 20 | 20 | 1 | noise | −1 |

Two clusters grow from the cores 2 and 9; point 20 has no neighbour and stays noise. With `min_samples` $= 2$ every point except 20 would be a core; with `eps` $= 6$ the two groups would merge into one.

## The formulas {#formulas}

The $\varepsilon$-neighbourhood of point $p$:

$$N_\varepsilon(p) = \{\, q : d(p, q) \le \varepsilon \,\}$$

Core point: $|N_\varepsilon(p)| \ge \text{min\_samples}$.

A point $q$ is *density-reachable* from a core $p$ if there is a chain $p = p_1, p_2, \dots, p_m = q$ in which every $p_{i}$ (except possibly the last) is a core point and $p_{i+1} \in N_\varepsilon(p_i)$. A cluster is a maximal set of points density-reachable from one another through cores.

- $d$ — the distance (`metric`, Euclidean by default); $\varepsilon$ — `eps`.
- Border points reachable from cores of two clusters go to whichever cluster reaches them first, so their label can depend on data order.

## Where it works and where it fails {#where}

**Works well**

- Clusters of arbitrary shape: geographic hot spots, moons and rings, tracks in sensor data.
- When outliers should be flagged rather than forced into a group: fraud, faulty readings.
- You don't know how many clusters there are.

**Fails or misleads**

- Clusters with very different densities: one `eps` cannot fit both — try OPTICS or HDBSCAN.
- High-dimensional data: distances concentrate and `eps` becomes hard to choose.
- Unscaled features change what "within `eps`" means.
- No `predict()`: new points need a new fit (or a classifier trained on the labels). In Sightline, "How it decides" therefore works for training points only.

## Parameters {#params}

| parameter | what it controls | default | raise | lower |
|---|---|---|---|---|
| `eps` | neighbourhood radius | `0.5` | more points are neighbours: clusters merge, less noise | clusters break apart, more noise |
| `min_samples` | neighbours (incl. the point) needed to be a core | `5` | only dense areas form clusters, more noise | even thin areas form clusters |
| `metric` | the distance function | `'euclidean'` | e.g. `'manhattan'`, `'cosine'` (for directions) | — |

A k-distance plot (distance to the `min_samples`-th neighbour, sorted) helps to pick `eps`: choose the value near the "knee" of the curve.

## Python code {#code}

{{code}}

## Try it in the app {#try}

On "Two moons" DBSCAN finds both moons — something k-means cannot. Raise `eps` until the moons merge, lower it until they dissolve into noise (hollow rings).

{{try}}
