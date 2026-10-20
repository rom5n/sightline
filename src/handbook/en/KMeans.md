k-means splits points into $k$ groups (clusters) without any labels. It keeps $k$ centres; each point belongs to its nearest centre, and each centre sits at the mean of its points. Alternating these two rules quickly reaches a stable grouping. Because there are no labels, there is no test accuracy — the app reports the *silhouette* instead. Basics: [the intro](#hb/intro/training).

## The idea {#idea}

A city wants to open $k$ post offices so that people walk as little as possible. Put the offices anywhere to start. Every house now uses its nearest office. Then move each office to the middle of the houses that use it. Some houses now have a closer office, so they switch; move the offices again. After a few rounds nobody switches and the offices stop moving — that is the k-means solution.

## How it works, step by step {#steps}

1. **Initialise** $k$ centres. The default `'k-means++'` picks the first centre at random from the data and each next one with probability proportional to its squared distance from the centres already chosen, so they start spread out.
2. **Assign** every point to its nearest centre.
3. **Update** every centre to the mean of the points assigned to it.
4. **Repeat** steps 2–3 until the centres stop moving (within `tol`) or `max_iter` rounds are done.
5. With `n_init` > 1 the whole procedure runs several times from different starts and the run with the lowest inertia is kept. Sightline runs these Lloyd steps one per frame and draws each centre's path.

**Worked example.** Seven points: A (1, 1), B (1.5, 2), C (3, 3.5), D (5, 7), E (3.5, 5), F (4.5, 5), G (3.5, 4.5); $k = 2$; start centres $c_1 = (1, 1)$, $c_2 = (5, 7)$.

*Round 1 — assign.*

| point | distance to $c_1$ | distance to $c_2$ | nearest |
|---|---|---|---|
| A (1, 1) | 0.00 | 7.21 | $c_1$ |
| B (1.5, 2) | 1.12 | 6.10 | $c_1$ |
| C (3, 3.5) | 3.20 | 4.03 | $c_1$ |
| D (5, 7) | 7.21 | 0.00 | $c_2$ |
| E (3.5, 5) | 4.72 | 2.50 | $c_2$ |
| F (4.5, 5) | 5.32 | 2.06 | $c_2$ |
| G (3.5, 4.5) | 4.30 | 2.92 | $c_2$ |

*Round 1 — update.* $c_1 = \left(\frac{1 + 1.5 + 3}{3}, \frac{1 + 2 + 3.5}{3}\right) = (1.833, 2.167)$; $c_2 = \left(\frac{5 + 3.5 + 4.5 + 3.5}{4}, \frac{7 + 5 + 5 + 4.5}{4}\right) = (4.125, 5.375)$.

*Round 2.* With the new centres every point keeps its group, so the centres do not move: converged. The inertia (sum of squared distances to the own centre) is $10.708$.

## The formulas {#formulas}

The objective — inertia, the within-cluster sum of squares:

$$J = \sum_{i=1}^{n} \lVert \mathbf{x}_i - \boldsymbol{\mu}_{c(i)}\rVert^2$$

The two alternating steps (Lloyd's algorithm):

$$c(i) = \arg\min_{j} \lVert\mathbf{x}_i - \boldsymbol{\mu}_j\rVert^2, \qquad \boldsymbol{\mu}_j = \frac{1}{|S_j|}\sum_{i \in S_j}\mathbf{x}_i$$

Silhouette of point $i$ (the app shows the mean over all points):

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

- $\boldsymbol{\mu}_j$ — centre $j$; $c(i)$ — the cluster of point $i$; $S_j$ — the points in cluster $j$.
- $a(i)$ — mean distance from $i$ to the other points of its cluster; $b(i)$ — mean distance to the points of the nearest other cluster. $s$ ranges from −1 (wrong cluster) to 1 (well separated).
- Each step can only lower $J$, so the algorithm always stops — but possibly in a local minimum, which is why several starts help.

## Where it works and where it fails {#where}

**Works well**

- Compact, roughly round groups of similar size: customer segments, colour quantisation of images, grouping sensors or documents.
- Large data: each round is linear in the number of points.

**Fails or misleads**

- You must choose $k$. Inertia always falls with larger $k$; look for an elbow or compare silhouettes (see the code).
- Elongated, curved or nested clusters (two moons, rings): k-means cuts them with straight borders — use [DBSCAN](#hb/DBSCAN/idea).
- Clusters of very different size or density; outliers pull centres.
- Unscaled features distort the distances.

## Parameters {#params}

| parameter | what it controls | default | raise | lower |
|---|---|---|---|---|
| `n_clusters` | number of clusters $k$ | `8` | finer split, lower inertia, possibly splitting real groups | coarser groups, may merge real ones |
| `init` | how the first centres are placed | `'k-means++'` | `'random'`: $k$ random points, more likely to need several restarts | — |
| `n_init` | number of complete restarts; the best is kept | `'auto'` (1 for k-means++, 10 for random) | more robust result, slower | faster, more sensitive to the start |
| `max_iter` | maximum rounds per run | `300` | — | may stop before converging |

## Python code {#code}

{{code}}

## Try it in the app {#try}

On "Blobs" watch the centres glide along their dashed paths and points change colour as they switch groups. Set `n_clusters` to the true number of blobs and compare the silhouette with other values.

{{try}}
