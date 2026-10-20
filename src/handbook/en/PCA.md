Principal component analysis (PCA) finds new axes for the data — the directions along which it varies most — and keeps only the first few. It is a linear way to compress many correlated features into a handful while losing as little spread as possible, and the standard way to look at data with more than two features. Sightline uses it for its 2-D plots of wider data too (the header then says "PCA view"). How to read the plots: [the intro](#hb/intro/where).

## The idea {#idea}

Photograph a long pencil lying on a table. From directly above you see its full length; from its end you see only a dot. The best single photo is the one taken from the side that shows the most extent. PCA finds that best "camera direction" for your data, then the best direction perpendicular to it, and so on. If the first two directions already show 95 % of the spread, a 2-D picture hides very little.

## How it works, step by step {#steps}

1. Centre the data: subtract each feature's mean (scikit-learn does not scale; standardise first if units differ).
2. Compute the covariance matrix of the features (or equivalently a singular value decomposition of the centred data).
3. Its eigenvectors are the principal components (new axes), its eigenvalues the variance along each; sort them from largest to smallest.
4. Keep the first `n_components` axes. The share of each eigenvalue in the total is the *explained variance ratio*.
5. Transform: each point's new coordinates are the dot products of its centred features with the kept axes. `inverse_transform` maps back, losing only what the dropped axes carried.

**Worked example.** Five points: $(1, 2), (2, 3), (3, 5), (4, 4), (5, 6)$.

- Mean: $(3, 4)$. Centred: $(-2, -2), (-1, -1), (0, 1), (1, 0), (2, 2)$.
- Covariance (dividing by $n - 1 = 4$): $\operatorname{Var}(x_1) = \frac{4 + 1 + 0 + 1 + 4}{4} = 2.5$, $\operatorname{Var}(x_2) = 2.5$, $\operatorname{Cov} = \frac{4 + 1 + 0 + 0 + 4}{4} = 2.25$:
$$C = \begin{pmatrix} 2.5 & 2.25 \\ 2.25 & 2.5 \end{pmatrix}$$
- Eigenvalues: $2.5 \pm 2.25$, i.e. $\lambda_1 = 4.75$, $\lambda_2 = 0.25$, with axes $\mathbf{u}_1 = (0.707, 0.707)$ (the diagonal) and $\mathbf{u}_2 = (0.707, -0.707)$.
- Explained variance: $4.75 / 5 = 95\,\%$ on the first axis, $5\,\%$ on the second.
- First coordinates: $(-2, -2)\cdot\mathbf{u}_1 = -2.828$, then $-1.414,\ 0.707,\ 0.707,\ 2.828$.

One number per point keeps 95 % of the spread: the points lie almost on a diagonal line.

## The formulas {#formulas}

Covariance of the centred data $\tilde X$ ($n$ rows):

$$C = \frac{1}{n - 1}\tilde X^\top \tilde X$$

Eigen-decomposition and projection onto the first $k$ components:

$$C\,\mathbf{u}_j = \lambda_j\,\mathbf{u}_j, \qquad \mathbf{z} = U_k^\top(\mathbf{x} - \bar{\mathbf{x}})$$

Explained variance ratio of component $j$:

$$r_j = \frac{\lambda_j}{\sum_{l=1}^{d}\lambda_l}$$

Whitening (`whiten=True`) divides each coordinate by $\sqrt{\lambda_j}$, so every component has variance 1.

- $\mathbf{u}_j$ — component $j$ (unit vector), stored as a row of `components_`; $\lambda_j$ — `explained_variance_`.
- $U_k$ — the $d \times k$ matrix of the first $k$ components; $\mathbf{z}$ — the new coordinates.

## Where it works and where it fails {#where}

**Works well**

- Visualising many features in 2-D or 3-D; checking whether classes separate at all.
- Compressing correlated measurements (sensor arrays, spectra, survey answers) before another model.
- Removing noise: dropping components with tiny variance.

**Fails or misleads**

- Non-linear structure (spirals, curved surfaces): PCA only rotates and drops axes — try t-SNE, UMAP or kernel PCA.
- Large variance is not the same as important for prediction: a small component can carry the class difference.
- Features on different scales: the biggest-unit feature dominates — standardise first.
- Components mix all features, so they are hard to name; look at the loadings.

## Parameters {#params}

| parameter | what it controls | default | raise | lower |
|---|---|---|---|---|
| `n_components` | number of components kept (or a share like `0.95` of variance to keep) | `None` (all) — Sightline uses 2 for the plot | keeps more detail, less compression | more compression, more information lost |
| `whiten` | rescale components to unit variance | `False` | `True`: decorrelated, equal-variance outputs (useful before some models); loses the relative scale | — |
| `svd_solver` | algorithm used | `'auto'` | `'full'`: exact LAPACK; `'randomized'`: fast approximation for large data; `'covariance_eigh'`: fast when rows ≫ features | — |

PCA is fitted in one step; in Sightline the Loss panel shows the explained variance per component, and the plot draws the original feature directions as arrows.

## Python code {#code}

{{code}}

## Try it in the app {#try}

The button opens PCA on "Blobs". For a real reduction, generate a table with more numeric columns (Data → Generator, "+ Add column") and read in the Loss panel how much of the spread the 2-D picture keeps.

{{try}}
