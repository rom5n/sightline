`HistGradientBoostingClassifier` is [gradient boosting](#hb/GradientBoostingClassifier/idea) made fast for large data. Before training it sorts every feature into at most 255 bins; finding a split then means scanning a short histogram instead of every value. It also handles missing values natively and stops early on its own when a validation score stops improving — a guard against [overfitting](#hb/intro/overfitting). It follows the design of LightGBM.

## The idea {#idea}

To decide where to cut a list of 100 000 house prices, you do not need to try every price: sorting them into a few hundred price bands and counting what falls into each band is almost as precise and far faster. Histogram boosting does exactly this for every feature once, at the start. Each tree then only adds up a few numbers per band — the gradient and its curvature — to find the best cut.

## How it works, step by step {#steps}

1. **Bin the features** once: each feature is cut into up to `max_bins` = 255 quantile bins; missing values get their own bin.
2. Start from the log-odds of the class shares, as in classic gradient boosting.
3. For every point compute the gradient $g_i = p_i - y_i$ and the hessian $h_i = p_i(1 - p_i)$ of the log-loss.
4. Grow a tree **leaf by leaf**: for the leaf with the best possible split, sum $g$ and $h$ per bin and scan the bins to find the split with the highest *gain*. Stop at `max_leaf_nodes` (31) leaves, or at `max_depth`, or when leaves would get fewer than `min_samples_leaf` (20) points.
5. Each leaf gets the value $-\nu\, G / (H + \lambda)$, where $G$, $H$ are the sums of $g$, $h$ in the leaf.
6. Repeat for up to `max_iter` trees; with more than 10 000 rows, 10 % of the training data is held out and training stops once the validation loss has not improved for 10 iterations.

**Worked example.** The four points from the gradient boosting chapter: $x = 1, 2, 3, 4$, classes $0, 1, 1, 1$, so the start is $\ln 3 \approx 1.0986$ and $p = 0.75$ everywhere. Each value gets its own bin.

- Gradients $g = p - y$: $0.75, -0.25, -0.25, -0.25$; hessians $h = 0.1875$ each.
- Split between bin 1 and bin 2: $G_L = 0.75$, $H_L = 0.1875$; $G_R = -0.75$, $H_R = 0.5625$; in total $G = 0$, $H = 0.75$.
- Gain with $\lambda = 0$: $\frac{1}{2}\left(\frac{0.75^2}{0.1875} + \frac{0.75^2}{0.5625} - \frac{0^2}{0.75}\right) = \frac{1}{2}(3 + 1) = 2$.
- Leaf values with $\nu = 0.1$: left $-0.1 \cdot 0.75/0.1875 = -0.4$; right $-0.1 \cdot (-0.75)/0.5625 = +0.133$.

The new scores, $0.699$ and $1.232$, are exactly those of classic gradient boosting here — the difference is only *how fast* the split is found (and that the default `min_samples_leaf=20` would not even split four points).

## The formulas {#formulas}

Gradient and hessian of the log-loss at the current score $F$:

$$g_i = \sigma(F(\mathbf{x}_i)) - y_i, \qquad h_i = \sigma(F(\mathbf{x}_i))\,\big(1 - \sigma(F(\mathbf{x}_i))\big)$$

Gain of splitting a leaf into $L$ and $R$:

$$\text{gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right]$$

Leaf value and model update:

$$w_{\text{leaf}} = -\nu\,\frac{G}{H + \lambda}, \qquad F \leftarrow F + w_{\text{leaf}(\mathbf{x})}$$

- $G$, $H$ — sums of $g_i$ and $h_i$ over the points in a leaf (computed per bin, then accumulated).
- $\lambda$ — `l2_regularization`; $\nu$ — `learning_rate`.

## Where it works and where it fails {#where}

**Works well**

- Tabular data from about 10 000 rows up to millions: tens of times faster than classic gradient boosting at similar accuracy.
- Missing values: no imputation needed; each split learns which side NaNs go to.
- Categorical features via `categorical_features` (Sightline one-hot encodes instead, for consistency with the other models).

**Fails or misleads**

- Small data (hundreds of rows): binning and `min_samples_leaf=20` limit detail; classic boosting or a forest may do better.
- Early stopping on a random validation split makes results depend on `random_state`.
- Still a black-box ensemble; explain single decisions with care.

## Parameters {#params}

| parameter | what it controls | default | raise | lower |
|---|---|---|---|---|
| `max_iter` | maximum number of trees | `100` | more capacity (early stopping may stop sooner) | fewer trees, may underfit |
| `learning_rate` | shrinkage $\nu$ of each tree | `0.1` | faster fit, more overfitting risk | needs more trees, often better generalisation |
| `max_leaf_nodes` | leaves per tree (trees grow best-leaf first) | `31` | more complex trees | simpler trees |
| `max_depth` | depth limit of each tree | `None` | — | caps interactions per tree |
| `l2_regularization` | $\lambda$ in leaf values and gains | `0.0` | smaller leaf values, smoother model | — |

Sightline's Loss panel replays training and validation loss per tree and marks the early-stopping point when it happens.

## Python code {#code}

{{code}}

## Try it in the app {#try}

"Spirals" needs many small corrections: watch the validation curve flatten and see whether early stopping kicks in (it does on large datasets).

{{try}}
