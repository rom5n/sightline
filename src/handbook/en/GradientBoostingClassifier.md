Gradient boosting builds a model step by step: it starts from a simple guess and adds small [decision trees](#hb/DecisionTreeClassifier/idea) one after another, each trained to correct what the model so far still gets wrong. Where a [random forest](#hb/RandomForestClassifier/idea) averages independent trees, boosting chains dependent ones. It is often the most accurate model on tabular data — and needs care against [overfitting](#hb/intro/overfitting).

## The idea {#idea}

Think of a golfer: the first shot leaves the ball 40 m short. The second shot does not start from the tee again; it only has to cover the remaining 40 m, and it is played a little carefully. Each further shot corrects what is left. Gradient boosting works the same way: the first tree gives a rough answer, each next tree is fitted to the *remaining error*, and only a fraction of its correction (the learning rate) is added, so that no single tree can ruin the result.

## How it works, step by step {#steps}

For two classes with log-loss (scikit-learn's default):

1. Start with one number for every point: the log-odds of the class shares, $F_0 = \ln\frac{p}{1 - p}$.
2. Turn the current scores into probabilities $p_i = \sigma(F(\mathbf{x}_i))$ and compute the *pseudo-residuals* $r_i = y_i - p_i$ — the negative gradient of the log-loss.
3. Fit a small regression tree (depth 3 by default) to these residuals.
4. Give each leaf one value: a Newton step, $\gamma = \sum r_i / \sum p_i(1 - p_i)$ over the points in the leaf.
5. Update the scores: $F \leftarrow F + \nu\,\gamma_{\text{leaf}(\mathbf{x})}$ with learning rate $\nu$.
6. Repeat steps 2–5 for `n_estimators` trees. In Sightline each frame shows the model after one more tree (`staged_predict_proba`).

**Worked example.** Four points: $x = 1, 2, 3, 4$ with classes $0, 1, 1, 1$; one stump; $\nu = 0.1$.

1. Class 1 share is $0.75$: $F_0 = \ln(0.75/0.25) = \ln 3 \approx 1.0986$, so every point starts with $p = 0.75$.
2. Residuals $y - p$: $-0.75,\ 0.25,\ 0.25,\ 0.25$.
3. The stump splits at $x \le 1.5$: left $\{-0.75\}$, right $\{0.25, 0.25, 0.25\}$.
4. Leaf values, with $p(1 - p) = 0.1875$ for every point: left $-0.75 / 0.1875 = -4$; right $0.75 / (3 \cdot 0.1875) = 1.333$.
5. New scores: $x = 1$: $1.0986 - 0.1 \cdot 4 = 0.6986$ → $p = 0.668$; $x = 2, 3, 4$: $1.0986 + 0.1333 = 1.2319$ → $p = 0.774$.

| $x$ | class | start $p$ | residual | leaf value | new score | new $p$ |
|---|---|---|---|---|---|---|
| 1 | 0 | 0.75 | −0.75 | −4.000 | 0.699 | 0.668 |
| 2 | 1 | 0.75 | 0.25 | 1.333 | 1.232 | 0.774 |
| 3 | 1 | 0.75 | 0.25 | 1.333 | 1.232 | 0.774 |
| 4 | 1 | 0.75 | 0.25 | 1.333 | 1.232 | 0.774 |

One tree moved every probability towards the truth by a small, controlled amount; the next tree starts from these new residuals.

## The formulas {#formulas}

Additive model after $M$ trees:

$$F_M(\mathbf{x}) = F_0 + \nu\sum_{m=1}^{M} h_m(\mathbf{x}), \qquad p(y = 1 \mid \mathbf{x}) = \sigma(F_M(\mathbf{x}))$$

Pseudo-residuals (negative gradient of the log-loss) for tree $m$:

$$r_{im} = -\frac{\partial\, \ell(y_i, F)}{\partial F}\bigg|_{F = F_{m-1}(\mathbf{x}_i)} = y_i - p_i$$

Leaf value of region $R_{jm}$ (one Newton step):

$$\gamma_{jm} = \frac{\sum_{i \in R_{jm}} r_{im}}{\sum_{i \in R_{jm}} p_i(1 - p_i)}$$

- $\nu$ — learning rate; $h_m$ — tree $m$, returning its leaf value $\gamma$; $F_0$ — log-odds of the class prior.
- With $K > 2$ classes each stage builds $K$ trees, one per class, and softmax replaces the sigmoid.

## Where it works and where it fails {#where}

**Works well**

- Structured, tabular data when accuracy matters most: credit risk, click prediction, ranking, insurance pricing.
- Mixed feature types and interactions, without scaling.

**Fails or misleads**

- Training is sequential and slower than a forest; for more than ~10 000 rows prefer [HistGradientBoosting](#hb/HistGradientBoostingClassifier/idea).
- Too many trees with a high learning rate overfit: watch the test loss in the Loss panel turn upwards.
- Noisy labels: later trees chase the noise.
- Like all tree models, no extrapolation beyond the training range.

## Parameters {#params}

| parameter | what it controls | default | raise | lower |
|---|---|---|---|---|
| `n_estimators` | number of boosting stages (trees) | `100` | lower training error; past some point the test error rises | underfits if too few |
| `learning_rate` | shrinkage $\nu$ of each tree's correction | `0.1` | faster learning, fewer trees needed, more overfitting risk | needs more trees, usually generalises better |
| `max_depth` | depth of each tree | `3` | each tree models more interactions; overfits sooner | simpler trees (1 = stumps, additive effects only) |
| `subsample` | share of rows drawn (without replacement) for each tree | `1.0` | — | below 1.0: "stochastic gradient boosting", less variance, enables out-of-bag improvement estimates |

`learning_rate` and `n_estimators` trade off: halving one usually means doubling the other.

## Python code {#code}

{{code}}

## Try it in the app {#try}

On "Concentric circles" watch the first trees carve out the inner ring and later ones polish the edge; the Loss panel shows training and validation loss per stage.

{{try}}
