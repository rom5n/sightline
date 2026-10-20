ElasticNet combines the two penalties of [Ridge](#hb/Ridge/idea) (L2) and [Lasso](#hb/Lasso/idea) (L1). It can set weights to exactly zero like Lasso, but keeps groups of correlated features together like Ridge. The mix is one number, `l1_ratio`. Background: [overfitting](#hb/intro/overfitting).

## The idea {#idea}

Three thermometers hang in the same room. Lasso would read one and ignore the other two, picking which one almost at random. Ridge would read all three and average them, but would also keep every useless sensor in the house. ElasticNet does both: it drops the sensors that do not help and, among the ones that do, spreads the trust across the group.

## How it works, step by step {#steps}

1. Prediction as in linear regression: $\hat y = \mathbf{w}^\top\mathbf{x} + b$.
2. Objective: half the mean squared error + $\alpha \cdot$ `l1_ratio` $\cdot$ (sum of $|w|$) + $\alpha \cdot (1 - $ `l1_ratio`$)/2 \cdot$ (sum of $w^2$).
3. Minimise with coordinate descent, one weight at a time, as in Lasso. Each update is a soft threshold followed by a ridge-style shrink.

**Worked example.** The five points $x = 1..5$, $y = 2, 4, 5, 4, 6$; $S_{xy}/n = 1.6$, $S_{xx}/n = 2$. With $\alpha = 1$ and `l1_ratio` $= 0.5$:

- L1 part: soft-threshold $1.6$ by $\alpha \cdot 0.5 = 0.5$ → $1.1$;
- L2 part: divide by $S_{xx}/n + \alpha(1 - 0.5) = 2 + 0.5 = 2.5$;
- slope $w = 1.1 / 2.5 = 0.44$, intercept $b = 4.2 - 0.44 \cdot 3 = 2.88$.

| model | slope $w$ |
|---|---|
| least squares | 0.80 |
| Ridge, $\alpha = 1$ | 0.727 |
| ElasticNet, $\alpha = 1$, `l1_ratio` $= 0.5$ | 0.44 |
| Lasso, $\alpha = 1$ | 0.30 |

## The formulas {#formulas}

What scikit-learn's `ElasticNet` minimises, with $\rho$ = `l1_ratio`:

$$\min_{\mathbf{w}, b}\; \frac{1}{2n}\lVert\mathbf{y} - X\mathbf{w} - b\rVert^2 + \alpha\rho\lVert\mathbf{w}\rVert_1 + \frac{\alpha(1 - \rho)}{2}\lVert\mathbf{w}\rVert_2^2$$

Coordinate update for one weight (features centred):

$$w_j \leftarrow \frac{\operatorname{soft}\!\left(\frac{1}{n}\sum_i x_{ij} r_{ij},\ \alpha\rho\right)}{\frac{1}{n}\sum_i x_{ij}^2 + \alpha(1 - \rho)}$$

- $\rho = 1$ gives Lasso exactly; $\rho = 0$ gives a ridge penalty (with Lasso's $\frac{1}{2n}$ scaling).
- $\operatorname{soft}(c, t) = \operatorname{sign}(c)\max(|c| - t, 0)$; $r_{ij}$ — residual without feature $j$.

## Where it works and where it fails {#where}

**Works well**

- Many features, some in correlated groups, and you want a sparse model: spectra, genomics, marketing channels that move together.
- When Lasso's choice among correlated features is unstable from one sample to another.

**Fails or misleads**

- Two knobs to tune ($\alpha$, $\rho$) — use cross-validation (`ElasticNetCV`).
- Still a linear model: curved patterns need other models.
- Unscaled features distort both penalties.

## Parameters {#params}

| parameter | what it controls | default | raise | lower |
|---|---|---|---|---|
| `alpha` | total penalty strength | `1.0` | smaller weights, more zeros, more bias | closer to least squares |
| `l1_ratio` | share of L1 in the penalty ($\rho$) | `0.5` | towards 1: more like Lasso, more exact zeros | towards 0: more like Ridge, groups kept together, fewer zeros |
| `max_iter` | maximum coordinate-descent sweeps | `1000` | allows convergence | may stop early (warning) |
| `fit_intercept` | whether to learn $b$ | `True` | — | line through the origin |

## Python code {#code}

The data has a group of three correlated useful features: as `l1_ratio` falls, the weights within the group become more equal.

{{code}}

## Try it in the app {#try}

Move `l1_ratio` between 0 and 1 at a fixed `alpha` and compare how fast the line flattens.

{{try}}
