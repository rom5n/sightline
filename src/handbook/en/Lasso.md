Lasso is [linear regression](#hb/LinearRegression/idea) with an L1 penalty — the sum of the *absolute* weights. Unlike [Ridge](#hb/Ridge/idea), this penalty pushes weak weights all the way to zero, so Lasso also picks which features to keep. It is a defence against [overfitting](#hb/intro/overfitting) that doubles as feature selection.

## The idea {#idea}

You pack for a trip with a strict fee per item, regardless of size. A small fee per kilo would make you pack lighter versions of everything (that is Ridge); a fee per *item* makes you leave whole items at home. Lasso's penalty works like the per-item fee: a feature only stays if it reduces the error by more than it costs. Features that barely help get a weight of exactly 0 and drop out of the model.

## How it works, step by step {#steps}

1. Prediction as in linear regression: $\hat y = \mathbf{w}^\top\mathbf{x} + b$.
2. Objective: half the mean squared error plus $\alpha$ times the sum of absolute weights.
3. There is no closed form, so scikit-learn uses *coordinate descent*: it updates one weight at a time, keeping the others fixed, and sweeps over all weights again and again until they stop changing (up to `max_iter` sweeps).
4. Each one-weight update has a simple solution called *soft thresholding*: compute the best weight without the penalty, then move it towards zero by a fixed amount; if it would cross zero, it becomes exactly zero.

**Worked example.** The five points $x = 1..5$, $y = 2, 4, 5, 4, 6$. With one centred feature the update is

$$w = \frac{\operatorname{soft}\!\left(S_{xy}/n,\ \alpha\right)}{S_{xx}/n}, \qquad \operatorname{soft}(c, \alpha) = \operatorname{sign}(c)\max(|c| - \alpha, 0)$$

Here $S_{xy}/n = 8/5 = 1.6$ and $S_{xx}/n = 10/5 = 2$:

| $\alpha$ | $\operatorname{soft}(1.6, \alpha)$ | slope $w$ | intercept $b$ |
|---|---|---|---|
| 0.5 | 1.1 | 0.55 | 2.55 |
| 1 (default) | 0.6 | 0.30 | 3.30 |
| 2 | 0 | **0** | 4.20 (the mean) |

At $\alpha = 2$ the feature is switched off completely — something Ridge never does.

## The formulas {#formulas}

What scikit-learn's `Lasso` minimises:

$$\min_{\mathbf{w}, b}\; \frac{1}{2n}\lVert\mathbf{y} - X\mathbf{w} - b\rVert^2 + \alpha\lVert\mathbf{w}\rVert_1, \qquad \lVert\mathbf{w}\rVert_1 = \sum_j |w_j|$$

Coordinate-descent update for weight $j$ (features centred):

$$w_j \leftarrow \frac{\operatorname{soft}\!\left(\frac{1}{n}\sum_i x_{ij}\, r_{ij},\ \alpha\right)}{\frac{1}{n}\sum_i x_{ij}^2}, \qquad r_{ij} = y_i - \sum_{k\ne j} w_k x_{ik}$$

- $n$ — number of training rows; $\alpha$ — penalty strength.
- $r_{ij}$ — the residual of row $i$ with feature $j$ left out.
- Note the $\frac{1}{2n}$: Lasso's $\alpha$ is on a different scale from Ridge's.

## Where it works and where it fails {#where}

**Works well**

- Many candidate features of which only a few matter: picking predictors of sales from dozens of indicators, selecting genes.
- When the final model must be short and readable.

**Fails or misleads**

- Groups of correlated features: Lasso keeps one more or less at random and drops the others — [ElasticNet](#hb/ElasticNet/idea) keeps the group together.
- More features than rows: at most $n$ features can be non-zero.
- Unscaled features: the penalty depends on the units — scale first.

## Parameters {#params}

| parameter | what it controls | default | raise | lower |
|---|---|---|---|---|
| `alpha` | strength of the L1 penalty | `1.0` | more weights become exactly 0; simpler model, more bias; large enough → all zero, prediction = mean | more features kept; closer to least squares (use `LinearRegression` for $\alpha = 0$) |
| `max_iter` | maximum coordinate-descent sweeps | `1000` | allows convergence on hard problems | may stop before converging (a warning; Sightline marks it) |
| `fit_intercept` | whether to learn $b$ (never penalised) | `True` | — | line through the origin |

In Sightline each frame of training is a coordinate-descent sweep, so you can watch weights reach zero one by one.

## Python code {#code}

With 10 features of which 3 are real, Lasso keeps exactly those 3.

{{code}}

## Try it in the app {#try}

Raise `alpha` on "Linear with noise" until the line goes flat: that is the moment the only weight becomes 0.

{{try}}
