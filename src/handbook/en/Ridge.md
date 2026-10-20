Ridge is [linear regression](#hb/LinearRegression/idea) with a brake on the weights. It adds the sum of squared weights to the error it minimises, so it prefers small, evenly shared weights. This makes it stable when features are correlated or plentiful. Regularisation is a defence against [overfitting](#hb/intro/overfitting).

## The idea {#idea}

Imagine three witnesses who saw the same car and give almost the same description. Plain least squares may decide to believe one witness completely and contradict the other two with large positive and negative weights — the numbers fit, but the story is fragile. Ridge charges a fee for every large weight, so the cheapest explanation is to trust the three witnesses a little each. The predictions stay almost as good, and the weights stop jumping around when the data changes slightly.

## How it works, step by step {#steps}

1. Write the prediction as in linear regression: $\hat y = \mathbf{w}^\top\mathbf{x} + b$.
2. Add a penalty $\alpha\lVert\mathbf{w}\rVert^2$ to the sum of squared errors.
3. Minimise the total. Like ordinary least squares it has a closed form, the only change is $\alpha$ added on the diagonal of $X^\top X$. The intercept is not penalised (scikit-learn centres the data first).

**Worked example.** The five points from the linear regression chapter: $x = 1..5$, $y = 2, 4, 5, 4, 6$, so $S_{xy} = 8$ and $S_{xx} = 10$.

For one centred feature the ridge slope is $w = S_{xy}/(S_{xx} + \alpha)$:

| $\alpha$ | slope $w$ | intercept $b = \bar y - w\bar x$ | training MSE |
|---|---|---|---|
| 0 (least squares) | $8/10 = 0.800$ | 1.800 | 0.480 |
| 1 (default) | $8/11 \approx 0.727$ | 2.018 | 0.491 |
| 10 | $8/20 = 0.400$ | 3.000 | 0.800 |

The training error rises a little, the slope shrinks — the price for a steadier model. It never reaches exactly zero, however large $\alpha$ gets.

## The formulas {#formulas}

What scikit-learn's `Ridge` minimises:

$$\min_{\mathbf{w}, b}\; \lVert \mathbf{y} - X\mathbf{w} - b\rVert^2 + \alpha\lVert\mathbf{w}\rVert^2$$

Closed form (with centred $X$ and $\mathbf{y}$):

$$\hat{\mathbf{w}} = \left(X^\top X + \alpha I\right)^{-1} X^\top\mathbf{y}$$

- $\alpha \ge 0$ — penalty strength; $\alpha = 0$ gives ordinary least squares.
- $\lVert\mathbf{w}\rVert^2 = \sum_j w_j^2$ — the L2 norm squared.
- $I$ — the identity matrix; adding $\alpha I$ makes $X^\top X$ invertible even when features are duplicated.

## Where it works and where it fails {#where}

**Works well**

- Many correlated features: sensor readings, economic indicators, overlapping measurements of the same thing.
- More features than rows (genes, text counts): least squares has no unique answer, ridge does.
- When you want all features kept but tamed.

**Fails or misleads**

- When you want a short list of important features: no weight becomes exactly zero — use [Lasso](#hb/Lasso/idea).
- Features on different scales: the penalty hits a feature measured in small units harder. Scale features first.
- Curved relationships: still a straight line.

## Parameters {#params}

| parameter | what it controls | default | raise | lower |
|---|---|---|---|---|
| `alpha` | strength of the L2 penalty | `1.0` | smaller, more equal weights; more bias, less variance; very large → prediction close to the mean | closer to least squares; at 0 identical to linear regression |
| `fit_intercept` | whether to learn $b$ (never penalised) | `True` | — | line through the origin |

## Python code {#code}

The second half shows two nearly identical features: least squares splits the weight as 2.82 / 0.23, ridge as 1.53 / 1.51.

{{code}}

## Try it in the app {#try}

Raise `alpha` from 1 to 1000 on "Linear with noise" and watch the line flatten towards the average value.

{{try}}
