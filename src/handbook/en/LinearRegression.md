Linear regression predicts a number by adding up the features, each multiplied by its own weight. It is the simplest regression model and the yardstick for all others. New to training and loss? Start with [the intro](#hb/intro/loss).

## The idea {#idea}

A taxi fare is roughly *a fixed starting fee plus a price per kilometre*. If you only know past rides — distance and fare — you can still recover both numbers: draw the straight line that passes closest to all the points. Its slope is the price per kilometre, the point where it crosses zero distance is the starting fee. Linear regression does exactly this, for any number of features: one weight per feature (a "price") plus an intercept (the "starting fee").

## How it works, step by step {#steps}

1. Write the prediction as a weighted sum: $\hat y = w_1 x_1 + \dots + w_d x_d + b$.
2. Measure how wrong it is with the squared error, summed over all training points.
3. Choose the weights and intercept that make that sum as small as possible. Unlike most models this needs no iterations: the best weights have a closed-form solution (scikit-learn solves it with a least-squares routine).

**Worked example.** Five points: $x = 1, 2, 3, 4, 5$ and $y = 2, 4, 5, 4, 6$.

- Means: $\bar x = 3$, $\bar y = 4.2$.
- Deviations from the mean: $x - \bar x = -2, -1, 0, 1, 2$; $y - \bar y = -2.2, -0.2, 0.8, -0.2, 1.8$.
- Products summed: $S_{xy} = 4.4 + 0.2 + 0 - 0.2 + 3.6 = 8.0$. Squares of $x$ deviations summed: $S_{xx} = 4 + 1 + 0 + 1 + 4 = 10$.
- Slope $w = S_{xy}/S_{xx} = 0.8$; intercept $b = \bar y - w \bar x = 4.2 - 2.4 = 1.8$.

| $x$ | $y$ | prediction $0.8x + 1.8$ | error $y - \hat y$ | squared |
|---|---|---|---|---|
| 1 | 2 | 2.6 | −0.6 | 0.36 |
| 2 | 4 | 3.4 | 0.6 | 0.36 |
| 3 | 5 | 4.2 | 0.8 | 0.64 |
| 4 | 4 | 5.0 | −1.0 | 1.00 |
| 5 | 6 | 5.8 | 0.2 | 0.04 |

Sum of squared errors $= 2.4$, so MSE $= 0.48$. The total spread around the mean is $\sum (y - \bar y)^2 = 8.8$, so $R^2 = 1 - 2.4/8.8 \approx 0.727$: the line explains about 73 % of the variation. No other straight line gives a smaller sum than 2.4.

## The formulas {#formulas}

Prediction:

$$\hat y = \mathbf{w}^\top \mathbf{x} + b = \sum_{j=1}^{d} w_j x_j + b$$

What is minimised (ordinary least squares):

$$\min_{\mathbf{w},\, b} \; \sum_{i=1}^{n} \left(y_i - \mathbf{w}^\top \mathbf{x}_i - b\right)^2$$

Closed-form solution (with a column of ones added to $X$ for the intercept):

$$\hat{\mathbf{w}} = \left(X^\top X\right)^{-1} X^\top \mathbf{y}$$

For one feature this reduces to $w = S_{xy}/S_{xx}$ and $b = \bar y - w\bar x$, as in the example.

- $\mathbf{x}_i$ — the features of point $i$ ($d$ numbers); $y_i$ — its true value; $\hat y$ — the prediction.
- $\mathbf{w}$ — the weights, one per feature; $b$ — the intercept.
- $X$ — the $n \times d$ matrix of all training features; $\mathbf{y}$ — the vector of targets.

## Where it works and where it fails {#where}

**Works well**

- A baseline for any number-prediction task: house price from area and rooms, sales from ad spend, energy use from temperature.
- When you need to explain the model: each weight says "one more unit of this feature adds $w$ to the prediction".
- Very large data: training is a single matrix computation.

**Fails or misleads**

- Curved relationships (growth that saturates, U-shapes): the line cannot bend — add features such as $x^2$ or use a tree-based model.
- Outliers: squaring the error lets one far-away point drag the line.
- Strongly correlated features: the weights become unstable and hard to interpret — use [Ridge](#hb/Ridge/idea).
- More features than rows: the solution is not unique; use Ridge or Lasso.

## Parameters {#params}

| parameter | what it controls | default | raise / turn on | lower / turn off |
|---|---|---|---|---|
| `fit_intercept` | whether to learn the intercept $b$ | `True` | — | the line is forced through the origin $(0, 0)$; only correct when the target is 0 whenever all features are 0 |
| `positive` | forces every weight to be ≥ 0 | `False` | weights that would be negative become 0 (useful when a negative effect is impossible, e.g. amounts of ingredients) | ordinary, unconstrained weights |

Linear regression has no regularisation knob; that is what Ridge, Lasso and ElasticNet add. In Sightline it is fitted in one step, so the Loss panel shows a summary (R² and a histogram of errors) rather than a curve.

## Python code {#code}

{{code}}

## Try it in the app {#try}

The "Linear with noise" preset has one feature, so you see the fitted line directly. Drag a point far up and watch one outlier tilt the whole line — the squared error at work.

{{try}}
