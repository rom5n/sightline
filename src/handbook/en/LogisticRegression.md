Logistic regression is the standard first model for classification. It computes a weighted sum of the features, like [linear regression](#hb/LinearRegression/idea), and squeezes it into a probability between 0 and 1. Its border between classes is always a straight line (or a flat plane). See [the intro](#hb/intro/loss) for what log-loss is.

## The idea {#idea}

A bank scores a loan application: income adds points, existing debt removes points, a long job history adds a few more. The total score is not yet a decision; it is turned into "chance of repaying" with an S-shaped curve: very negative scores give a chance close to 0, very positive ones close to 1, and a score of exactly 0 means 50/50. Logistic regression learns the points for each feature (the weights) so that these chances match what really happened in past applications.

## How it works, step by step {#steps}

1. Score each point: $z = w_1x_1 + \dots + w_dx_d + b$.
2. Turn the score into a probability with the sigmoid: $p = \sigma(z) = 1/(1 + e^{-z})$.
3. Measure the log-loss over the training points, plus a penalty for large weights.
4. Change the weights in the direction that lowers that total (scikit-learn uses the L-BFGS method, which also uses the curvature of the loss). In Sightline each animation frame is one such iteration.
5. Predict class 1 when $p \ge 0.5$, i.e. when $z \ge 0$. The border $z = 0$ is a straight line.

**Worked example.** Six points on one axis: $x = -2, -1, 0, 1, 2, 3$ with classes $0, 0, 1, 0, 1, 1$.

*One training step by hand.* Start from $w = 0$, $b = 0$: every point gets $p = \sigma(0) = 0.5$. The gradient of the mean log-loss is $\frac{1}{n}\sum (p_i - y_i)x_i$ for $w$ and $\frac{1}{n}\sum (p_i - y_i)$ for $b$:

- $p - y = 0.5, 0.5, -0.5, 0.5, -0.5, -0.5$;
- for $w$: $(0.5\cdot(-2) + 0.5\cdot(-1) - 0.5\cdot 0 + 0.5\cdot 1 - 0.5\cdot 2 - 0.5\cdot 3)/6 = -3.5/6 \approx -0.583$;
- for $b$: $(0.5 + 0.5 - 0.5 + 0.5 - 0.5 - 0.5)/6 = 0$.

The gradient is negative for $w$, so increasing $w$ lowers the loss: larger $x$ means class 1 is more likely. Repeating such steps converges to $w \approx 0.767$, $b \approx -0.383$ (with the default penalty $C = 1$).

*Predicting by hand* for a new point $x = 1.5$: $z = 0.767 \cdot 1.5 - 0.383 \approx 0.767$, $p = 1/(1 + e^{-0.767}) \approx 0.683$, so class 1 with 68 %. The border is at $z = 0$, i.e. $x = 0.383/0.767 \approx 0.5$.

| $x$ | true class | $z$ | $p$(class 1) | predicted |
|---|---|---|---|---|
| −2 | 0 | −1.92 | 0.128 | 0 |
| −1 | 0 | −1.15 | 0.240 | 0 |
| 0 | 1 | −0.38 | 0.405 | 0 ✗ |
| 1 | 0 | 0.38 | 0.595 | 1 ✗ |
| 2 | 1 | 1.15 | 0.760 | 1 |
| 3 | 1 | 1.92 | 0.872 | 1 |

Two points sit on the "wrong" side of any single threshold, so 4 of 6 is the best a straight border can do here.

## The formulas {#formulas}

Probability of class 1 for two classes:

$$p(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x} + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$

What scikit-learn minimises (default L2 penalty):

$$\min_{\mathbf{w}, b}\; C\sum_{i=1}^{n} \Big[-y_i \ln p_i - (1 - y_i)\ln(1 - p_i)\Big] + \frac{1}{2}\lVert\mathbf{w}\rVert^2$$

With more than two classes each class $k$ gets its own score $z_k$ and the probabilities come from the softmax:

$$p(y = k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j} e^{z_j}}$$

- $\mathbf{x}$ — the features; $\mathbf{w}$, $b$ — weights and intercept; $z$ — the score (log-odds).
- $p_i$ — predicted probability of class 1 for point $i$; $y_i \in \{0, 1\}$ — its true class.
- $C$ — inverse strength of the penalty: large $C$ trusts the data, small $C$ keeps the weights small.
- $\lVert\mathbf{w}\rVert^2$ — the sum of squared weights (the intercept is not penalised).

## Where it works and where it fails {#where}

**Works well**

- A baseline for any classification: spam or not, churn or stay, disease or healthy.
- When you need trustworthy probabilities (they are well calibrated out of the box) and an explanation: a weight of 0.7 means one unit more multiplies the odds by $e^{0.7} \approx 2$.
- Many features and lots of data (text, one-hot categories) — fast and robust.

**Fails or misleads**

- Curved borders (two moons, circles, XOR): a straight line cannot separate them — add features or use trees, SVC with a kernel or an MLP.
- Perfectly separable data with a large $C$: the weights keep growing and the probabilities become extreme.
- Unscaled features: the penalty treats all weights alike, so a feature measured in thousands is penalised differently — scale features first.

## Parameters {#params}

| parameter | what it controls | default | raise | lower |
|---|---|---|---|---|
| `C` | inverse strength of the L2 penalty | `1.0` | weaker penalty: bigger weights, closer fit, risk of overfitting | stronger penalty: smaller weights, smoother, more confident only with strong evidence |
| `max_iter` | maximum number of solver iterations | `100` | lets the solver converge on hard problems | stops early; a "did not converge" warning is likely (Sightline shows the stop) |
| `fit_intercept` | whether to learn $b$ | `True` | — | the border must pass through the origin |
| `class_weight` | weights of the classes in the loss | `None` | `'balanced'`: each class weighted by $n / (k \cdot n_k)$, so a rare class counts as much as a common one | `None`: every point counts the same |

The type of penalty is set by `l1_ratio` in Advanced (0 = L2, 1 = L1); in scikit-learn 1.8+ the old `penalty` argument is deprecated, so Sightline hides it.

## Python code {#code}

{{code}}

## Try it in the app {#try}

On "Two moons" the straight border cannot follow the curve; watch the probability shading flatten, then raise `C` to 100 and see that a straight line stays a straight line.

{{try}}
