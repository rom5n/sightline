A random forest trains many [decision trees](#hb/DecisionTreeClassifier/idea), each on a slightly different version of the data, and averages their class probabilities. Single deep trees [overfit](#hb/intro/overfitting) in different ways; averaging many of them cancels much of that noise while keeping their flexibility.

## The idea {#idea}

Ask one person to guess the number of sweets in a jar and the answer can be far off. Ask a hundred people who each looked from a different angle and average their guesses — the average is usually much closer than most individuals. This "wisdom of the crowd" only works if the guesses are not all wrong in the same way. A random forest makes its trees disagree on purpose: each tree sees a random resample of the rows and, at each split, only a random subset of the features.

## How it works, step by step {#steps}

1. Draw a *bootstrap sample*: pick $n$ rows from the $n$ training rows *with replacement* (some rows appear twice, about 37 % not at all).
2. Grow a decision tree on it. At every node, choose the best split only among $m$ randomly chosen features (default $m = \sqrt{d}$).
3. Repeat for `n_estimators` trees, each with its own random sample and feature choices.
4. To predict, pass the point through every tree and average their class probabilities; the class with the highest average wins.
5. Rows a tree did not see (*out-of-bag*) give a free estimate of test accuracy (`oob_score=True`).

**Worked example.** Eight points with two features; the forest has three trees, and we ask about the point $q = (2.5, 3.5)$:

| tree | its root question (from its own bootstrap sample) | where $q$ goes | P(class 1) |
|---|---|---|---|
| 1 | $x_2 \le 2.5$? | no → a part with class 1 | 1.0 |
| 2 | $x_1 \le 3.0$? | yes → a part with class 0 | 0.0 |
| 3 | $x_2 \le 2.5$? | no → a part with class 1 | 1.0 |

Average: $(1.0 + 0.0 + 1.0)/3 \approx 0.667$, so the forest predicts class 1 with 67 %. Because the trees chose different first questions, their mistakes do not line up; with 100 trees the averaged probability changes smoothly across the plane instead of in hard steps.

## The formulas {#formulas}

Forest prediction from $T$ trees:

$$p(y = k \mid \mathbf{x}) = \frac{1}{T}\sum_{t=1}^{T} p_t(y = k \mid \mathbf{x}), \qquad \hat y = \arg\max_k\, p(y = k \mid \mathbf{x})$$

Why averaging helps: for $T$ trees whose errors each have variance $\sigma^2$ and pairwise correlation $\rho$, the variance of the average is

$$\operatorname{Var} = \rho\,\sigma^2 + \frac{1 - \rho}{T}\,\sigma^2$$

The second term shrinks as trees are added; the random feature choice lowers $\rho$, which shrinks the first term.

Probability that a given row is *not* in a bootstrap sample: $(1 - 1/n)^n \approx e^{-1} \approx 0.37$.

- $p_t$ — probabilities from tree $t$ (class shares in the leaf the point reaches).
- $d$ — number of features; $m$ — features tried per split.

## Where it works and where it fails {#where}

**Works well**

- A strong default for tabular data: fraud detection, churn, medical risk scores, sensor fault detection.
- Little tuning needed; robust to outliers and irrelevant features; no scaling needed.
- Built-in feature importance and out-of-bag validation.

**Fails or misleads**

- Hundreds of deep trees are slow to predict on huge data and large to store.
- Hard to explain one decision compared with a single tree or a linear model.
- Extrapolation: like any tree model it cannot predict outside the range it saw.
- Impurity-based feature importance favours features with many distinct values; prefer permutation importance for conclusions.

## Parameters {#params}

| parameter | what it controls | default | raise | lower |
|---|---|---|---|---|
| `n_estimators` | number of trees | `100` | smoother, more stable probabilities; slower; never makes it overfit | noisier predictions; faster |
| `max_depth` | depth limit of each tree | `None` | — | shallower trees: smoother border, may underfit |
| `max_features` | features tried at each split | `'sqrt'` ($\sqrt{d}$) | more features per split (`None` = all): stronger but more similar trees | fewer: more diverse trees, more randomness |
| `min_samples_leaf` | fewest training points per leaf | `1` | smoother probabilities, less noise | — |

In Sightline trees are added in batches (warm start), and the Loss panel shows test accuracy as the forest grows — plus out-of-bag accuracy when `oob_score` is on.

## Python code {#code}

{{code}}

## Try it in the app {#try}

On "Two moons" compare the border after 1 tree and after 100: the staircase of a single tree turns into a smooth probability map.

{{try}}
