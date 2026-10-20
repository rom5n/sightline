CatBoost is [gradient boosting](#hb/GradientBoostingClassifier/idea) with two distinctive ideas. It builds **symmetric (oblivious) trees** — the same question is asked across a whole level — and it reads **categorical columns directly** (`cat_features`), turning each category into a number with *ordered target statistics* instead of one-hot encoding. `CatBoostClassifier` and `CatBoostRegressor` share everything in this chapter; only the loss differs (log-loss vs RMSE). Overfitting background: [the intro](#hb/intro/overfitting).

## The idea {#idea}

Suppose you predict whether a customer buys, and one column is the city. One-hot encoding turns 300 cities into 300 mostly-zero columns. A shortcut is to replace each city by "how often customers from this city bought" — but if a row's own answer is included in that average, the model peeks at the label and overfits. CatBoost's fix is to put the rows in a random order and let each row use only the rows *before* it, like a student who may only look at past exam results, never at today's answer sheet.

The symmetric trees are a second safeguard: a depth-6 tree asks just six questions in total (one per level), so it is simple, fast to evaluate and hard to overfit.

## How it works, step by step {#steps}

1. **Categories → numbers.** Shuffle the rows. For each row, compute a target statistic of its category from the earlier rows only: (sum of labels of earlier rows with the same category + prior) / (count of those rows + 1). CatBoost computes several such statistics (and combinations of categorical columns) per feature.
2. **Start** from a constant (the bias): the log-odds of the class share, or the mean for regression.
3. **Grow a symmetric tree** on the gradients of the loss: at each level choose the one split (feature + border) that is best *for all nodes of that level together*. A tree of depth $d$ has $2^d$ leaves.
4. **Leaf values** come from gradient (Newton) steps regularised by `l2_leaf_reg`, multiplied by the learning rate.
5. **Repeat** for `iterations` trees. Predict by adding the bias and every tree's leaf value; for classification apply the sigmoid (two classes) or softmax.

**Worked example 1 — ordered target statistics.** Six rows in their random order, prior $0.5$:

| row | colour | label | earlier rows with this colour | their labels | statistic |
|---|---|---|---|---|---|
| 1 | red | 1 | — | — | $(0 + 0.5)/(0 + 1) = 0.5$ |
| 2 | blue | 0 | — | — | $(0 + 0.5)/1 = 0.5$ |
| 3 | red | 1 | 1 | 1 | $(1 + 0.5)/2 = 0.75$ |
| 4 | red | 0 | 1, 3 | 1, 1 | $(2 + 0.5)/3 \approx 0.833$ |
| 5 | blue | 1 | 2 | 0 | $(0 + 0.5)/2 = 0.25$ |
| 6 | red | 1 | 1, 3, 4 | 1, 1, 0 | $(2 + 0.5)/4 = 0.625$ |

Row 4's statistic does not know that row 4 itself is a 0 — no leakage of the row's own label.

**Worked example 2 — a symmetric tree of depth 2.** Level 1 asks "$x > 0.4$?", level 2 asks "colour statistic above its border?". Every point answers both questions, and the leaf number is built from the answers like a binary number: $\text{leaf} = [\text{answer}_1] \cdot 1 + [\text{answer}_2] \cdot 2$. A point with answers *yes*, *no* lands in leaf $1 + 0 = 1$ (the second of four leaves). This is exactly what "How it decides" shows for the first tree.

## The formulas {#formulas}

Ordered target statistic for row $k$ in the permutation $\sigma$ (category $c_k$, prior $P$, prior weight $a$):

$$\hat x_k = \frac{\sum_{j:\ \sigma(j) < \sigma(k),\ c_j = c_k} y_j \;+\; a P}{\#\{j:\ \sigma(j) < \sigma(k),\ c_j = c_k\} \;+\; a}$$

Prediction of an ensemble of $T$ oblivious trees of depth $d$:

$$F(\mathbf{x}) = b + \sum_{t=1}^{T} v_t\!\left[\sum_{l=1}^{d} 2^{\,l-1}\, \mathbb{1}\big(\mathbf{x}_{f_{tl}} > c_{tl}\big)\right]$$

Classification: $p(y = 1 \mid \mathbf{x}) = \sigma(F(\mathbf{x}))$; regression: $\hat y = F(\mathbf{x})$, trained on RMSE $= \sqrt{\frac{1}{n}\sum_i (y_i - F(\mathbf{x}_i))^2}$.

- $b$ — the bias (start value); $v_t[\cdot]$ — leaf values of tree $t$ (already multiplied by the learning rate).
- $f_{tl}$, $c_{tl}$ — the feature (a number or a category statistic) and border tested at level $l$ of tree $t$.
- $\mathbb{1}(\cdot)$ — 1 if true, 0 otherwise.

## Where it works and where it fails {#where}

**Works well**

- Tables with categorical columns: city, product type, device, user segment — especially with many distinct values.
- Good accuracy with the default settings; little tuning needed.
- Built-in handling of missing values (`nan_mode`).

**Fails or misleads**

- A large library (the Windows wheel bundles GPU code); heavy to ship in small apps.
- Slower per tree when categories are present (target statistics and their combinations are computed per tree).
- Text, images or very sparse data — other tools are better.
- Like all boosted trees: a black box compared to linear models; no extrapolation.

## Parameters {#params}

| parameter | what it controls | default | raise | lower |
|---|---|---|---|---|
| `iterations` | number of trees | `1000` (Sightline starts at 300 to keep runs short) | lower training loss; with a fixed learning rate, overfitting eventually | fewer trees, faster, may underfit |
| `learning_rate` | shrinkage of each tree | `None` → chosen automatically from the data size and `iterations` | faster fit, fewer trees needed, more overfitting risk | needs more trees, usually smoother |
| `depth` | levels of each symmetric tree ($2^d$ leaves) | `6` | more interactions per tree, slower | simpler trees |
| `l2_leaf_reg` | L2 regularisation of leaf values | `3` | smaller leaf values, smoother, less overfitting | closer fit to the training data |

Advanced settings include `one_hot_max_size` (categories with at most this many values are one-hot encoded inside CatBoost), `bootstrap_type`, `random_strength` and `boosting_type` (`'Ordered'` applies the "only earlier rows" idea to the boosting itself).

## Python code {#code}

The script computes the ordered statistics from the table above, then trains on a DataFrame with a text column — passed as `cat_features`, no encoding.

{{code}}

## Try it in the app {#try}

The button loads a generated table whose `color` column is text. The header shows `cat_features: color`, and step 1 of "How it decides" shows the colour going in as text.

{{try}}
