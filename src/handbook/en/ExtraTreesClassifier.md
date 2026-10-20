Extra Trees ("extremely randomised trees") is a [random forest](#hb/RandomForestClassifier/idea) with one more dose of randomness: instead of searching for the best threshold, each candidate split uses a threshold drawn at random. The trees become cheaper to build and more different from each other; their average is smoother. Background: [overfitting](#hb/intro/overfitting).

## The idea {#idea}

A random forest is a crowd of experts who each studied a different sample. Extra Trees is a crowd of quick guessers: each of them, at every question, picks a random cut-off and only checks which of a few such random cuts works best. Any single guesser is cruder than an expert, but there are many of them and their errors are very independent, so the average is about as good — and they finish much faster.

## How it works, step by step {#steps}

1. By default every tree uses *all* training rows (`bootstrap=False`) — the randomness comes from the splits instead.
2. At a node, pick $m$ random features (default $\sqrt{d}$). For each, draw one threshold uniformly between that feature's minimum and maximum *in this node*.
3. Among these $m$ random splits keep the one with the lowest impurity (Gini by default).
4. Grow the tree until leaves are pure or limits stop it; build `n_estimators` such trees.
5. Predict by averaging the trees' class probabilities, exactly as a random forest does.

**Worked example.** The six points from the decision tree chapter, $x = 1..6$ with classes $0, 0, 1, 1, 1, 0$. In the root the feature ranges from 1 to 6, so a threshold is drawn uniformly in $[1, 6]$. Two trees might draw:

| tree | random threshold | left / right classes | weighted Gini |
|---|---|---|---|
| A | 2.1 | 0 0 / 1 1 1 0 | 0.250 (the best possible split, found by luck) |
| B | 3.8 | 0 0 1 / 1 1 0 | 0.444 |

A decision tree would always choose 2.5; Extra Trees accepts tree B's poorer split. Averaged over hundreds of trees with different random cuts, the border lands in roughly the same place, but it is softer, and no expensive search over all thresholds was needed.

## The formulas {#formulas}

Random threshold for feature $j$ in node $N$:

$$t_j \sim \mathcal{U}\left(\min_{i \in N} x_{ij},\ \max_{i \in N} x_{ij}\right)$$

Chosen split among the $m$ candidates (lowest weighted impurity $I$):

$$j^* = \arg\min_{j}\ \left[\frac{|L_j|}{|N|}I(L_j) + \frac{|R_j|}{|N|}I(R_j)\right]$$

Prediction (as in a random forest):

$$p(y = k \mid \mathbf{x}) = \frac{1}{T}\sum_{t=1}^{T} p_t(y = k \mid \mathbf{x})$$

- $\mathcal{U}(a, b)$ — uniform distribution between $a$ and $b$; $L_j$, $R_j$ — the two parts made by the threshold $t_j$.

## Where it works and where it fails {#where}

**Works well**

- The same tabular problems as random forests, when training time matters.
- Noisy data: the extra randomness lowers variance further, giving smoother borders.

**Fails or misleads**

- Sharp, precisely located borders: random thresholds blur them; a random forest or boosting may be more accurate.
- Many irrelevant features: random splits on useless features waste depth — raise `max_features`.
- Fully grown trees on all rows still memorise single points; `min_samples_leaf` > 1 often helps (see the code).

## Parameters {#params}

| parameter | what it controls | default | raise | lower |
|---|---|---|---|---|
| `n_estimators` | number of trees | `100` | smoother average, slower | noisier |
| `max_depth` | depth limit of each tree | `None` | — | shallower trees, smoother border |
| `max_features` | random features (each with one random threshold) per split | `'sqrt'` | the best of more random cuts: closer to an ordinary forest | more randomness, more diverse trees |
| `min_samples_leaf` | fewest training points per leaf | `1` | smoother probabilities (often better on noisy data) | — |

## Python code {#code}

On 1 500 noisy points Extra Trees trains faster than a random forest at similar accuracy, and `min_samples_leaf=5` helps clearly.

{{code}}

## Try it in the app {#try}

Train on "Two moons", then raise `min_samples_leaf` to 10 and watch the speckles inside each moon disappear.

{{try}}
