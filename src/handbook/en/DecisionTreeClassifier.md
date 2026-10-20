A decision tree classifies by asking a sequence of yes/no questions about one feature at a time — "is $x_1 \le 2.5$?" — until it reaches a leaf, and answers with the majority class of the training points in that leaf. It is the building block of [random forests](#hb/RandomForestClassifier/idea) and [boosting](#hb/GradientBoostingClassifier/idea). Deep trees are the textbook example of [overfitting](#hb/intro/overfitting).

## The idea {#idea}

Think of the game "twenty questions" or a doctor's triage chart: *Fever? → yes → Cough? → no → Rash? …* Each question splits the patients into two groups that are more alike than before. A good question is one that separates the sick from the healthy as cleanly as possible. A decision tree learns such a chart from data: at every step it tries all features and thresholds and keeps the question that makes the two resulting groups the most "pure".

## How it works, step by step {#steps}

1. Put all training points in the root node.
2. For every feature and every threshold between two neighbouring values, split the node into "≤ threshold" and "> threshold" and measure the impurity of the two parts (default: Gini).
3. Keep the split with the lowest weighted impurity. Create the two child nodes.
4. Repeat on each child until a node is pure, or a limit stops it (`max_depth`, `min_samples_split`, `min_samples_leaf`).
5. To predict, send a point down from the root, answering each question, and return the class proportions of the leaf it lands in.

**Worked example.** Six points on one axis: $x = 1, 2, 3, 4, 5, 6$ with classes $0, 0, 1, 1, 1, 0$. The root has three of each class: Gini $= 1 - (0.5^2 + 0.5^2) = 0.5$.

Candidate thresholds lie halfway between neighbours:

| split | left classes | right classes | Gini left | Gini right | weighted |
|---|---|---|---|---|---|
| $x \le 1.5$ | 0 | 0 1 1 1 0 | 0 | 0.480 | 0.400 |
| $x \le 2.5$ | 0 0 | 1 1 1 0 | 0 | 0.375 | **0.250** |
| $x \le 3.5$ | 0 0 1 | 1 1 0 | 0.444 | 0.444 | 0.444 |
| $x \le 4.5$ | 0 0 1 1 | 1 0 | 0.5 | 0.5 | 0.500 |
| $x \le 5.5$ | 0 0 1 1 1 | 0 | 0.480 | 0 | 0.400 |

For example, the right part of $x \le 2.5$ holds three 1s and one 0: $1 - (0.75^2 + 0.25^2) = 0.375$, weighted by $4/6$ gives $0.25$. The tree picks $x \le 2.5$. The right child $\{1, 1, 1, 0\}$ is then split at $x \le 5.5$ into $\{1, 1, 1\}$ and $\{0\}$ — both pure. Final tree: two questions, three leaves, 100 % training accuracy.

## The formulas {#formulas}

Impurity of a node whose points have class shares $p_1, \dots, p_K$:

$$\text{Gini} = 1 - \sum_{k=1}^{K} p_k^2, \qquad \text{entropy} = -\sum_{k=1}^{K} p_k \log_2 p_k$$

Quality of a split of node $N$ into left $L$ and right $R$ (the tree maximises this decrease):

$$\Delta = I(N) - \frac{|L|}{|N|} I(L) - \frac{|R|}{|N|} I(R)$$

Prediction for a point that reaches leaf $\ell$:

$$p(y = k \mid \mathbf{x}) = \frac{\#\{\text{training points of class } k \text{ in } \ell\}}{\#\{\text{training points in } \ell\}}$$

- $I(\cdot)$ — Gini or entropy of a node; $|N|$ — number of training points in it.
- `criterion='log_loss'` is the same as `'entropy'` in scikit-learn.

## Where it works and where it fails {#where}

**Works well**

- When rules must be read and checked by people: credit decisions, triage, troubleshooting guides.
- Mixed features on any scale: thresholds don't care about units, so no scaling is needed.
- Interactions ("only if young *and* a smoker") are captured naturally.

**Fails or misleads**

- Deep trees memorise noise: training accuracy 100 %, test much lower. Limit `max_depth` or raise `min_samples_leaf`.
- Unstable: a few changed points can produce a very different tree — forests and boosting fix this.
- Borders are always made of axis-parallel steps; a diagonal border needs many small steps.

## Parameters {#params}

| parameter | what it controls | default | raise | lower |
|---|---|---|---|---|
| `max_depth` | the longest chain of questions | `None` (grow until pure) | more detailed borders, more overfitting | simpler tree, smoother border, may underfit (depth 1 = a "stump") |
| `criterion` | how impurity is measured | `'gini'` | `'entropy'` / `'log_loss'`: slightly different splits, usually similar results | — |
| `min_samples_split` | the fewest points a node needs to be split | `2` | stops splitting small nodes: smaller tree | — |
| `min_samples_leaf` | the fewest points allowed in a leaf | `1` | every answer rests on more points: smoother, less noise | — |

In Sightline, **+1 level** grows the tree one level at a time, and the Loss panel shows test accuracy for each depth, so you can see where deeper stops helping.

## Python code {#code}

{{code}}

## Try it in the app {#try}

XOR needs two questions: one on $x_1$, then one on $x_2$. Train, then open **How it decides** and follow a point from the root to its leaf.

{{try}}
