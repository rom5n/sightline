k-nearest neighbours (KNN) does not build a model at all: it stores the training points, and to classify a new point it finds the $k$ closest ones and lets them vote. It is the simplest *non-parametric* method — the border can take any shape the data suggests. Its single knob, $k$, moves it between [overfitting](#hb/intro/overfitting) and underfitting.

## The idea {#idea}

You move to a new street and want to know whether it is a quiet or a noisy one. You ask your three nearest neighbours: two say quiet, one says noisy — so it is probably quiet. You did not learn any rule about streets; you simply trusted the closest examples. Ask only one neighbour and a single odd answer misleads you; ask the whole town and you lose the local picture. Choosing $k$ is choosing how many neighbours to ask.

## How it works, step by step {#steps}

1. "Training" only stores the points (and builds a search structure such as a KD-tree for speed).
2. For a new point, measure its distance to every stored point (Euclidean by default).
3. Take the $k$ nearest (`n_neighbors`).
4. Vote: with `weights='uniform'` each neighbour counts once; with `'distance'` each counts $1/d$, so closer neighbours matter more.
5. The class shares of the vote are the predicted probabilities; the largest wins.

**Worked example.** Class 0 at $(0.5, 0.5), (2, 1), (1, 3.5)$; class 1 at $(4, 4.5), (5.5, 3), (4, 2)$. Query $q = (3, 2)$, $k = 3$.

| point | class | distance to $q$ |
|---|---|---|
| (4, 2) | 1 | $\sqrt{1^2 + 0^2} = 1.000$ |
| (2, 1) | 0 | $\sqrt{1^2 + 1^2} = 1.414$ |
| (1, 3.5) | 0 | $\sqrt{2^2 + 1.5^2} = 2.500$ |
| (4, 4.5) | 1 | 2.693 |
| (5.5, 3) | 1 | 2.693 |
| (0.5, 0.5) | 0 | 2.915 |

- Uniform vote of the three nearest: class 0 twice, class 1 once → $P(0) = 2/3 \approx 0.667$, predict class 0.
- Distance-weighted: class 0 gets $1/1.414 + 1/2.5 = 0.707 + 0.4 = 1.107$, class 1 gets $1/1 = 1$ → $P(0) = 1.107/2.107 \approx 0.525$ — still class 0, but much less certain, because the single class-1 neighbour is the closest.

## The formulas {#formulas}

Minkowski distance (default $p = 2$: Euclidean; $p = 1$: Manhattan):

$$d(\mathbf{x}, \mathbf{x}') = \left(\sum_{j=1}^{d} |x_j - x'_j|^p\right)^{1/p}$$

Predicted probability with neighbourhood $N_k(\mathbf{x})$ and weights $w_i$:

$$p(y = c \mid \mathbf{x}) = \frac{\sum_{i \in N_k(\mathbf{x})} w_i\,[y_i = c]}{\sum_{i \in N_k(\mathbf{x})} w_i}, \qquad w_i = \begin{cases}1 & \text{uniform}\\ 1/d(\mathbf{x}, \mathbf{x}_i) & \text{distance}\end{cases}$$

- $N_k(\mathbf{x})$ — the $k$ training points closest to $\mathbf{x}$; $[\cdot]$ — 1 if true, else 0.
- If a query coincides with a training point under `'distance'`, that point gets all the weight.

## Where it works and where it fails {#where}

**Works well**

- Small data with irregular class shapes; recommendation by similar items ("customers like you").
- A fast, assumption-free baseline; adding new data needs no retraining.

**Fails or misleads**

- Prediction is slow on large data: every query searches the stored points.
- Many features: distances become similar for all points (the *curse of dimensionality*), and neighbours stop being meaningful.
- Unscaled features: a feature in thousands dominates the distance — always scale.
- Imbalanced classes: the majority class wins most neighbourhoods; use `'distance'` weights or rebalance.

## Parameters {#params}

| parameter | what it controls | default | raise | lower |
|---|---|---|---|---|
| `n_neighbors` | number of neighbours $k$ | `5` | smoother border, more stable, may miss small details (underfit) | follows every point; $k = 1$ memorises the training data (overfit) |
| `weights` | how neighbours' votes count | `'uniform'` | `'distance'`: closer neighbours count more, sharper local detail | — |
| `metric` | the distance function | `'minkowski'` | other metrics, e.g. `'chebyshev'` | — |
| `p` | power of the Minkowski distance | `2` (Euclidean) | larger $p$ approaches the largest single difference | `1`: Manhattan, more robust to one large difference |

KNN has nothing to train, so in Sightline the Loss panel shows test accuracy for every $k$ from 1 to 30 and marks the current one.

## Python code {#code}

{{code}}

## Try it in the app {#try}

On "Two moons" set `n_neighbors` to 1 and look at the islands around single points; then try 30 and see the border become smooth. The curve in the Loss panel tells you which $k$ was best on the test set.

{{try}}
