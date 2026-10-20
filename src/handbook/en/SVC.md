The support vector classifier (SVC) looks for the border that leaves the **widest possible margin** between the classes. Only the points closest to the border — the *support vectors* — decide where it goes. With a *kernel* the same idea produces curved borders. Margin and the penalty `C` are a direct way to control [overfitting](#hb/intro/overfitting).

## The idea {#idea}

You are painting the widest possible road between two villages without touching any house. The road's centre line is the border; its edges touch a few houses on each side — those are the support vectors. Every other house could move around freely without changing the road. A wide road is a cautious, robust border: a new house slightly off its expected place still ends up on the right side. If a few houses are in the way, the soft-margin version lets the road pass over them, at a price set by `C`.

## How it works, step by step {#steps}

1. Write the border as $\mathbf{w}^\top\mathbf{x} + b = 0$; the margin edges are $\mathbf{w}^\top\mathbf{x} + b = \pm 1$, so the margin width is $2/\lVert\mathbf{w}\rVert$.
2. Making the margin wide means making $\lVert\mathbf{w}\rVert$ small, while keeping points on the right side; points inside the margin or on the wrong side pay a penalty proportional to how far they are in ($\xi_i$), weighted by `C`.
3. This is a convex quadratic problem; libsvm solves it in its *dual* form, which gives a weight $\alpha_i$ for each training point. Only points with $\alpha_i > 0$ — the support vectors — remain in the model.
4. For curved borders, replace every dot product $\mathbf{x}_i^\top\mathbf{x}$ by a kernel $K(\mathbf{x}_i, \mathbf{x})$ — for example the RBF kernel, a similarity that fades with distance. This is the same as a linear border in a much richer feature space, without ever computing that space.
5. Predict the class from the sign of the decision value $f(\mathbf{x})$. Sightline shades by a sigmoid of $f$ (scikit-learn 1.9 deprecates `probability=True`).

**Worked example.** Six points: class 0 at $(1, 2), (2, 3), (2, 1)$; class 1 at $(4, 3), (5, 4), (4, 1)$. Linear kernel, large $C$ (practically a hard margin).

- The widest road is vertical: border $x_1 = 3$, i.e. $\mathbf{w} = (1, 0)$, $b = -3$.
- The edges are $x_1 = 2$ and $x_1 = 4$: margin width $2/\lVert\mathbf{w}\rVert = 2$.
- scikit-learn keeps two support vectors, $(2, 1)$ and $(4, 1)$, with $\alpha_i y_i = -0.5$ and $+0.5$.

Decision value for $q = (2.5, 2)$ using only the support vectors:

$$f(q) = -0.5\,\langle(2, 1), q\rangle + 0.5\,\langle(4, 1), q\rangle - 3 = -0.5 \cdot 7 + 0.5 \cdot 12 - 3 = -0.5$$

Negative, so class 0 — the same as $\mathbf{w}^\top q + b = 2.5 - 3 = -0.5$. Moving the point $(1, 2)$ anywhere further left would not change the model at all.

## The formulas {#formulas}

Soft-margin primal problem:

$$\min_{\mathbf{w}, b, \xi}\; \frac{1}{2}\lVert\mathbf{w}\rVert^2 + C\sum_{i=1}^{n}\xi_i \quad\text{s.t.}\quad y_i(\mathbf{w}^\top\phi(\mathbf{x}_i) + b) \ge 1 - \xi_i,\ \ \xi_i \ge 0$$

Decision function from the dual solution:

$$f(\mathbf{x}) = \sum_{i \in \text{SV}} \alpha_i y_i\, K(\mathbf{x}_i, \mathbf{x}) + b, \qquad 0 \le \alpha_i \le C$$

Kernels: linear $K = \mathbf{x}^\top\mathbf{x}'$; RBF $K = \exp(-\gamma\lVert\mathbf{x} - \mathbf{x}'\rVert^2)$; polynomial $K = (\gamma\,\mathbf{x}^\top\mathbf{x}' + r)^{d}$. With `gamma='scale'`, $\gamma = 1/(d \cdot \operatorname{Var}(X))$.

- $y_i \in \{-1, +1\}$; $\xi_i$ — how far point $i$ violates the margin; $\phi$ — the feature map implied by the kernel.
- $d$ in `'scale'` is the number of features; $\operatorname{Var}(X)$ is the variance of all feature values together.
- More than two classes: scikit-learn trains one SVC per pair of classes (one-vs-one) and lets them vote.

## Where it works and where it fails {#where}

**Works well**

- Small to medium data with complex but clean borders: handwriting digits, protein classification, sensor signatures.
- Many features relative to rows (text, genomics) with a linear kernel.

**Fails or misleads**

- Training time grows roughly between $n^2$ and $n^3$: beyond ~10 000–50 000 rows prefer linear models, SGD or boosting. (Sightline shows a progress bar and Cancel for SVC.)
- Features must be scaled: the RBF distance is meaningless across units.
- `C` and `gamma` interact; a poor pair gives either a blob or a memorised map. Use a grid search.
- No native probabilities.

## Parameters {#params}

| parameter | what it controls | default | raise | lower |
|---|---|---|---|---|
| `C` | penalty for margin violations | `1.0` | fewer violations allowed: narrower margin, fewer support vectors, closer fit, overfitting risk | wider, softer margin, more support vectors, smoother border |
| `kernel` | shape of the border | `'rbf'` | `'poly'`: polynomial curves; `'linear'`: straight border, fastest; `'sigmoid'` | — |
| `gamma` | reach of one support vector (RBF, poly, sigmoid) | `'scale'` | each point influences only its close neighbourhood: wiggly border, overfitting | wide influence: smooth, almost linear border |
| `degree` | polynomial degree (only for `kernel='poly'`) | `3` | more flexible curves | simpler curves |

## Python code {#code}

{{code}}

## Try it in the app {#try}

No straight line separates "Concentric circles"; the RBF kernel draws a ring. Support vectors are outlined in the plot — raise `C` and watch their number drop.

{{try}}
