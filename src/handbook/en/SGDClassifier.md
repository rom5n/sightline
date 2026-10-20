`SGDClassifier` is a linear classifier — a straight border, like [logistic regression](#hb/LogisticRegression/idea) — trained by *stochastic gradient descent*: it looks at one point at a time and nudges the weights a little after each. With the default hinge loss it is a linear support vector machine; with `loss='log_loss'` it is logistic regression. Background on loss and steps: [the intro](#hb/intro/training).

## The idea {#idea}

You are tuning a radio in a noisy car. Instead of measuring the whole spectrum and computing the perfect setting, you turn the knob a little, listen, and turn again. Each correction is based on what you hear right now, so it is quick and a bit noisy — but after enough small turns you are on the station. SGD does the same: every training point gives a tiny correction to the weights, and the steps get smaller over time so the weights settle.

## How it works, step by step {#steps}

1. Start with all weights $\mathbf{w} = 0$ and $b = 0$.
2. Shuffle the training points. For each point compute the score $z = \mathbf{w}^\top\mathbf{x} + b$.
3. Compute the gradient of the loss *for this point only* and step against it: $\mathbf{w} \leftarrow \mathbf{w} - \eta\,(\text{gradient} + \alpha\mathbf{w})$.
4. One pass over all points is an *epoch*. Repeat epochs until the loss stops improving (`tol`) or `max_iter` epochs are done. In Sightline each frame is one epoch (`partial_fit`).

With the hinge loss and labels $y \in \{-1, +1\}$ the point's gradient is simple: if the point is on the right side with room to spare ($y z \ge 1$) nothing happens; otherwise the weights move towards it: $\mathbf{w} \leftarrow \mathbf{w} + \eta\, y\,\mathbf{x}$, $b \leftarrow b + \eta\, y$.

**Worked example** (fixed step $\eta = 0.1$, no penalty, to keep the numbers round). Points: $A = (1, 2)$ with $y = +1$, $B = (2, 0)$ with $y = -1$, $C = (0, 1)$ with $y = +1$.

| step | point | score $z$ | $y z < 1$? | new $\mathbf{w}$ | new $b$ |
|---|---|---|---|---|---|
| start | — | — | — | (0, 0) | 0 |
| 1 | A | 0 | yes | (0.1, 0.2) | 0.1 |
| 2 | B | $0.2 + 0 + 0.1 = 0.3$ | yes ($-0.3$) | (−0.1, 0.2) | 0 |
| 3 | C | $0 + 0.2 + 0 = 0.2$ | yes | (−0.1, 0.3) | 0.1 |

After one epoch all three points are on the correct side: scores $0.6$, $-0.1$, $0.4$. They are still inside the margin, so further epochs keep pushing them out.

## The formulas {#formulas}

What `SGDClassifier` minimises:

$$\min_{\mathbf{w}, b}\; \frac{1}{n}\sum_{i=1}^{n} L\big(y_i,\ \mathbf{w}^\top\mathbf{x}_i + b\big) + \alpha\, R(\mathbf{w})$$

Losses: hinge $L = \max(0,\ 1 - y z)$; log-loss $L = \ln(1 + e^{-y z})$. Penalties: $R = \frac{1}{2}\lVert\mathbf{w}\rVert^2$ (`'l2'`), $\lVert\mathbf{w}\rVert_1$ (`'l1'`) or a mix (`'elasticnet'`).

Update for one point with the default `learning_rate='optimal'`:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta_t\left(\frac{\partial L}{\partial \mathbf{w}} + \alpha \frac{\partial R}{\partial \mathbf{w}}\right), \qquad \eta_t = \frac{1}{\alpha\,(t + t_0)}$$

- $y_i \in \{-1, +1\}$ — the class as a sign; $z = \mathbf{w}^\top\mathbf{x} + b$ — the score.
- $\alpha$ — penalty strength, also sets the step-size schedule; $t$ — number of updates so far; $t_0$ — a starting offset chosen by scikit-learn.

## Where it works and where it fails {#where}

**Works well**

- Very large data (millions of rows) or data arriving in a stream: `partial_fit` learns batch by batch without keeping everything in memory.
- Sparse, high-dimensional inputs such as text features.

**Fails or misleads**

- Unscaled features: one feature in thousands dominates the steps — always scale (the code below uses a pipeline).
- Small data: a closed-form or L-BFGS model (`LogisticRegression`, `SVC` with a linear kernel) is more precise and has fewer knobs.
- Curved borders: it is still linear.
- The default hinge loss gives no probabilities; use `loss='log_loss'` or `'modified_huber'` if you need them.

## Parameters {#params}

| parameter | what it controls | default | raise / change | lower |
|---|---|---|---|---|
| `loss` | the per-point loss | `'hinge'` (linear SVM) | `'log_loss'`: logistic regression with probabilities; `'modified_huber'`: smooth, probabilities, robust to outliers; `'perceptron'`: no margin | — |
| `penalty` | the regulariser $R$ | `'l2'` | `'l1'`: sparse weights; `'elasticnet'`: both | `None`: no penalty |
| `alpha` | penalty strength (and step schedule) | `0.0001` | smaller weights, smoother border, smaller steps | closer fit to the training data, larger steps |
| `max_iter` | maximum number of epochs | `1000` | more time to converge | stops early, maybe before converging |

## Python code {#code}

{{code}}

## Try it in the app {#try}

On "Blobs" press **Step** a few times: each press is one epoch, and you can watch the straight border swing into place.

{{try}}
