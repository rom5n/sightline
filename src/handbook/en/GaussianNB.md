Gaussian naive Bayes describes each class by a bell curve (a normal distribution) for every feature, assumes the features are independent within a class, and uses Bayes' rule to turn "how typical is this point for each class" into class probabilities. Training is just computing means and variances, so it is instant and works with very little data. See [the intro](#hb/intro/split) for how it is evaluated.

## The idea {#idea}

A doctor sees a patient with a temperature of 38.5 °C. She knows that patients with flu usually have around 39 °C, patients with a cold around 37.5 °C, and that colds are far more common than flu. She weighs "how typical is 38.5 for each illness" by "how common is each illness" and picks the more likely one. Naive Bayes does exactly this, for every feature at once — and *naively* multiplies the evidence of the features as if they were unrelated.

## How it works, step by step {#steps}

1. For every class $c$, compute its share of the training data — the prior $P(c)$ — unless you set `priors` yourself.
2. For every class and every feature compute the mean $\mu_{cj}$ and variance $\sigma^2_{cj}$ of that feature within the class.
3. Add a tiny $\varepsilon$ to all variances (`var_smoothing` × the largest feature variance) so that none is zero.
4. For a new point, compute for each class: prior × the product of the bell-curve heights (likelihoods) of its feature values.
5. Normalise these numbers to sum to 1: these are the class probabilities; predict the largest.

**Worked example.** One feature. Class A at $x = 1, 2, 3$; class B at $x = 6, 7, 8$. New point $x = 4$.

- Means: $\mu_A = 2$, $\mu_B = 7$. Variances (dividing by $n$): $\sigma^2 = \frac{(1-2)^2 + 0 + (3-2)^2}{3} = 0.667$ for both.
- Priors: 3 of 6 rows each, so $0.5$ and $0.5$.
- Likelihoods: $\mathcal{N}(4 \mid 2, 0.667) = \frac{1}{\sqrt{2\pi \cdot 0.667}}\, e^{-(4-2)^2/(2 \cdot 0.667)} \approx 0.4886 \cdot e^{-3} \approx 0.02433$; $\mathcal{N}(4 \mid 7, 0.667) \approx 0.4886 \cdot e^{-6.75} \approx 0.00057$.
- Posterior: $P(A \mid 4) = \frac{0.5 \cdot 0.02433}{0.5 \cdot 0.02433 + 0.5 \cdot 0.00057} \approx 0.977$, so class A with 97.7 %.

With equal variances and equal priors the border lies exactly halfway between the means, at $x = 4.5$, where both classes get 50 %.

## The formulas {#formulas}

Bayes' rule with the naive independence assumption:

$$P(c \mid \mathbf{x}) = \frac{P(c)\prod_{j=1}^{d} \mathcal{N}(x_j \mid \mu_{cj}, \sigma^2_{cj})}{\sum_{c'} P(c')\prod_{j=1}^{d} \mathcal{N}(x_j \mid \mu_{c'j}, \sigma^2_{c'j})}$$

Normal density:

$$\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

Variance smoothing: $\sigma^2_{cj} \leftarrow \sigma^2_{cj} + \varepsilon$, $\varepsilon = \text{var\_smoothing} \cdot \max_j \operatorname{Var}(x_j)$.

- $P(c)$ — prior of class $c$; $\mu_{cj}$, $\sigma^2_{cj}$ — mean and variance of feature $j$ in class $c$.
- In practice scikit-learn adds log-likelihoods instead of multiplying small numbers.

## Where it works and where it fails {#where}

**Works well**

- Very little data, or many features: spam filters, document topics (with the multinomial variant), quick medical screening scores.
- A fast baseline that trains in milliseconds and can learn incrementally (`partial_fit`).

**Fails or misleads**

- Correlated features: the same evidence is counted several times, so probabilities become overconfident (often near 0 or 1) even when the predicted class is right.
- Features that are far from bell-shaped (skewed, multimodal) — transform them or use another model.
- The border is a quadratic curve at most; it cannot follow complex shapes like spirals.

## Parameters {#params}

| parameter | what it controls | default | raise | lower |
|---|---|---|---|---|
| `var_smoothing` | share of the largest variance added to every variance | `1e-9` | wider bell curves: smoother, less confident probabilities | sharper curves; at 0 a zero-variance feature would break the model |
| `priors` | the class probabilities before seeing $\mathbf{x}$ | `None` (class shares in the training data) | giving a class a larger prior moves the border towards the other class | — |

Naive Bayes is fitted in one step; in Sightline the Loss panel shows the learned priors and per-class means instead of a curve.

## Python code {#code}

{{code}}

## Try it in the app {#try}

On "Blobs" each class really is a bell curve, so naive Bayes does well; open **How it decides** to see the likelihoods and priors multiplied for one point.

{{try}}
