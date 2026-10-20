AdaBoost (*adaptive boosting*) trains a sequence of weak models — by default decision stumps, trees with a single question — and makes each one focus on the points the previous ones got wrong, by giving those points more weight. The final prediction is a vote in which better stumps have a bigger say. It was the first practical [boosting](#hb/GradientBoostingClassifier/idea) method. See [the intro](#hb/intro/training) for training basics.

## The idea {#idea}

A student revises for an exam with a set of flashcards. After each round she puts the cards she got wrong at the front of the pile and goes through them more often, while the ones she knows well appear rarely. Each round she learns a simple rule of thumb; in the exam she combines all her rules, trusting the ones that worked best during practice more. AdaBoost does the same with data points instead of cards and stumps instead of rules of thumb.

## How it works, step by step {#steps}

scikit-learn uses the SAMME algorithm:

1. Give every training point the same weight $w_i = 1/n$.
2. Train a stump on the weighted points (it minimises the *weighted* error).
3. Compute its weighted error $e$ and its "say" $\alpha = \nu\left(\ln\frac{1 - e}{e} + \ln(K - 1)\right)$ — with two classes the last term is 0.
4. Multiply the weight of every misclassified point by $e^{\alpha}$, then rescale all weights to sum to 1.
5. Repeat steps 2–4 for `n_estimators` stumps (training stops early if a stump is perfect).
6. Predict by letting all stumps vote; each vote counts $\alpha$, the class with the largest total wins.

**Worked example.** Six points $x = 1..6$ with classes $0, 0, 1, 1, 1, 0$; learning rate $\nu = 1$.

*Stump 1.* All weights are $1/6$. The best stump is $x \le 2.5$ → class 0, else class 1. It is wrong only on $x = 6$: $e_1 = 1/6$, $\alpha_1 = \ln\frac{5/6}{1/6} = \ln 5 \approx 1.609$.

*Reweighting.* The weight of $x = 6$ becomes $\frac{1}{6}\cdot 5 = \frac{5}{6}$; the total is $\frac{10}{6}$, so after rescaling the five correct points weigh $0.1$ each and $x = 6$ weighs $0.5$.

*Stump 2.* With these weights the best stump is $x \le 5.5$ → class 1, else class 0. It is wrong on $x = 1$ and $x = 2$: $e_2 = 0.2$, $\alpha_2 = \ln 4 \approx 1.386$.

| $x$ | class | stump 1 ($\alpha = 1.609$) | stump 2 ($\alpha = 1.386$) | vote for 0 | vote for 1 | prediction |
|---|---|---|---|---|---|---|
| 1, 2 | 0 | 0 | 1 | 1.609 | 1.386 | 0 ✓ |
| 3, 4, 5 | 1 | 1 | 1 | 0 | 2.996 | 1 ✓ |
| 6 | 0 | 1 | 0 | 1.386 | 1.609 | 1 ✗ |

After two stumps 5 of 6 points are right; $x = 6$ now carries even more weight, so a third stump will target it.

## The formulas {#formulas}

Weighted error and say of stump $m$ (SAMME, $K$ classes):

$$e_m = \frac{\sum_i w_i\,[h_m(\mathbf{x}_i) \ne y_i]}{\sum_i w_i}, \qquad \alpha_m = \nu\left(\ln\frac{1 - e_m}{e_m} + \ln(K - 1)\right)$$

Weight update:

$$w_i \leftarrow w_i \cdot \exp\!\big(\alpha_m\,[h_m(\mathbf{x}_i) \ne y_i]\big), \quad \text{then normalise to } \textstyle\sum_i w_i = 1$$

Final classifier:

$$\hat y(\mathbf{x}) = \arg\max_k \sum_{m=1}^{M} \alpha_m\,[h_m(\mathbf{x}) = k]$$

- $[\cdot]$ is 1 when the condition holds, else 0; $h_m$ — stump $m$; $\nu$ — `learning_rate`.
- This is equivalent to fitting an additive model with the exponential loss $e^{-y F(\mathbf{x})}$.

## Where it works and where it fails {#where}

**Works well**

- Clean data where simple rules combine well: face detection (the classic Viola–Jones detector), quality checks with a few reliable signals.
- A big improvement over a single weak learner with almost no tuning.

**Fails or misleads**

- Noisy labels and outliers: every round increases their weight, so later stumps chase them — the exponential loss is unforgiving.
- Problems that need deep interactions: stumps add effects one feature at a time; use deeper base trees or gradient boosting.

## Parameters {#params}

| parameter | what it controls | default | raise | lower |
|---|---|---|---|---|
| `n_estimators` | maximum number of weak learners | `50` | more rounds, lower training error, possible overfitting on noise | fewer rounds, simpler model |
| `learning_rate` | multiplies each $\alpha$ ($\nu$) | `1.0` | each stump counts more, faster fit | smaller steps, needs more stumps, more robust |
| base tree depth (`estimator` = `DecisionTreeClassifier(max_depth=1)`) | depth of each weak learner | `1` (stump) | each learner captures interactions; overfits sooner | — |

In Sightline the base tree depth is a main parameter; the Loss panel replays the error rate after each stump (`staged_predict`).

## Python code {#code}

{{code}}

## Try it in the app {#try}

XOR cannot be split by any single stump. Watch the first few stumps fail, the misclassified corners gain weight, and the vote of later stumps carve out the checkerboard.

{{try}}
