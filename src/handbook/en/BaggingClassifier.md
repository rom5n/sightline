Bagging (*bootstrap aggregating*) trains several copies of one model, each on a random resample of the training rows, and averages their predictions. It is the general recipe behind the [random forest](#hb/RandomForestClassifier/idea) — but works with any base model. It reduces variance, the part of the error that comes from [overfitting](#hb/intro/overfitting) to one particular sample.

## The idea {#idea}

A teacher wants to know how good a student really is, but every test has some lucky or unlucky questions. So she gives the student ten tests built from random questions out of the same pool and averages the scores — the average says more than any single test. Bagging does this for models: each copy learns from a different random selection of rows, so each copy's quirks are different, and the average keeps what they agree on.

## How it works, step by step {#steps}

1. From the $n$ training rows draw a bootstrap sample: $n$ draws *with replacement* (with the defaults; `max_samples` can make the samples smaller).
2. Optionally also draw a subset of the features (`max_features` < 1.0).
3. Train one copy of the base model (by default a decision tree) on that sample.
4. Repeat `n_estimators` times.
5. Predict by averaging the copies' class probabilities (or their votes, if the base model has no probabilities).

**Worked example.** Eight training rows numbered 0–7, three trees. The bootstrap samples drawn (row numbers, sorted):

| tree | rows in its sample | missing rows | its vote for $q = (3, 2.5)$ |
|---|---|---|---|
| 1 | 0 0 1 2 2 3 4 6 | 5, 7 | class 0 |
| 2 | 0 0 1 3 3 4 5 5 | 2, 6, 7 | class 0 |
| 3 | 2 3 4 5 5 5 6 6 | 0, 1, 7 | class 1 |

Each sample has eight rows but only five or six distinct ones. Two of three trees say class 0, so $P(\text{class }0) = 2/3 \approx 0.667$. Tree 3 never saw rows 0 and 1 (both class 0), which is why it disagrees — the kind of sample-specific quirk that averaging smooths out.

## The formulas {#formulas}

Bagged probability from $B$ copies $f_1, \dots, f_B$:

$$\hat p(y = k \mid \mathbf{x}) = \frac{1}{B}\sum_{b=1}^{B} \hat p_b(y = k \mid \mathbf{x})$$

Expected share of distinct rows in a bootstrap sample of size $n$:

$$1 - \left(1 - \frac{1}{n}\right)^n \approx 1 - e^{-1} \approx 0.632$$

Bias–variance view: the averaged model has about the same bias as one copy, but its variance falls towards $\rho\,\sigma^2$ as $B$ grows (see the random forest chapter).

- $\hat p_b$ — probabilities predicted by copy $b$, trained on bootstrap sample $b$.

## Where it works and where it fails {#where}

**Works well**

- Unstable, high-variance base models: deep decision trees, KNN with small $k$, small neural networks.
- When you want an ensemble but must keep a specific base model.
- Out-of-bag estimates come for free (`oob_score=True`), as in a forest.

**Fails or misleads**

- Stable, high-bias models (linear models, very shallow trees): averaging copies that make the same mistake does not fix it — use boosting.
- Trees on correlated features: without random feature choice per split the copies remain similar; a random forest is usually better for trees.
- Cost grows linearly with `n_estimators`.

## Parameters {#params}

| parameter | what it controls | default | raise | lower |
|---|---|---|---|---|
| `n_estimators` | number of copies | `10` | smoother, more stable average; slower | noisier |
| `max_samples` | rows drawn per copy (count or share) | `None` (= as many as there are rows) | — | smaller samples: copies differ more, faster, each one weaker |
| `max_features` | share of features per copy | `1.0` | — | each copy sees fewer features ("random subspaces"): more diversity |

In Sightline copies are added in batches and the Loss panel tracks test accuracy (and out-of-bag accuracy when enabled) as the ensemble grows.

## Python code {#code}

A single split is too noisy to compare models, so the code averages over 10 splits: one deep tree reaches 0.833, ten bagged trees 0.857.

{{code}}

## Try it in the app {#try}

Compare the speckled border of a single deep tree (DecisionTreeClassifier) with bagging's smoother one on the same "Two moons" data.

{{try}}
