Every model in Sightline learns the same way: it starts with a guess, measures how wrong the guess is, and changes itself to be less wrong. This chapter explains the four ideas the other chapters build on — **training**, **loss**, **overfitting** and the **train/test split** — and how to read what the app shows you.

## The idea {#idea}

Think of learning to throw darts. You throw, see how far you missed, and adjust your arm. After many throws you hit closer to the centre. A model does the same with numbers: its *parameters* (weights, thresholds, centres…) are the arm, the *loss* is the distance from the centre, and *training* is the series of adjustments.

The catch: you want to hit the board *in a match*, not only in your own practice room. A model that only remembers its practice throws has learned nothing useful. That is why part of the data is kept aside for a fair test.

## How it works, step by step {#steps}

### Training {#training}

1. Choose a model and its settings (the *hyperparameters*, e.g. the depth of a tree).
2. Start from initial parameters (often zeros or random values).
3. Compute predictions for the training points.
4. Compute the loss: one number that says how bad those predictions are.
5. Change the parameters to lower the loss (a gradient step, a new tree, a moved centre…).
6. Repeat from step 3 until the loss stops improving or a limit is reached.

In the app, each frame of the animation is one pass of steps 3–5. Some models (a decision tree, SVC, Naive Bayes) find their parameters in one go; the app then shows "Fitted in one step".

### Loss {#loss}

The loss turns "how wrong" into a number.

**For numbers (regression)** the usual loss is the *mean squared error*. Worked example: true values $3, 5, 7$, predictions $2.5, 5, 8$. Errors: $-0.5, 0, 1$; squared: $0.25, 0, 1$; mean: $1.25 / 3 \approx 0.417$.

**For classes (classification)** the usual loss is the *log-loss*: it punishes a model for being confident and wrong. Worked example with four points, true classes $1, 0, 1, 1$ and predicted probabilities of class 1 equal to $0.9, 0.2, 0.6, 0.3$:

| point | true class | p(class 1) | probability of the *true* class | $-\ln$ of it |
|---|---|---|---|---|
| 1 | 1 | 0.9 | 0.9 | 0.105 |
| 2 | 0 | 0.2 | 0.8 | 0.223 |
| 3 | 1 | 0.6 | 0.6 | 0.511 |
| 4 | 1 | 0.3 | 0.3 | 1.204 |

The mean is $2.043 / 4 \approx 0.511$. Point 4 alone contributes more than half: the model gave the right class only 30 %.

### Overfitting {#overfitting}

A flexible model can lower the training loss all the way to zero by memorising every point, noise included. It then does *worse* on new data. On 400 noisy "moons" points (the code below):

| tree depth | accuracy on training data | accuracy on test data |
|---|---|---|
| unlimited | 1.00 | 0.82 |
| 3 | 0.89 | 0.88 |

The unlimited tree is perfect on what it has seen and clearly worse on what it has not — that gap is overfitting. The opposite problem, *underfitting*, is a model too simple for the pattern (a straight line through a circle): bad on both.

### Train/test split {#split}

To notice overfitting you need data the model never trained on. Sightline puts 25 % of the rows aside as the **test set** before training; the model learns from the other 75 %. For classification the split is *stratified*: each class keeps its share in both parts. The split is random but repeatable — its `random_state` (seed) is in **Advanced → Data split**. The main metric in the app is always measured on the test set.

## The formulas {#formulas}

Mean squared error over $n$ points:

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}\left(y_i - \hat y_i\right)^2$$

Log-loss for two classes:

$$\text{log-loss} = -\frac{1}{n}\sum_{i=1}^{n}\Big[y_i \ln p_i + (1 - y_i)\ln(1 - p_i)\Big]$$

Accuracy:

$$\text{accuracy} = \frac{\text{number of correct predictions}}{n}$$

R² (for regression):

$$R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}$$

where $y_i$ is the true value (or class 0/1) of point $i$, $\hat y_i$ the prediction, $p_i$ the predicted probability of class 1, $\bar y$ the mean of the true values and $n$ the number of points. R² is 1 for perfect predictions and 0 for a model that always predicts the mean.

## Where it works and where it fails {#where}

**Reading the app's charts**

- **The big plot** shows the data and what the model has learned. For classifiers the background is the probability of each class (stronger colour = more confident) and the white line is the decision border; the dashed line is the border 10 steps ago. For regressors it is the fitted line or a value heatmap. With more than two features, the plot is a 2-D PCA projection and says so in the header.
- **The Loss panel** shows the training loss (solid) and the test loss (dashed) as training goes on. Both falling = learning. Training falling while test rises = overfitting starts. Models fitted in one step show a summary instead of a curve.
- **The metric panel** is accuracy (classification), R² (regression), silhouette (clustering) or variance kept (PCA), always on the test set where there is one.
- **How it decides** walks one point through the real computation of the fitted model.
- **The timeline** under the controls replays the training after it finishes.

**Typical pitfalls**

- Comparing models on training accuracy — always compare on the test set.
- Using features on very different scales with distance- or gradient-based models (KNN, SVC, MLP, SGD) — turn on **Scale features** in Insights.
- Too little data for a flexible model — prefer simpler models or stronger regularisation.

## Parameters {#params}

Every model chapter has a table of the model's main parameters. Three settings in Sightline apply to all models:

| setting | where | what it controls | default |
|---|---|---|---|
| test share | fixed | rows kept aside for testing | 25 % (stratified for classes) |
| split seed | Advanced → Data split | which rows go to the test set | 42 |
| scale features | Insights | standardise each feature to mean 0, std 1 before training | off |

Raising the split seed changes nothing in principle, but on small data the test score can move a few points — a reminder that one test set is only an estimate.

## Python code {#code}

The script reproduces the worked examples and the overfitting table above.

{{code}}
