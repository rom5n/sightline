# @en Gaussian naive Bayes: a bell curve per class and feature, combined with Bayes' rule.
# @ru Гауссов наивный Байес: колокол на каждый класс и признак, объединённые по правилу Байеса.
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# @en Worked example: class A at 1, 2, 3 and class B at 6, 7, 8; query x = 4.
# @ru Пример: класс A в точках 1, 2, 3 и класс B в 6, 7, 8; запрос x = 4.
x = np.array([[1], [2], [3], [6], [7], [8]])
y = np.array([0, 0, 0, 1, 1, 1])
nb = GaussianNB().fit(x, y)
print("means:", nb.theta_.ravel(), "variances:", nb.var_.ravel().round(4))  # 2, 7 and 0.6667
var = nb.var_[0, 0]


def bell(v, mean):
    return np.exp(-(v - mean) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)


lik = np.array([bell(4, 2), bell(4, 7)])
post = lik * 0.5 / (lik * 0.5).sum()
print("likelihoods:", lik.round(5), "posterior by hand:", post.round(3), "sklearn:", nb.predict_proba([[4]]).round(3))

X, y = make_classification(n_samples=600, n_features=4, n_informative=3, n_redundant=0, n_clusters_per_class=1, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# @en Defaults: var_smoothing=1e-9, priors=None (class shares of the training data).
# @ru По умолчанию: var_smoothing=1e-9, priors=None (доли классов в обучающих данных).
model = GaussianNB().fit(X_train, y_train)
print("test accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 3))
print("class priors:", model.class_prior_.round(3))

# @en Fixed priors shift the decision towards the class you say is more common.
# @ru Заданные априорные вероятности сдвигают решение к классу, который вы считаете более частым.
print("share predicted as class 0 with learned priors:", round(float(np.mean(model.predict(X_test) == 0)), 3))
skewed = GaussianNB(priors=[0.8, 0.2]).fit(X_train, y_train)
print("with priors 0.8/0.2:", round(float(np.mean(skewed.predict(X_test) == 0)), 3))
print("P(class) for one row:", model.predict_proba(X_test[:1]).round(3))
