A multi-layer perceptron (MLP) is a small neural network: layers of "neurons", each computing a weighted sum of its inputs and passing it through a nonlinear function. Stacking such layers lets it draw almost any border. It is trained by *backpropagation*: the error at the output is passed back through the layers to tell every weight which way to move. It is [logistic regression](#hb/LogisticRegression/idea) with hidden layers in between. Training basics: [the intro](#hb/intro/training).

## The idea {#idea}

Think of a factory line. The first station looks at raw parts and flags simple things ("this edge is curved", "this corner is sharp"). The next station combines those flags into bigger patterns ("two curved edges facing each other"). The last station decides "accept" or "reject". Nobody tells the stations what to look for: after each wrong decision, a supervisor walks back down the line and tells each station how much its flags contributed to the mistake, so it adjusts. Hidden layers are those stations; backpropagation is the supervisor.

## How it works, step by step {#steps}

1. **Forward pass.** For each hidden layer: $\mathbf{h} = \operatorname{ReLU}(\mathbf{x} W_1 + \mathbf{b}_1)$, where ReLU keeps positive values and turns negatives into 0. The output layer gives a score $z$, and the sigmoid (two classes) or softmax (more) turns it into probabilities.
2. **Loss.** Log-loss on the training points, plus an L2 penalty `alpha` on the weights.
3. **Backward pass.** Compute the gradient of the loss for every weight with the chain rule, from the output back to the first layer.
4. **Update.** Adjust the weights with the Adam optimiser (default), whose step size starts from `learning_rate_init`.
5. **Repeat** over mini-batches (default 200 points) and epochs until the loss stops improving or `max_iter` epochs are done. In Sightline each frame is one epoch (`partial_fit`), and the network view shows weights and activations.

**Worked example.** A 2-2-1 network with given weights, input $\mathbf{x} = (1, 2)$, true class 1.

- Hidden neuron 1: $0.5 \cdot 1 + 1.0 \cdot 2 - 0.5 = 2.0$ → ReLU → $h_1 = 2.0$.
- Hidden neuron 2: $-1.0 \cdot 1 + 0.5 \cdot 2 + 0.2 = 0.2$ → ReLU → $h_2 = 0.2$.
- Output: $z = 1.2 \cdot 2.0 - 0.8 \cdot 0.2 + 0.1 = 2.34$, $p = \sigma(2.34) \approx 0.912$.

Backward, for the log-loss $L = -\ln p$:

| quantity | formula | value |
|---|---|---|
| output error | $\delta = p - y$ | $-0.0878$ |
| gradient for $v_1$ (hidden 1 → output) | $\delta \cdot h_1$ | $-0.1757$ |
| gradient for $v_2$ | $\delta \cdot h_2$ | $-0.0176$ |
| error reaching hidden 1 | $\delta \cdot v_1 \cdot \operatorname{ReLU}'(2.0)$ | $-0.0878 \cdot 1.2 \cdot 1 = -0.1054$ |
| gradient for the weight $x_2 \to h_1$ | $-0.1054 \cdot x_2$ | $-0.2108$ |

All gradients are negative, so a step against them *increases* these weights and pushes $p$ further towards 1. A plain gradient step with rate $0.1$ would make $v_1 = 1.2 + 0.1 \cdot 0.1757 \approx 1.218$.

## The formulas {#formulas}

Forward pass with $L$ hidden layers ($\mathbf{a}_0 = \mathbf{x}$):

$$\mathbf{a}_\ell = g\!\left(\mathbf{a}_{\ell-1} W_\ell + \mathbf{b}_\ell\right), \quad \ell = 1..L, \qquad p = \sigma\!\left(\mathbf{a}_L \mathbf{v} + c\right)$$

Objective (scikit-learn):

$$J = \frac{1}{n}\sum_{i=1}^{n}\Big[-y_i\ln p_i - (1 - y_i)\ln(1 - p_i)\Big] + \frac{\alpha}{2n}\sum_{\ell}\lVert W_\ell\rVert^2$$

Backpropagation of the error $\boldsymbol{\delta}$:

$$\boldsymbol{\delta}_{\text{out}} = p - y, \qquad \boldsymbol{\delta}_{\ell} = \left(\boldsymbol{\delta}_{\ell+1} W_{\ell+1}^\top\right) \odot g'(\mathbf{z}_\ell), \qquad \frac{\partial J}{\partial W_\ell} = \mathbf{a}_{\ell-1}^\top \boldsymbol{\delta}_\ell$$

- $g$ — activation: ReLU $\max(0, z)$, tanh, logistic or identity; $g'$ — its derivative; $\odot$ — element-wise product.
- $W_\ell$, $\mathbf{b}_\ell$ — weights and biases of layer $\ell$; $\mathbf{z}_\ell$ — its values before activation.
- More than two classes: softmax output and multi-class log-loss.

## Where it works and where it fails {#where}

**Works well**

- Curved, complex borders when there is enough data: sensor patterns, handwriting, non-linear tabular relations.
- Many outputs at once and flexible architectures (deeper networks in dedicated libraries).

**Fails or misleads**

- Unscaled features: training stalls or diverges — always standardise.
- Small data: easily overfits; increase `alpha` or use early stopping.
- Results depend on the random initialisation and on several knobs (layers, rate, epochs).
- Hard to explain; for tabular data, boosted trees are often as accurate with less tuning.

## Parameters {#params}

| parameter | what it controls | default | raise | lower |
|---|---|---|---|---|
| `hidden_layer_sizes` | number of hidden layers and neurons in each | `(100,)` | more layers/neurons: more complex borders, slower, overfitting risk | simpler network, may underfit |
| `activation` | the nonlinearity $g$ | `'relu'` | `'tanh'` / `'logistic'`: smooth curves; `'identity'`: no nonlinearity, the network is linear | — |
| `learning_rate_init` | starting step size of Adam | `0.001` | faster but the loss may jump around | slower, steadier |
| `alpha` | L2 penalty on the weights | `0.0001` | smaller weights, smoother border | closer fit to the data |
| `max_iter` | maximum number of epochs | `200` | more time to converge | stops earlier (a "did not converge" warning) |

## Python code {#code}

The script checks the forward pass above, trains a network on circles, and recomputes one prediction by hand from `coefs_` and `intercepts_`.

{{code}}

## Try it in the app {#try}

On "Concentric circles" the network bends its border into a ring within a few dozen epochs. Hover a point to see which neurons fire; **How it decides** walks the forward pass layer by layer.

{{try}}
