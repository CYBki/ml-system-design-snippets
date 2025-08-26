# Machine Learning Optimization and Activation Functions Documentation

This repository contains concise documentation for key optimization algorithms and activation functions used in machine learning and deep learning.

## Optimization Algorithms

### 1. AdaGrad (Adaptive Gradient)

**Overview:**
AdaGrad is a family of adaptive learning rate algorithms designed for stochastic optimization. It adapts the learning rate for each parameter individually based on the historical gradients.

**Key Features:**
- **Adaptive Learning Rates**: Different learning rates for each parameter
- **Second-order Information**: Uses accumulated squared gradients as Hessian approximation
- **Sparse Gradient Optimization**: Particularly effective for problems with sparse features

**Algorithm:**
```
G_t = G_{t-1} + g_t ⊙ g_t
θ_{t+1} = θ_t - η / √(G_t + ε) ⊙ g_t
```

Where:
- `G_t`: Accumulated squared gradients
- `g_t`: Current gradient
- `η`: Learning rate
- `ε`: Small constant for numerical stability

**Advantages:**
- Higher learning rates for infrequent features
- Lower step sizes in high-curvature directions
- No manual tuning of learning rate decay

**Disadvantages:**
- Learning rate can approach zero over time
- Sensitive to initial conditions
- May stop learning before reaching optimal solution

**Applications:**
- Neural networks with sparse gradients
- Natural language processing tasks
- Recurrent neural networks

---

### 2. Momentum

**Overview:**
Momentum is an extension to gradient descent that builds inertia in the search direction to overcome local minima and reduce oscillations in noisy gradients.

**Key Features:**
- **Exponentially Weighted Gradients**: Uses history of past gradients
- **Oscillation Reduction**: Smooths out gradient updates
- **Escape Local Minima**: Helps overcome small hills in loss landscape

**Algorithm:**
```
v_t = βv_{t-1} + (1-β)g_t
θ_{t+1} = θ_t - ηv_t
```

Where:
- `v_t`: Velocity (momentum term)
- `β`: Momentum coefficient (typically 0.9)
- `g_t`: Current gradient
- `η`: Learning rate

**Advantages:**
- Faster convergence than standard gradient descent
- Reduces oscillations in gradient descent
- Can escape shallow local minima
- Works well with high-curvature functions

**Disadvantages:**
- Introduces additional hyperparameter (momentum coefficient)
- May overshoot the target if momentum is too high
- Requires tuning for optimal performance

**Applications:**
- Deep neural networks
- Non-convex optimization problems
- Ridge and logistic regression
- Support vector machines

---

### 3. RMSProp (Root Mean Square Propagation)

**Overview:**
RMSProp is an adaptive learning rate algorithm that addresses the diminishing learning rate problem in AdaGrad by using a moving average of squared gradients.

**Key Features:**
- **Adaptive Learning Rate**: Individual learning rates for each parameter
- **Moving Average**: Uses exponential decay instead of accumulating all gradients
- **Mini-batch Friendly**: Works well with stochastic mini-batch training

**Algorithm:**
```
v_t = ρv_{t-1} + (1-ρ)g_t²
θ_{t+1} = θ_t - η/√(v_t + ε) * g_t
```

Where:
- `v_t`: Moving average of squared gradients
- `ρ`: Decay rate (typically 0.9)
- `g_t`: Current gradient
- `η`: Learning rate
- `ε`: Small constant for numerical stability

**Advantages:**
- Prevents learning rate from diminishing to zero
- Adapts learning rate based on gradient magnitude
- Works well with non-stationary objectives
- Suitable for online and mini-batch learning

**Disadvantages:**
- Still requires manual tuning of global learning rate
- Can be sensitive to hyperparameter choices
- May converge slower than more advanced optimizers like Adam

**Applications:**
- Neural network training
- Recurrent neural networks
- Non-convex optimization problems
- Online learning scenarios

---

## Activation Functions

### 1. ELU (Exponential Linear Unit)

**Overview:**
ELU is an activation function that converges cost to zero faster and produces more accurate results than ReLU, with smooth negative values.

**Formula:**
```
ELU(x) = {
  x           if x > 0
  α(e^x - 1)  if x ≤ 0
}
```

**Characteristics:**
- **Output Range**: (-α, ∞)
- **Smooth**: Differentiable everywhere
- **Zero-centered**: Mean activation closer to zero

**Advantages:**
- Smooth transition for negative values
- Can produce negative outputs (unlike ReLU)
- Reduces vanishing gradient problem
- Faster convergence than ReLU

**Disadvantages:**
- Computationally more expensive than ReLU
- Can blow up activations for positive inputs
- Requires tuning of α parameter

**Use Cases:**
- Deep neural networks
- When zero-centered activations are desired
- Alternative to ReLU in hidden layers

---

### 2. ReLU (Rectified Linear Unit)

**Overview:**
ReLU is the most popular activation function in deep learning, defined as the positive part of its argument.

**Formula:**
```
ReLU(x) = max(0, x)
```

**Characteristics:**
- **Output Range**: [0, ∞)
- **Non-linear**: Despite simple appearance
- **Sparse**: Many neurons output zero

**Advantages:**
- Computationally efficient
- Mitigates vanishing gradient problem
- Promotes sparsity in the network
- Simple to implement and understand

**Disadvantages:**
- Dying ReLU problem (neurons can become inactive)
- Unbounded output can cause exploding activations
- Not zero-centered
- Only suitable for hidden layers

**Variants:**
- **Leaky ReLU**: Allows small negative slope
- **Parametric ReLU**: Learnable negative slope

**Use Cases:**
- Hidden layers in deep neural networks
- Convolutional neural networks
- When computational efficiency is important

---

### 3. Tanh (Hyperbolic Tangent)

**Overview:**
Tanh is a zero-centered activation function that squashes inputs to the range [-1, 1].

**Formula:**
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

**Characteristics:**
- **Output Range**: (-1, 1)
- **Zero-centered**: Better than sigmoid
- **Smooth**: Continuously differentiable

**Derivative:**
```
tanh'(x) = 1 - tanh²(x)
```

**Advantages:**
- Zero-centered output
- Stronger gradients than sigmoid
- Smooth and differentiable
- Bounded output prevents exploding activations

**Disadvantages:**
- Suffers from vanishing gradient problem
- Computationally more expensive than ReLU
- Can saturate for large input values

**Use Cases:**
- Hidden layers when zero-centered activation is needed
- Recurrent neural networks (LSTM, GRU)
- When bounded activation is required

---

### 4. Softmax

**Overview:**
Softmax is a generalization of the sigmoid function that converts a vector of real numbers into a probability distribution.

**Formula:**
```
Softmax(x_i) = e^(x_i) / Σ(e^(x_j)) for j=1 to n
```

**Characteristics:**
- **Output Range**: (0, 1) for each element
- **Sum to One**: All outputs sum to 1
- **Probability Distribution**: Represents class probabilities

**Properties:**
- Differentiable
- Monotonic
- Emphasizes the largest input values

**Advantages:**
- Natural interpretation as probabilities
- Smooth gradients for backpropagation
- Works well with cross-entropy loss
- Handles multi-class classification naturally

**Disadvantages:**
- Can suffer from vanishing gradients
- Sensitive to outliers in input
- Computationally intensive for large vocabularies

**Use Cases:**
- Output layer for multi-class classification
- Attention mechanisms in transformers
- Policy networks in reinforcement learning
- Language modeling (next token prediction)

---

## Summary

### Optimization Algorithm Comparison

| Algorithm | Learning Rate | Memory | Best For |
|-----------|---------------|---------|----------|
| AdaGrad | Adaptive | O(d) | Sparse features |
| Momentum | Fixed | O(d) | Smooth convergence |
| RMSProp | Adaptive | O(d) | Non-stationary problems |

### Activation Function Comparison

| Function | Range | Smooth | Zero-centered | Use Case |
|----------|-------|--------|---------------|----------|
| ELU | (-α, ∞) | Yes | Nearly | Hidden layers |
| ReLU | [0, ∞) | No | No | Hidden layers |
| Tanh | (-1, 1) | Yes | Yes | Hidden/RNN layers |
| Softmax | (0, 1) | Yes | No | Output layer |

---

## References

1. Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization.
2. Tieleman, T. and Hinton, G. (2012). Lecture 6.5 - RMSProp, Neural Networks for Machine Learning.
3. Glorot, X., Bordes, A., & Bengio, Y. (2011). Deep sparse rectifier neural networks.
4. Clevert, D. A., Unterthiner, T., & Hochreiter, S. (2015). Fast and accurate deep network learning by exponential linear units.

