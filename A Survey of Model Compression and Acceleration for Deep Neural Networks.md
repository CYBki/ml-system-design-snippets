# A Survey of Model Compression and Acceleration for Deep Neural Networks

**Authors:** Yu Cheng, Duo Wang, Pan Zhou, Tao Zhang

**Published:** IEEE Signal Processing Magazine, Special Issue on Deep Learning for Image Understanding

## Abstract

Deep neural networks (DNNs) have recently achieved great success in many visual recognition tasks. However, existing deep neural network models are computationally expensive and memory intensive, hindering their deployment in devices with low memory resources or in applications with strict latency requirements. This paper reviews recent techniques for compacting and accelerating DNN models over the past five years. These techniques are divided into four categories: parameter pruning and quantization, low-rank factorization, transferred/compact convolutional filters, and knowledge distillation.

## 1. Introduction

Deep neural networks have received significant attention and achieved dramatic accuracy improvements in many tasks. These networks rely on millions or even billions of parameters, with GPU computation capability playing a key role in their success.

### Key Examples:
- **AlexNet (2012)**: 60 million parameters, five convolutional layers, three fully-connected layers
- **Face verification networks**: Hundreds of millions of parameters on LFW dataset
- **ResNet-50**: Over 95MB memory, 3.8 billion floating-point multiplications per image

### Motivation for Compression:
- Real-time applications (online learning, incremental learning)
- Virtual reality, augmented reality, smart wearable devices
- Portable devices with limited resources (memory, CPU, energy, bandwidth)
- Distributed systems, embedded devices, FPGAs

After discarding redundant weights, networks can save more than 75% of parameters and 50% computational time while maintaining performance.

## 2. Four Categories of Compression Approaches

| Category | Description | Applications | Key Features |
|----------|-------------|--------------|--------------|
| Parameter pruning and quantization | Reducing redundant parameters not sensitive to performance | Convolutional and fully connected layers | Robust, supports both training from scratch and pre-trained models |
| Low-rank factorization | Matrix/tensor decomposition to estimate informative parameters | Convolutional and fully connected layers | Standardized pipeline, easily implemented |
| Transferred/compact convolutional filters | Special structural convolutional filters to save parameters | Convolutional layers only | Application-dependent, only supports training from scratch |
| Knowledge distillation | Training compact networks with distilled knowledge | Convolutional and fully connected layers | Performance sensitive to applications, only supports training from scratch |

## 3. Parameter Pruning and Quantization

### 3.1 Quantization and Binarization

**Network Quantization** reduces the number of bits required to represent each weight:

- **K-means scalar quantization**: Applied to parameter values
- **8-bit quantization**: Significant speed-up with minimal accuracy loss
- **16-bit fixed-point representation**: Reduced memory usage and floating-point operations

#### Three-Stage Compression Method:
1. **Pruning**: Learn connectivity, prune small-weight connections
2. **Quantization**: Quantize weights using weight sharing
3. **Huffman Encoding**: Apply to quantized weights and codebook

#### Binary Neural Networks:
- **BinaryConnect**: Direct training with binary weights
- **BinaryNet**: Weights and activations constrained to +1 or -1
- **XNOR-Net**: ImageNet classification using binary convolutional networks

**Limitations**: Significant accuracy reduction on large CNNs like GoogleNet, simple matrix approximations ignore binarization effects on accuracy.

### 3.2 Network Pruning

#### Early Approaches:
- **Biased Weight Decay**
- **Optimal Brain Damage**: Hessian-based connection reduction
- **Optimal Brain Surgeon**: Second-order derivatives for network pruning

#### Modern Techniques:
- **Data-free pruning**: Remove redundant neurons
- **HashedNets**: Low-cost hash function for weight grouping
- **Deep compression**: Remove redundant connections, quantize weights, Huffman coding

#### Sparsity Constraints:
- **Group sparsity**: Applied to convolutional filters
- **Structured sparsity regularizers**: Reduce trivial filters, channels, or layers
- **L₁/L₂ norm regularizers**: For filter-level pruning

**Issues**: More iterations for convergence, manual sensitivity setup, reduces model size but not efficiency.

### 3.3 Structural Matrix Design

For fully-connected layers f(x,M) = σ(Mx), structured matrices reduce parameter storage and computation.

#### Circulant Matrices:
```
R = circ(r) = [r₀  r_{d-1}  ...  r₂  r₁]
              [r₁  r₀      r_{d-1} r₂]
              [⋮   r₁      r₀      ⋮ ]
              [r_{d-1} r_{d-2} ... r₁ r₀]
```
- Memory cost: O(d) instead of O(d²)
- Time complexity: O(d log d) using FFT

#### Adaptive Fastfood Transform:
```
R = SHGΠHB
```
Where S, G, B are random diagonal matrices, Π is permutation matrix, H is Walsh-Hadamard matrix.
- Storage cost: O(nd) → O(n)
- Computational cost: O(nd) → O(n log d)

**Drawbacks**: Structural constraints may hurt performance, difficult to find proper structural matrices.

## 4. Low-Rank Approximation and Sparsity

Convolution operations contribute bulk computation in DNNs. This approach views convolution kernels as 3D tensors and exploits structural sparsity.

### Key Techniques:
- **Separable 1D filters**: Dictionary learning approach
- **Tensor decomposition schemes**: 4.5× speedup with 1% accuracy drop
- **Canonical Polyadic (CP) decomposition**: For kernel tensors
- **Batch Normalization (BN) Low-rank**: Always-existing decomposition

### Performance Comparison on ILSVRC-2012:

| Model | Method | TOP-5 Accuracy | Speed-up | Compression Rate |
|-------|--------|----------------|----------|------------------|
| AlexNet | Baseline | 80.03% | 1× | 1× |
| | BN Low-rank | 80.56% | 1.09× | 4.94× |
| | CP Low-rank | 79.66% | 1.82× | 5× |
| VGG-16 | Baseline | 90.60% | 1× | 1× |
| | BN Low-rank | 90.47% | 1.53× | 2.72× |
| | CP Low-rank | 90.31% | 2.05× | 2.75× |

### For Fully-Connected Layers:
- **Low-rank matrix factorization**: For acoustic modeling
- **Truncated SVD**: Decompose fully connected layers
- **Multi-task architectures**: Compact deep learning designs

**Issues**: Computationally expensive decomposition, layer-by-layer approach prevents global compression, extensive retraining required.

## 5. Transferred/Compact Convolutional Filters

Based on equivariant group theory: T'Φ(x) = Φ(Tx)

### Transform Types:

#### 1. Negation Function:
```
T(Wx) = W₋x
```
- Achieves 2× compression rate
- Acts as strong regularizer

#### 2. Multi-bias Nonlinearity:
```
T'Φ(x) = Wx + δ
```
- δ represents multi-bias factors
- Generates more patterns at low computational cost

#### 3. Rotation Transforms:
```
T'Φ(x) = WTθ
```
- θ ∈ {90°, 180°, 270°} for rotation angles

#### 4. Translation Functions:
```
T'Φ(x) = T(·, x, y)
```
- Applied to 2D filters with zero padding

### Performance on CIFAR Datasets:

| Model | CIFAR-100 Error | CIFAR-10 Error | Compression Rate |
|-------|-----------------|----------------|------------------|
| VGG-16 | 34.26% | 9.85% | 1× |
| MBA | 33.66% | 9.76% | 2× |
| CRELU | 34.57% | 9.92% | 2× |
| CIRC | 35.15% | 10.23% | 4× |
| DCNN | 33.57% | 9.65% | 1.62× |

### Compact Filters:
- **3×3 → 1×1 decomposition**: Significant acceleration
- **SqueezeNet**: ~50× fewer parameters
- **MobileNets**: Adopted similar techniques

**Limitations**: Work well for wide/flat architectures (VGGNet, AlexNet) but not thin/deep ones (ResNet). Transform assumptions sometimes too strong.

## 6. Knowledge Distillation

Exploits knowledge transfer to compress models by training smaller networks to reproduce larger network outputs.

### Core Framework:
- **Student-teacher paradigm**: Student penalized according to softened teacher output
- **Class distribution learning**: Via softmax output mimicking
- **Ensemble compression**: Multiple teacher networks into single student

### Key Approaches:

#### 1. Knowledge Distillation (KD):
- Compress ensemble of teacher networks
- Student predicts both output and classification labels
- Promising results across image classification tasks

#### 2. FitNets:
- Train thin but deep networks
- Student mimics full feature maps of teacher
- Extended to allow thinner and deeper student models

#### 3. Extensions:
- **Parametric student models**: Approximate Monte Carlo teachers
- **Higher layer representation**: Knowledge using neurons in higher hidden layers
- **Function-preserving transformations**: Between network specifications
- **Attention Transfer (AT)**: Transfer attention maps as activation summaries

### Validation Results:
Tested on MNIST, CIFAR-10, CIFAR-100, SVHN, and AFLW datasets, showing competitive or superior performance with notably fewer parameters and multiplications.

**Advantages**: Makes deeper models shallower, significantly reduces computational cost.

**Disadvantages**: Limited to softmax loss functions, generally less competitive performance compared to other approaches.

## 7. Other Approaches

### Attention Mechanisms:
- **Dynamic Deep Neural Networks (D2NN)**: Execute subset of neurons based on input
- **Dynamic Capacity Networks (DCN)**: Combine small and large capacity sub-networks
- **Mixture-of-Experts (MoE)**: Sparsely-gated layers with conditional computation

### Architecture Modifications:
- **Global Average Pooling**: Replace fully-connected layers
- **Stochastic Depth**: Randomly drop layers during training, use deep networks at test time
- **Pyramidal Residual Networks**: With stochastic depth

### Computational Optimizations:
- **FFT-based convolutions**: Reduce convolutional overheads
- **Winograd algorithm**: Fast convolution implementation
- **Detail-preserving Pooling (DPP)**: Based on inverse bilateral filters
- **MobileNetV2**: Novel inverted residual structure

## 8. Benchmarks and Evaluation

### Standard Baseline Models:

| Baseline Models | Representative Works |
|-----------------|---------------------|
| AlexNet | Structural matrix, low-rank factorization |
| Network in Network | Low-rank factorization |
| VGG nets | Transferred filters, low-rank factorization |
| Residual networks | Compact filters, stochastic depth, parameter sharing |
| All-CNN-nets | Transferred filters |
| LeNets | Parameter sharing, parameter pruning |

### Evaluation Metrics:

#### Compression Rate:
```
α(M, M*) = a/a*
```
Where a = parameters in original model, a* = parameters in compressed model

#### Space Saving:
```
β(M, M*) = (a - a*)/a*
```

#### Speedup Rate:
```
δ(M, M*) = s/s*
```
Where s = running time of original model, s* = running time of compressed model

### Application-Specific Considerations:
- **Deep CNNs with fully-connected layers**: Most parameters in fully-connected layers
- **Image classification tasks**: Most operations in first few convolutional layers
- Compression focus should vary based on application requirements

## 9. Challenges and Future Directions

### 9.1 General Selection Guidelines:

#### Pre-trained Model Compression:
- Choose **pruning & quantization** or **low-rank factorization**

#### End-to-end Solutions:
- Consider **low-rank** and **transferred convolutional filters**

#### Domain-specific Applications:
- Methods with **human prior knowledge** (medical images, rotation properties)

#### Stable Performance Requirements:
- Utilize **pruning & quantization** for reasonable compression with minimal accuracy loss

#### Small/Medium Datasets:
- Try **knowledge distillation** for robust performance

#### Maximum Gain:
- **Combine multiple techniques** (e.g., convolutional layers with low-rank + fully-connected layers with pruning)

### 9.2 Technical Challenges:

1. **Limited Configuration Freedom**: Most approaches build on well-designed CNN models with limited architectural flexibility

2. **Hardware Constraints**: Various small platforms (mobile, robotic, self-driving cars) require specialized compression methods

3. **Channel Pruning Complexity**: Removing channels dramatically changes following layer inputs

4. **Prior Knowledge Control**: Structural matrix and transferred filter methods impose human knowledge affecting performance and stability

5. **Knowledge Distillation Performance**: Need to improve KD-based approaches and explore performance enhancement

6. **Interpretability**: Black box mechanisms unclear (why certain neurons/connections are pruned)

### 9.3 Future Directions:

#### Neural Architecture Search:
- **Automated hyperparameter configuration**: Reinforcement learning for efficient design space sampling
- **Hardware-aware approaches**: Incorporate hardware accelerator feedback (Hardware-Aware Automated Quantization - HAQ)

#### Channel Pruning Solutions:
- **Training-based methods**: Sparse constraints during training
- **Iterative algorithms**: Two-step channel pruning in each layer
- **Scaling factor regularization**: Automatic unimportant channel identification

#### Knowledge Distillation Enhancements:
- **New knowledge types**: Beyond direct parameter reduction
- **Selectivity knowledge**: Essential neuron selection for tasks
- **Contrastive loss**: Instead of KL divergence for distillation

#### Generalized Transformations:
- **Spatial transformation families**: Beyond predefined transformations
- **Joint learning**: Transform and model parameters simultaneously

#### Broader Applications:
- **Beyond image classification**: Video processing, vision+language, GANs
- **Natural language models**: Deep NLP model compression
- **Multi-modal applications**: Large-scale vision and language models

## 10. Conclusion

This comprehensive survey categorizes deep neural network compression and acceleration techniques into four main approaches, each addressing different aspects of the model efficiency challenge. The choice of method depends on specific application requirements, hardware constraints, and performance objectives.

Key takeaways:
- **No single best approach**: Selection depends on application needs and constraints
- **Complementary techniques**: Multiple methods can be combined for maximum benefit
- **Trade-offs exist**: Between compression rate, speedup, and accuracy preservation
- **Future focus**: Automated methods, hardware awareness, and broader application domains

The field continues to evolve with emerging challenges in mobile deployment, real-time processing, and specialized hardware acceleration, requiring continued innovation in model compression and acceleration techniques.
