# Class-Balanced Loss Based on Effective Number of Samples

## Overview

This paper proposes "Class-Balanced Loss" to improve deep learning model performance on long-tailed data distributions. The approach addresses the common real-world scenario where a few classes dominate the dataset while most classes are under-represented.

## Core Problem

Real-world datasets frequently exhibit long-tailed distributions:
- A few dominant classes contain most examples
- Most classes are represented by relatively few examples
- Traditional CNNs perform poorly on weakly represented classes
- Models become biased toward dominant classes

## Key Innovation: "Effective Number of Samples"

The paper's main contribution is introducing a theoretical framework that considers **data overlap** rather than just raw sample counts:

### Core Concept
- As new data points are added, marginal benefit diminishes
- Due to similarities between samples, the "effective" number of samples is less than the actual count
- Each sample is associated with a small neighboring region instead of a single point

### Mathematical Formulation
**Effective Number Formula:**
```
E_n = (1 - β^n) / (1 - β)
```

Where:
- `n`: actual number of samples
- `β ∈ [0,1)`: hyperparameter 
- `β = (N-1)/N` where N is the number of unique prototypes in the class

### Key Properties
- When `β = 0`: E_n = 1 (all samples identical)
- When `β → 1`: E_n → n (all samples unique)
- Higher β means larger N (more unique prototypes)

## Class-Balanced Loss Function

The proposed loss function adds a weighting factor inversely proportional to the effective number of samples:

```
CB(p,y) = [(1-β)/(1-β^n_y)] × L(p,y)
```

Where:
- `p`: model's predicted class probabilities
- `y`: ground truth label
- `n_y`: number of samples in ground truth class
- `L(p,y)`: original loss function

### Loss Function Variants

The framework can be applied to multiple loss functions:

1. **Class-Balanced Softmax Cross-Entropy**
2. **Class-Balanced Sigmoid Cross-Entropy** 
3. **Class-Balanced Focal Loss**

## Experimental Results

### Datasets Tested
- **Long-Tailed CIFAR-10/100**: Artificially created with imbalance factors 10-200
- **iNaturalist 2017/2018**: Real-world long-tailed datasets (5,089 and 8,142 classes)
- **ImageNet ILSVRC 2012**: Standard benchmark (1,000 classes)

### Key Findings

#### CIFAR Experiments
- **CIFAR-10**: β = 0.9999 works best (close to inverse class frequency)
- **CIFAR-100**: Lower β values (0.99-0.999) perform better
- Fine-grained datasets require smaller β than coarse-grained ones

#### Large-Scale Results
- **iNaturalist**: Significant improvements with ResNet architectures
- **ImageNet**: Outperforms standard softmax cross-entropy
- ResNet-50 with class-balanced loss achieves comparable performance to larger models

### Performance Improvements
- CIFAR-10 (imbalance 50): 25.19% → 20.73% error rate
- CIFAR-100 (imbalance 50): 56.15% → 54.68% error rate  
- iNaturalist 2018: 42.86% → 35.21% error rate (ResNet-152)

## Technical Implementation Details

### Training Considerations
- Sigmoid-based losses require bias initialization: `b = -log((1-π)/π)`
- Remove L2 regularization for bias terms
- Learning rate decay of 0.01 instead of 0.1 for CIFAR experiments
- Linear warm-up for first 5 epochs

### Hyperparameter Selection
- β search space: {0.9, 0.99, 0.999, 0.9999}
- Loss type: {softmax, sigmoid, focal}
- Focal loss γ: {0.5, 1.0, 2.0}

## Strengths and Limitations

### Strengths
- **Theoretically grounded**: Based on random covering theory
- **Model-agnostic**: Works with different architectures and loss functions
- **Practical**: Simple formula with clear interpretation
- **Effective**: Consistent improvements across datasets
- **General**: Applicable beyond computer vision

### Limitations
- **Manual tuning**: β parameter requires dataset-specific adjustment
- **Simplified assumptions**: Binary overlap model (full overlap vs. no overlap)
- **No distribution assumptions**: Doesn't leverage prior knowledge about data distribution
- **Hyperparameter sensitivity**: Performance varies significantly with β choice

## Theoretical Insights

### Data Overlap Interpretation
The effective number captures diminishing returns:
- Each j-th sample contributes β^(j-1) to the effective number
- Total contribution: Σ(j=1 to n) β^(j-1)
- Asymptotic limit: N = 1/(1-β)

### Connection to Real-World Data
- Heavy data augmentation reduces effective N
- Near-duplicate samples in large datasets
- Fine-grained classes have fewer unique prototypes than coarse-grained ones

## Practical Guidelines

### When to Use
- Long-tailed datasets with significant class imbalance
- Real-world applications with natural class hierarchies
- When re-sampling is impractical or ineffective

### Parameter Selection Strategy
1. Start with β = 0.999 for most datasets
2. Use lower β for fine-grained classification tasks
3. Cross-validation for optimal hyperparameter selection
4. Consider dataset characteristics and class granularity

## Impact and Applications

This work has become widely adopted for handling class imbalance in:
- **Computer Vision**: Image classification, object detection
- **Natural Language Processing**: Text classification with imbalanced labels  
- **Medical AI**: Rare disease diagnosis
- **Recommendation Systems**: Long-tail item recommendation

## Conclusion

The Class-Balanced Loss provides an elegant solution to long-tailed data distribution challenges by:
1. Introducing the concept of effective sample size
2. Providing a theoretically motivated re-weighting scheme
3. Achieving consistent improvements across diverse datasets
4. Offering a general framework applicable to various loss functions

The method's success lies in its ability to balance between no re-weighting (β=0) and inverse frequency re-weighting (β→1), allowing adaptive adjustment based on data characteristics.

## References

Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S. (2019). Class-balanced loss based on effective number of samples. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 9268-9277).
