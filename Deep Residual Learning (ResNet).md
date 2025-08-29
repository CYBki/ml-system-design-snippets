# Deep Residual Learning (ResNet) — Concise Notes

TL;DR: Very deep nets used to get worse as you added layers (the degradation problem). ResNet fixes this by letting layers learn residuals F(x)=H(x)−xF(x) = H(x) - x and adding a shortcut (skip) connection so the block outputs F(x)+xF(x) + x. This makes optimization much easier and enables 100+ layer nets that set new SOTA results.

<img width="1524" height="710" alt="image" src="https://github.com/user-attachments/assets/e18dd56e-0771-4fd5-a86f-a161c48917c8" />

## 1) Introduction &amp; Problem Statement


Observation: As depth increases, both training and test error can increase (e.g., a 56-layer net performing worse than a 20-layer one).


Name: This is the degradation problem and it is not caused by overfitting; it’s an optimization issue.



## 2) Residual Learning Approach


Idea: Instead of learning a direct mapping H(x)H(x), make layers learn the residual
F(x)=H(x)−x⇒block&nbsp;output=F(x)+xF(x) = H(x) - x \quad\Rightarrow\quad \text{block output} = F(x) + x


Why it works: If the optimal mapping is close to the identity, pushing F(x)→0F(x) \to 0 is easier than forcing layers to learn an exact identity from scratch.


Mechanism: Shortcut (identity) connections add the input xx back to the transformed signal.


### Visual — Residual (Basic) Block
flowchart LR
    X[Input x] --&gt;|skip| ADD((+))
    X --&gt; CONV1[Conv 3x3] --&gt; BN1[BatchNorm] --&gt; RELU[ReLU]
    RELU --&gt; CONV2[Conv 3x3] --&gt; BN2[BatchNorm] --&gt; ADD --&gt; OUT[ReLU]

If spatial/channel sizes differ, the skip path uses a 1×1 conv to match dimensions.

## 3) Architectural Details


Building blocks:


2-layer basic blocks (e.g., 3×3 → 3×3 with BN + ReLU).


Shortcut sums input with block output.




When sizes change: Use 1×1 conv in the shortcut to align channels/stride.


Batch Normalization: Applied after each convolution.


Bottleneck design (for 50+ layers):
1×1 → 3×3 → 1×1


First 1×1 reduces channels, 3×3 does the heavy lifting, final 1×1 restores channels.


Cuts computation while allowing very deep networks (e.g., 101/152 layers).





## 4) Experimental Results### ImageNet



Depth &amp; efficiency: ResNet-152 is ~8× deeper than VGG-19 yet less complex (fewer FLOPs/params in practice due to bottlenecks).


Accuracy: Achieved ~3.57% top-5 error in ILSVRC 2015, winning the challenge.


Effect: The degradation problem is largely resolved—deeper ⇒ better.


### CIFAR-10


Trained up to 1202 layers successfully.


Residual outputs F(x)F(x) are typically small, supporting the “near-identity” intuition.


ResNet-110 reached ~6.43% error, state-of-the-art at the time.



## 5) Object Detection &amp; Transfer


Transfer wins:


PASCAL VOC: &gt; +3% mAP vs. VGG-16 backbones.


MS COCO: ~28% relative improvement.


Topped ImageNet detection/localization and COCO detection/segmentation leaderboards when introduced.





## 6) Key Takeaways &amp; Contributions


Optimization made easy: Shortcut connections remove barriers to training very deep nets.


Identity for free: Identity mappings improve performance without extra parameters (when shapes match).


General principle: Residual learning is broadly useful—now a default building block in modern CV and beyond (e.g., NLP, speech, generative models).


Practical impact: 100+ layer networks are trainable and reliable, redefining SOTA across tasks.



## Handy One-Glance Summary (Cheat Sheet)


Problem: Deeper ≠ better (training error goes up) → degradation.


Fix: Learn residuals F(x)F(x); output F(x)+xF(x) + x via skip.


Blocks: Basic (3×3, 3×3) or Bottleneck (1×1, 3×3, 1×1).


Tricks: BN after each conv; 1×1 skip for dimension/stride changes.


Results: ImageNet win (ResNet-152); CIFAR-10 up to 1202 layers; big boosts in detection.


Impact: Residual blocks became a core primitive for deep learning.



Based on He et al., “Deep Residual Learning for Image Recognition” (2015).
