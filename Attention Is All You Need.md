# Attention Is All You Need - Detailed Block-by-Block Analysis

## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.

**Key Points:**
- Proposes the Transformer architecture that relies entirely on attention mechanisms
- Eliminates recurrent and convolutional layers completely
- Achieves state-of-the-art results on machine translation tasks
- More parallelizable and faster to train than existing models

## 1. Introduction

This section establishes the context and motivation for the Transformer architecture.

**Current State:** Recurrent Neural Networks (RNNs), particularly LSTM and GRU variants, dominate sequence modeling tasks like machine translation and language modeling.

**Problem with RNNs:** The sequential nature of RNNs prevents parallelization within training examples, which becomes critical for longer sequences due to memory constraints.

**Attention Mechanisms:** While attention has become integral to sequence modeling, it's typically used alongside recurrent networks.

**Proposed Solution:** The Transformer architecture that relies entirely on attention mechanisms, eliminating recurrence completely, allowing for greater parallelization and achieving new state-of-the-art translation quality.

## 2. Background

This section reviews related work and positions the Transformer within existing approaches.

**Previous Approaches:**
- Extended Neural GPU, ByteNet, and ConvS2S use convolutional neural networks
- These models compute hidden representations in parallel but struggle with long-range dependencies
- The number of operations to relate distant positions grows with distance

**Self-Attention:** An attention mechanism that relates different positions within a single sequence. Previously used in various NLP tasks but not as the sole mechanism for sequence transduction.

**Transformer's Innovation:** First transduction model relying entirely on self-attention without sequence-aligned RNNs or convolution.

## 3. Model Architecture

The core technical contribution of the paper - the Transformer architecture.

### Overall Structure
The Transformer follows the encoder-decoder paradigm:
- **Encoder:** Maps input sequence to continuous representations
- **Decoder:** Generates output sequence auto-regressively
- Both use stacked self-attention and feed-forward layers

### 3.1 Encoder and Decoder Stacks

**Encoder:**
- Stack of N = 6 identical layers
- Each layer has two sub-layers:
  1. Multi-head self-attention mechanism
  2. Position-wise fully connected feed-forward network
- Residual connections around each sub-layer followed by layer normalization
- All sub-layers produce outputs of dimension d_model = 512

**Decoder:**
- Also N = 6 identical layers
- Three sub-layers per layer:
  1. Masked multi-head self-attention (prevents attending to future positions)
  2. Multi-head attention over encoder output
  3. Position-wise feed-forward network
- Same residual connections and layer normalization as encoder

### 3.2 Attention Mechanism

The heart of the Transformer architecture.

**General Concept:** Attention maps a query and key-value pairs to an output, computed as a weighted sum of values.

#### 3.2.1 Scaled Dot-Product Attention

**Formula:** Attention(Q,K,V) = softmax(QK^T/√d_k)V

**Components:**
- Q (queries), K (keys), V (values) are matrices
- d_k is the dimension of the keys
- Scaling factor 1/√d_k prevents softmax saturation for large d_k

**Advantages:** Faster and more space-efficient than additive attention due to optimized matrix multiplication.

#### 3.2.2 Multi-Head Attention

**Concept:** Instead of single attention with d_model dimensions, perform h parallel attention functions with smaller dimensions.

**Process:**
1. Linearly project Q, K, V to h different representations
2. Perform attention on each projection in parallel
3. Concatenate results and project again

**Benefits:** Allows model to attend to information from different representation subspaces at different positions.

**Configuration:** h = 8 heads, d_k = d_v = d_model/h = 64

#### 3.2.3 Applications in the Model

Three types of attention layers:
1. **Encoder-decoder attention:** Queries from decoder, keys/values from encoder
2. **Encoder self-attention:** All of Q, K, V from same encoder layer
3. **Decoder self-attention:** Masked to prevent leftward information flow

### 3.3 Position-wise Feed-Forward Networks

**Structure:** Two linear transformations with ReLU activation
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

**Dimensions:** Input/output d_model = 512, inner layer d_ff = 2048

### 3.4 Embeddings and Softmax

- Learned embeddings convert tokens to d_model dimensional vectors
- Same weight matrix shared between input embeddings, output embeddings, and pre-softmax linear transformation
- Embedding weights multiplied by √d_model

### 3.5 Positional Encoding

**Problem:** Without recurrence/convolution, model has no inherent sense of sequence order.

**Solution:** Add positional encodings to input embeddings using sine/cosine functions:
- PE(pos,2i) = sin(pos/10000^(2i/d_model))
- PE(pos,2i+1) = cos(pos/10000^(2i/d_model))

**Advantages:** Allows model to learn relative positions, may extrapolate to longer sequences.

## 4. Why Self-Attention

Comparative analysis of self-attention versus other layer types.

**Comparison Criteria:**
1. Total computational complexity per layer
2. Amount of parallelizable computation
3. Path length between long-range dependencies

**Results:**
- **Self-attention:** O(n²·d) complexity, O(1) sequential operations, O(1) path length
- **Recurrent:** O(n·d²) complexity, O(n) sequential operations, O(n) path length  
- **Convolutional:** O(k·n·d²) complexity, O(1) sequential operations, O(log_k(n)) path length

**Key Advantage:** Self-attention connects all positions with constant number of sequential operations, making it faster when n < d (typical for sentence representations).

**Additional Benefits:** More interpretable models - attention heads learn different tasks and exhibit syntactic/semantic behavior.

## 5. Training

### 5.1 Training Data and Batching
- **English-German:** WMT 2014 dataset (4.5M sentence pairs), 37K token vocabulary
- **English-French:** WMT 2014 dataset (36M sentences), 32K word-piece vocabulary
- Batched by sequence length (~25K source and target tokens per batch)

### 5.2 Hardware and Schedule
- **Base models:** 8 NVIDIA P100 GPUs, 0.4 seconds per step, 100K steps (12 hours)
- **Big models:** Same hardware, 1.0 seconds per step, 300K steps (3.5 days)

### 5.3 Optimizer
- Adam optimizer (β₁=0.9, β₂=0.98, ε=10⁻⁹)
- Learning rate schedule: increases linearly for first warmup_steps, then decreases proportionally to inverse square root of step number
- warmup_steps = 4000

### 5.4 Regularization
1. **Residual Dropout:** P_drop = 0.1 applied to sub-layer outputs and embeddings
2. **Label Smoothing:** ε_ls = 0.1, improves accuracy and BLEU despite hurting perplexity

## 6. Results

### 6.1 Machine Translation

**English-to-German (WMT 2014):**
- Transformer (big): 28.4 BLEU (new state-of-the-art, +2.0 BLEU improvement)
- Transformer (base): 27.3 BLEU (surpasses all previous models at fraction of training cost)

**English-to-French (WMT 2014):**
- Transformer (big): 41.8 BLEU (outperforms all single models at <1/4 training cost)

**Training Cost:** Significantly lower computational cost compared to competitive models.

### 6.2 Model Variations

Ablation studies on English-to-German translation:

**Key Findings:**
- Single-head attention is 0.9 BLEU worse than multi-head
- Reducing attention key size d_k hurts performance
- Bigger models perform better
- Dropout is crucial for avoiding overfitting
- Sinusoidal vs. learned positional encodings perform nearly identically

### 6.3 English Constituency Parsing

**Setup:** 4-layer transformer (d_model = 1024) on Penn Treebank WSJ

**Results:**
- **WSJ only:** 91.3 F1 (competitive with discriminative parsers)
- **Semi-supervised:** 92.7 F1 (outperforms most previous models)

**Significance:** Demonstrates generalizability beyond translation tasks, even outperforming RNN sequence-to-sequence models in small-data regimes.

## 7. Conclusion

**Contributions:**
- First sequence transduction model based entirely on attention
- Replaces recurrent layers with multi-headed self-attention
- Achieves new state-of-the-art on translation tasks with significantly faster training
- More parallelizable than recurrent/convolutional architectures

**Impact:** The Transformer architecture has become the foundation for modern NLP, leading to models like BERT, GPT, and their successors.

**Future Directions:** 
- Application to other modalities (images, audio, video)
- Investigation of local, restricted attention for large inputs/outputs
- Making generation less sequential

## Key Technical Innovations

1. **Self-Attention Mechanism:** Enables direct modeling of dependencies regardless of distance
2. **Multi-Head Attention:** Allows attention to different representation subspaces
3. **Positional Encoding:** Injects sequence order information without recurrence
4. **Parallelization:** Eliminates sequential computation bottleneck of RNNs
5. **Scalability:** Architecture scales effectively with model size

## Significance and Impact

The Transformer paper introduced a paradigm shift in sequence modeling that has fundamentally changed natural language processing and machine learning. Its architecture became the foundation for:

- **BERT** (Bidirectional Encoder Representations from Transformers)
- **GPT** series (Generative Pre-trained Transformers)  
- **T5** (Text-to-Text Transfer Transformer)
- Modern large language models

The core insight that "attention is all you need" has proven remarkably prescient, with attention-based models dominating not just NLP but expanding into computer vision, speech processing, and multimodal AI systems.
