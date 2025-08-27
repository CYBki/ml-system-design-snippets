# METEOR Metric: Complete Guide with Examples

## What is METEOR?

METEOR (Metric for Evaluation of Translation with Explicit ORdering) is a metric for evaluating machine translation output. It was designed to address problems found in the popular BLEU metric and provide better correlation with human judgment.

## Key Features

### Differences from BLEU
- **Precision and recall**: Uses harmonic mean of unigram precision and recall
- **Recall weighting**: Recall is weighted 9 times more than precision
- **Advanced matching**: Includes stemming and synonymy matching alongside exact word matching
- **Sentence-level correlation**: While BLEU seeks corpus-level correlation, METEOR targets sentence-level correlation

### Performance Comparison
- Up to 0.964 correlation with human judgment at corpus level (BLEU: 0.817)
- Maximum 0.403 correlation at sentence level

## How METEOR Works

### Stage 1: Alignment

METEOR creates unigram mappings between candidate and reference translations:

**Constraints:**
- Each candidate unigram maps to zero or one reference unigram
- When multiple alignments have the same number of mappings, choose the one with fewest crossings

**Matching Modules:**

| Module | Candidate | Reference | Match |
|--------|-----------|-----------|-------|
| Exact | Good | Good | Yes |
| Stemmer | Goods | Good | Yes |
| Synonymy | well | Good | Yes |

### Stage 2: Precision and Recall

**Unigram Precision (P):**
```
P = m / wₜ
```
- m = number of unigrams in candidate that are found in reference
- wₜ = total number of unigrams in candidate

**Unigram Recall (R):**
```
R = m / wᵣ
```
- m = same as above
- wᵣ = total number of unigrams in reference

### Stage 3: Harmonic Mean

```
Fₘₑₐₙ = (10 × P × R) / (R + 9 × P)
```

This formula weights recall 9 times more than precision.

### Stage 4: Fragmentation Penalty

**Chunk definition:** A set of unigrams that are adjacent in both hypothesis and reference.

**Penalty calculation:**
```
p = 0.5 × (c / uₘ)³
```
- c = number of chunks
- uₘ = number of matched unigrams

### Stage 5: Final Score

```
M = Fₘₑₐₙ × (1 - p)
```

## Detailed Examples

### Example 1: Word Order Change

**Reference:** "the cat sat on the mat"
**Hypothesis:** "on the mat sat the cat"

**Step-by-step calculation:**

1. **Matching:** 6/6 words matched
2. **Precision:** P = 6/6 = 1.0000
3. **Recall:** R = 6/6 = 1.0000
4. **Fₘₑₐₙ:** (10×1.0×1.0)/(1.0+9×1.0) = 10/10 = 1.0000
5. **Chunk analysis:**
   - Chunks: ["on"], ["the mat"], ["sat"], ["the cat"]
   - c = 3 chunks, uₘ = 6 matches
   - Fragmentation = 3/6 = 0.5
6. **Penalty:** p = 0.5 × (0.5)³ = 0.5 × 0.125 = 0.0625
7. **Final score:** M = 1.0000 × (1 - 0.0625) = **0.9375**

### Example 2: Perfect Match

**Reference:** "the cat sat on the mat"
**Hypothesis:** "the cat sat on the mat"

**Calculation:**

1. **Matching:** 6/6 words matched
2. **Precision:** P = 1.0000
3. **Recall:** R = 1.0000
4. **Fₘₑₐₙ:** 1.0000
5. **Chunk analysis:**
   - Single chunk: ["the cat sat on the mat"]
   - c = 1, uₘ = 6
   - Fragmentation = 1/6 = 0.1667
6. **Penalty:** p = 0.5 × (0.1667)³ = 0.0023
7. **Final score:** M = 1.0000 × (1 - 0.0023) = **0.9977**

### Example 3: Missing Word

**Reference:** "the cat sat on the mat"
**Hypothesis:** "the cat was sat on the mat"

**Calculation:**

1. **Matching:** 6/7 words matched ("was" doesn't match)
2. **Precision:** P = 6/7 = 0.8571
3. **Recall:** R = 6/6 = 1.0000
4. **Fₘₑₐₙ:** (10×0.8571×1.0)/(1.0+9×0.8571) = 8.571/8.714 = 0.9836
5. **Chunk analysis:**
   - Chunks: ["the cat"], ["sat on the mat"]
   - c = 2, uₘ = 6
   - Fragmentation = 2/6 = 0.3333
6. **Penalty:** p = 0.5 × (0.3333)³ = 0.0185
7. **Final score:** M = 0.9836 × (1 - 0.0185) = **0.9654**

## Advantages of METEOR

### Superiority over BLEU

1. **Includes recall:** Measures both precision and recall, not just precision
2. **Advanced matching:** Features stemming and synonymy capabilities
3. **Sentence-level correlation:** Better correlation with human judgment at sentence level
4. **Word order penalty:** Considers word order through fragmentation penalty

### Practical Benefits

- More balanced evaluation (precision + recall)
- Recognition of morphological variations
- Understanding of synonymous words
- More reliable sentence-level scores

## Limitations

1. **Computational cost:** More complex than BLEU
2. **Language dependency:** Stemming and synonymy modules are language-specific
3. **Parameter tuning:** Weighting parameters need to be adjusted

## Use Cases

- Evaluation of machine translation systems
- Sentence-level translation quality measurement
- Detailed analysis during system development
- Situations requiring high correlation with human evaluation

METEOR provides a powerful alternative to BLEU, especially in areas where BLEU falls short, offering a more comprehensive and balanced approach to machine translation evaluation.
