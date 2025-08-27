# BLEU Metric: Complete Guide with Examples

## What is BLEU?

BLEU (Bilingual Evaluation Understudy) is an algorithm for evaluating the quality of machine-translated text from one natural language to another. The core principle is simple: "The closer a machine translation is to a professional human translation, the better it is."

## Key Features

**Scoring System:**
- BLEU score is always a number between 0 and 1
- Values closer to 1 indicate the translation is more similar to reference translations
- Even human translations rarely achieve a perfect score of 1, as this would mean being identical to one reference translation

**Evaluation Method:**
- Translated segments (typically sentences) are compared with high-quality reference translations
- Scores are averaged across the entire corpus to estimate overall translation quality
- Intelligibility and grammatical correctness are not explicitly considered

## How BLEU Works: Step-by-Step with Examples

### Step 1: Basic Precision Problem

Let's start with a problematic example:

**Candidate Translation:** "the the the the the the the"
**Reference 1:** "the cat is on the mat"
**Reference 2:** "there is a cat on the mat"

Using simple precision:
- All 7 words in the candidate appear in the references
- Precision = 7/7 = 1.0 (perfect score!)
- But this is clearly a terrible translation

### Step 2: Modified N-gram Precision

BLEU fixes this by using **clipping**:

1. Count maximum occurrences of each word in any reference:
   - "the" appears maximum 2 times (in reference 1)
   - So max_count("the") = 2

2. Clip the candidate word counts:
   - "the" appears 7 times in candidate
   - Clipped to min(7, 2) = 2

3. Calculate modified precision:
   - Modified precision = 2/7 ≈ 0.29

### Step 3: N-gram Analysis

BLEU evaluates multiple n-gram lengths (typically 1-4):

**Example with candidate "the cat":**

| N-gram Type | Candidate N-grams | Reference Matches | Score |
|-------------|-------------------|-------------------|--------|
| Unigram (1) | "the", "cat" | Both match | 2/2 = 1.0 |
| Bigram (2) | "the cat" | Matches "the cat" in ref | 1/1 = 1.0 |

**Example with candidate "the the cat":**

| N-gram Type | Candidate N-grams | Calculation | Score |
|-------------|-------------------|-------------|--------|
| Unigram | "the"×2, "cat"×1 | min(2,2) + min(1,1) = 3 → 3/3 = 1.0 |
| Modified Unigram | "the"×2, "cat"×1 | clipped: 2 + 1 = 3 → 2/3 = 0.67 |
| Bigram | "the the", "the cat" | 0 + 1 = 1 → 1/2 = 0.5 |

### Step 4: Brevity Penalty

BLEU penalizes translations that are too short:

**Formula:** BP = e^(1-r/c) if c ≤ r, otherwise BP = 1

Where:
- c = length of candidate corpus
- r = length of effective reference corpus (closest length reference)

**Example:**
- Candidate: "cat" (length = 1)
- Best matching reference: "the cat is on the mat" (length = 6)
- r/c = 6/1 = 6
- BP = e^(1-6) = e^(-5) ≈ 0.007

This severely penalizes the very short translation.

### Step 5: Final BLEU Score

**Formula:** 
BLEU = BP × exp(Σ wₙ × log(pₙ))

Where:
- BP = Brevity Penalty
- wₙ = weight for n-gram (typically w₁ = w₂ = w₃ = w₄ = 0.25)
- pₙ = modified n-gram precision for n-grams

**Complete Example:**

**Candidate:** "The cat is on the mat"
**Reference:** "The cat is on the mat"

1. **N-gram precisions:**
   - p₁ = 6/6 = 1.0 (all unigrams match)
   - p₂ = 5/5 = 1.0 (all bigrams match)
   - p₃ = 4/4 = 1.0 (all trigrams match)
   - p₄ = 3/3 = 1.0 (all 4-grams match)

2. **Brevity penalty:**
   - c = 6, r = 6
   - BP = 1 (no penalty for equal length)

3. **BLEU score:**
   - BLEU = 1 × exp(0.25×log(1) + 0.25×log(1) + 0.25×log(1) + 0.25×log(1))
   - BLEU = 1 × exp(0) = 1.0

## Mathematical Components Explained

### Modified N-gram Precision
For each n-gram in the candidate:
- Count how many times it appears in candidate
- Count maximum appearances in any reference
- Take the minimum of these two counts
- Sum across all unique n-grams and normalize

### Brevity Penalty
- Prevents gaming the system with very short translations
- Only penalizes when candidate is shorter than reference
- Uses exponential decay for increasingly short translations

### Geometric Mean
- Combines different n-gram precisions
- Ensures good performance across all n-gram lengths
- More sensitive to poor performance in any single n-gram length

## Implementation Example

```python
def calculate_bleu_unigram_example():
    candidate = ["the", "cat", "is", "on", "mat"]
    reference = ["the", "cat", "is", "on", "the", "mat"]
    
    # Count occurrences
    candidate_counts = {"the": 1, "cat": 1, "is": 1, "on": 1, "mat": 1}
    reference_counts = {"the": 2, "cat": 1, "is": 1, "on": 1, "mat": 1}
    
    # Apply clipping
    clipped_counts = {}
    for word in candidate_counts:
        clipped_counts[word] = min(
            candidate_counts[word], 
            reference_counts.get(word, 0)
        )
    
    # Calculate precision
    numerator = sum(clipped_counts.values())  # 1+1+1+1+1 = 5
    denominator = len(candidate)  # 5
    precision = numerator / denominator  # 5/5 = 1.0
    
    return precision
```

## Strengths and Limitations

### Strengths
- Fast and inexpensive to compute
- Language-agnostic (mostly)
- High correlation with human judgments
- Widely adopted standard
- Good for comparing systems

### Limitations
- Cannot handle languages without word boundaries
- Heavily dependent on tokenization
- No guarantee that higher BLEU means better translation
- Focuses on precision, ignores recall
- Often used with single reference (though designed for multiple)

## Variants and Improvements

- **SacreBLEU**: Addresses tokenization issues for better reproducibility
- **iBLEU**: Interactive version for visual examination
- **Character-level BLEU**: For languages without clear word boundaries

BLEU remains a fundamental benchmark in machine translation evaluation, but should be used alongside other metrics and human evaluation for comprehensive assessment.
