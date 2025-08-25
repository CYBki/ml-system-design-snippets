# Bootstrap Aggregating (Bagging)

## 📝 Definition
Bootstrap Aggregating (Bagging) is a machine learning ensemble meta-algorithm that improves stability and accuracy by combining multiple models trained on bootstrap samples of the original dataset.
<img width="708" height="476" alt="Bootstrap" src="https://github.com/user-attachments/assets/7923d9d5-9809-4872-9c96-48e48d09593b" />

## 🔄 How It Works
1. **Bootstrap Sampling**: Create `m` training sets by sampling with replacement from original dataset
2. **Model Training**: Train a model on each bootstrap sample
3. **Aggregation**: Combine predictions by averaging (regression) or voting (classification)

## 📊 Key Statistics
- Each bootstrap sample contains ~63.2% unique samples (rest are duplicates)
- Formula: `1 - 1/e ≈ 0.632` for large datasets

## 🌳 Common Applications
- **Random Forests**: Bagging + random feature selection
- **Decision Trees**: Primary use case
- **Neural Networks**: Improves unstable procedures

## Advantages
- Reduces overfitting and variance
- Improves model stability
- Parallelizable training
- Works well with high-variance, low-bias learners

## Disadvantages
- High bias models remain biased
- Loss of interpretability
- Computationally expensive
- Cannot predict beyond training data range

## 🏗️ Algorithm (Classification)
```python
# Pseudocode
for i in range(m):
    D_i = bootstrap_sample(D)  # Sample with replacement
    C_i = train_model(D_i)     # Train classifier
    
C_final(x) = majority_vote([C_1(x), C_2(x), ..., C_m(x)])
```

## 📈 Best Practices
- Specify maximum tree depth to prevent overfitting
- Balance accuracy vs speed with number of models
- Prune datasets for better performance
- Use with unstable learning algorithms

## 🔗 Related Concepts
- Boosting
- Random Forest
- Out-of-bag error
- Cross-validation

---
*Originally developed by Leo Breiman (1994) based on Bradley Efron's bootstrapping concept*
