# Cross-validation

Cross-validation is a model validation technique for assessing how statistical analysis results will generalize to independent data. It uses resampling and sample splitting methods to test and train models on different data portions across iterations.

## Purpose

Cross-validation tests a model's ability to predict new, unseen data to identify problems like overfitting or selection bias. It provides insight into how well a model will generalize to independent datasets from real-world problems.

The process involves partitioning data into complementary subsets: performing analysis on one subset (training set) and validating on another (validation/test set). Multiple rounds with different partitions are averaged to estimate predictive performance.

## Types of Cross-validation

### Exhaustive Methods

**Leave-p-out (LpO CV)**: Uses p observations as validation set, remaining as training set. Repeated for all possible combinations. Computationally intensive for large p values.

**Leave-one-out (LOOCV)**: Special case where p=1. Requires n training iterations where n is the dataset size. Less computationally expensive than LpO but still demanding for large datasets.

### Non-exhaustive Methods

**k-fold Cross-validation**: Data randomly partitioned into k equal subsamples ("folds"). Each fold serves as validation set once while remaining k-1 folds train the model. Results are averaged across k iterations. Common choice: k=10.

**Stratified k-fold**: Ensures each partition maintains similar class label proportions, especially important for imbalanced datasets.

**Holdout Method**: Simple random split into training and test sets. Single run without averaging, making results potentially unstable.

**Monte Carlo Cross-validation**: Creates multiple random training/validation splits. Results averaged over splits. Advantage: flexible split proportions. Disadvantage: some observations may never be validated.

### Nested Cross-validation

Used when simultaneously selecting hyperparameters and estimating error. Contains inner loop for hyperparameter tuning and outer loop for performance estimation, preventing optimistic bias.

## Special Considerations

**Time Series**: Random splits problematic due to temporal correlations. Rolling cross-validation more appropriate for temporal data.

**Spatial Data**: Geographic correlations require spatial blocking methods or buffered cross-validation to prevent spatial leakage between training and test sets.

## Common Mistakes

1. **Feature selection on entire dataset** before cross-validation - leads to overoptimistic results
2. **Data preprocessing on full dataset** (scaling, centering) before splitting - introduces bias
3. **Data leakage** - allowing training data into test sets through duplicate or nearly identical samples

## Applications

- **Model comparison**: Compare different algorithms (SVM vs KNN) objectively
- **Feature selection**: Determine which variables contribute to predictive performance
- **Hyperparameter tuning**: Find optimal model parameters
- **Performance estimation**: Get realistic assessment of model capabilities

## Limitations

- High computational cost for complex models
- Large variance in estimates, especially with small datasets
- Requires training and validation sets from same population
- Human bias can compromise results without proper controls
- May not reflect real-world performance if underlying system evolves over time

Cross-validation is essential for reliable model evaluation but must be implemented carefully to avoid common pitfalls that can lead to overoptimistic performance estimates.
