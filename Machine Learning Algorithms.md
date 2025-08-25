# A Tour of Machine Learning Algorithms

## Summary (TL;DR)

* **Two perspectives**:

  1. **By learning style** (Supervised / Unsupervised / Semi-Supervised)
  2. **By similarity** (e.g., tree-based, neural networks, clustering, etc.)
* **Focus**: Popular methods for **classification and regression**, and where they fit.

# ---

## 1) By Learning Style

### Supervised Learning

Input data is labeled. The model predicts and learns from mistakes until a target accuracy is reached.
**Problems**: Classification, Regression
**Examples**: Logistic Regression, Backpropagation Neural Networks

### Unsupervised Learning

Input data is unlabeled. The model discovers structure (rules, clusters, compact representations).
**Problems**: Clustering, Dimensionality Reduction, Association Rule Mining
**Examples**: Apriori, K-Means

### Semi-Supervised Learning

Mix of labeled and unlabeled data. The model must organize data **and** learn to predict.
**Problems**: Classification, Regression
**Examples**: Extensions of flexible methods with assumptions for unlabeled data

# ---

## 2) By Similarity (and When to Use)

### Regression Family

Models a continuous target; iteratively improves using an error metric.
**Examples**: OLS / Linear Regression, Logistic Regression (for classification), Stepwise, MARS, LOESS
**When**: Tabular data, interpretability, roughly linear relationships.

### Instance-Based Methods

Store training instances and compare new points via a similarity metric.
**Examples**: kNN, LVQ, SOM, LWL, (listed) SVM
**When**: Simple baselines, small/medium data, good distance metric available.

### Regularization

Penalizes complexity to improve generalization (often added to regression).
**Examples**: Ridge, LASSO, Elastic Net, LARS
**When**: Many features, multicollinearity, need simple/robust models.

### Decision Trees

Branch on feature values; fast, often accurate, interpretable.
**Examples**: CART, ID3, C4.5/C5.0, CHAID, Decision Stump, M5, Conditional Trees
**When**: Tabular data, complex interactions, need interpretability.

### Bayesian Methods

Apply Bayes’ Theorem directly.
**Examples**: Naive Bayes (Gaussian/Multinomial), AODE, BBN/BN
**When**: High-dimensional sparse features (e.g., text) for simple, fast baselines.

### Clustering

Group data by similarity without labels.
**Examples**: K-Means, K-Medians, EM, Hierarchical
**When**: Exploratory analysis, segmentation, preprocessing.

### Association Rule Learning

Extract rules that explain relationships between variables (e.g., market basket).
**Examples**: Apriori, Eclat
**When**: Large transactional/log datasets to find “items bought together.”

### Artificial Neural Networks (Classical)

Neural models inspired by biology; broad family beyond deep learning.
**Examples**: Perceptron, MLP + Backprop/SGD, Hopfield, RBFN
**When**: Nonlinear relationships; adequate data and compute.

### Deep Learning

Large/complex neural nets; excels on raw data (images, audio, text).
**Examples**: CNN, RNN/LSTM, (Stacked) Autoencoders, DBM, DBN
**When**: Big data + complex patterns (vision, speech, language).

### Dimensionality Reduction

Summarize high-dimensional data while preserving structure.
**Examples**: PCA, PCR/PLS, Sammon, MDS, Projection Pursuit, LDA/MDA/QDA/FDA, t-SNE, UMAP
**When**: Visualization, noise reduction, compact features before supervised learning.

### Ensemble Methods

Combine multiple weaker models into a stronger predictor.
**Examples**: Bagging, Boosting (AdaBoost, GBM/GBRT), Random Forest, Blending, Stacking
**When**: Often the strongest tabular baselines.
**Quick difference**:

* **Bagging (e.g., Random Forest)**: Parallel training; reduces variance; robust to overfitting.
* **Boosting (e.g., GBM)**: Sequential focus on errors; reduces bias; needs careful tuning.

# ---

## Quick Starter Guide

* **Labeled tabular data, strong baseline** → **Random Forest** or **Gradient Boosting**
* **Simple / interpretable** → **Linear/Logistic Regression + Ridge/LASSO**
* **High-dimensional text** → **Naive Bayes**, then **Linear SVM/LogReg**
* **Images/signals** → **CNN**; sequences/time series/language → **RNN/LSTM**
* **Segmentation/exploration** → **K-Means** (often with **PCA/UMAP** first)
* **Many features, overfit risk** → **Regularization** and/or **tree ensembles**
* **Few labels, lots of data** → **Semi-supervised** or **representation learning + simple classifier**

# ---

## Final Notes

* “Regression” names both a **problem type** and a **method family**—mind the context.
* Not covered: **feature selection**, **evaluation metrics**, **optimization**, and subfields like **RL**, **recommenders**, **graphical models**, **NLP**, **CV**.
* In practice, strong results often come from **ensembles** or **deep learning**; reliable baselines come from **regularized linear models** and **tree-based ensembles**.
