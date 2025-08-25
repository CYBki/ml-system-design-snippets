# Stacking

## What is it?
**Goal:** Combine multiple strong-but-different models with a **meta-model** so the ensemble often performs **better than any single model**.

**Architecture**
- **Level-0 (base models):** Diverse models trained on the same data (LR, KNN, Trees, SVM, NB, RF, etc.).
- **Level-1 (meta-model):** Learns how to combine base model predictions (commonly Logistic/Linear Regression).

## How is it different from Bagging/Boosting?
- **Bagging:** Same model type on resampled data; diversity from data subsampling.
- **Boosting:** Sequential weak learners correcting predecessors.
- **Stacking:** **Different model types** on the **same data**, combined by **one meta-model**.

## How is it trained?
1. Train base models with **k-fold CV** and collect **out-of-fold (OOF) predictions**.
2. Use OOF predictions (optionally with original features) to **train the meta-model**.
3. Retrain base models on **all training data**, then at inference time feed their predictions to the meta-model.

> Key idea: OOF predictions prevent leakage—meta-model never sees base models’ predictions on examples they were trained on.

---

## scikit-learn: Minimal Recipes

### Classification
```python
# StackingClassifier: pipelines avoid leakage; SVC uses probabilities for meta-model.
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier

level0 = [
    ("lr",  LogisticRegression(max_iter=1000)),
    ("knn", KNeighborsClassifier()),
    ("cart", DecisionTreeClassifier()),
    ("svm", make_pipeline(StandardScaler(), SVC(probability=True))),
    ("nb",  GaussianNB()),
]

level1 = LogisticRegression(max_iter=1000)  # simple, well-regularized meta-model

clf = StackingClassifier(
    estimators=level0,
    final_estimator=level1,
    cv=5,                 # k-fold for OOF
    passthrough=False,    # True -> meta-model sees original features too
    stack_method="auto"   # uses predict_proba/decision_function as available
)
```

### Regression
```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor

level0 = [
    ("knn", KNeighborsRegressor()),
    ("cart", DecisionTreeRegressor()),
    ("svr",  SVR()),
]
level1 = LinearRegression()

reg = StackingRegressor(
    estimators=level0,
    final_estimator=level1,
    cv=5,
    passthrough=False
)
```

---

## When does stacking help?
- Base models have **complementary strengths** and **low error correlation**.
- Several models perform similarly but excel in **different regions** of the feature space.

## When is the benefit limited?
- One model is **clearly superior**.
- Base models are **too similar** (highly correlated errors).
- **Small datasets** → noisy CV → meta-model overfits.

---

## Practical Tips
- **Diversify base models:** linear, tree-based, distance-based, margin-based, and even other ensembles.
- **Keep the meta-model simple:** Logistic/Linear Regression with regularization is a strong default.
- Consider `passthrough=True` to add original features (monitor overfitting).
- **Prevent leakage:** put all preprocessing in **Pipelines**.
- **Calibrate probabilities** if the meta-model uses them (Platt/Isotonic).
- Tune: `cv`, `passthrough`, `stack_method`, meta-model regularization, base model hyperparameters.
- Evaluate with **repeated k-fold CV** and **compare to best single model**; pick the simpler model if performance is tied.

---

## Summary
**Stacking** combines diverse models via a **simple meta-model** trained on OOF predictions, often yielding **better generalization**. scikit-learn’s `StackingClassifier`/`StackingRegressor` automate OOF construction and blending, making stacking straightforward to try. Success hinges on **diversity**, **clean CV**, and a **well-regularized meta-model**.

