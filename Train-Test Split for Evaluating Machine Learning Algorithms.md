# Train–Test Split

## What is it?

Split the data into **train** and **test** sets, train the model on the train set, and evaluate on the **unseen** test set to estimate generalization performance. Works for classification, regression—any supervised algorithm.

## When is it appropriate?

* **Large datasets** (both train and test have enough samples to represent the domain).
* **Training is expensive** (e.g., deep nets) and you need a **quick estimate**.
* You want simple, fast baselines for initial comparisons.

> **Small data**: a single train–test split can be **noisy/misleading** → prefer **k-fold cross-validation**.

## How to configure

* **Split ratio**: common choices **80/20**, **67/33**, **50/50**. There’s no “best”; preserve representativeness in both sets.
* **Random selection**: randomize to reflect the underlying distribution.
* **Reproducibility**: set `random_state=<fixed>`.
* **Imbalanced classes** (classification): use `stratify=y` to preserve class ratios in both sets.

## Using scikit-learn (steps)

1. Prepare `X, y`.
2. Call `train_test_split(X, y, test_size=..., random_state=..., stratify=y)`.
3. Fit the model with `fit(X_train, y_train)`.
4. Predict on `X_test` and evaluate with suitable metrics (classification: accuracy, F1; regression: MAE/MSE/R²).

## Quick examples

### Classification (supports balanced/imbalanced)

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# X: features, y: labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=1, stratify=y  # stratify preserves class ratios
)

clf = RandomForestClassifier(random_state=1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

### Regression

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=1
)

reg = RandomForestRegressor(random_state=1)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
```

## Common pitfalls & tips

* ❌ **Time series**: don’t shuffle and randomly split. → Use **chronological** splits or `TimeSeriesSplit` / walk-forward validation.
* ❌ **Tuning on the test set**: keep test strictly for the **final** report. For tuning, use a separate **validation** split (e.g., 60/20/20) or **CV** / **nested CV**.
* ❌ Forgetting **stratify** with imbalanced classes → biased estimates.
* ❌ Overreliance on a **single split** (esp. small data): at least use **repeated holdout** (vary `random_state`) or **k-fold CV**.
* ✅ **Preprocessing leakage**: wrap scaling/encoding in a `Pipeline` so it’s fit **only on training** data.
* ✅ **Metric choice**: With class imbalance, prefer **F1/ROC-AUC/PR-AUC** over accuracy; for regression pick **MAE/MSE/R²** per objective.

## Quick summary

* **Train–test split** gives a **fast performance estimate** for large data and costly models.
* For small data or precise comparisons, **cross-validation** is more reliable.
* Use `random_state`, `stratify`, the **right metrics**, and a leakage-free **Pipeline** as standard hygiene.
