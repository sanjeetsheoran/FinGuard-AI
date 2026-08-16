# FinGuard AI - Feature Engineering & Preprocessing Plan

## 1. Objective

The objective of the feature engineering stage is to transform the validated IEEE-CIS fraud detection dataset into a reliable machine-learning-ready representation while minimizing information leakage and preserving useful fraud-related signals.

The preprocessing pipeline will be designed to support reproducible experimentation and future production deployment.

---

## 2. Preprocessing Principles

The FinGuard AI preprocessing pipeline will follow these principles:

1. Prevent train-validation-test data leakage.
2. Preserve useful information contained in missingness patterns.
3. Handle numerical and categorical features using appropriate strategies.
4. Avoid blindly removing rows with missing values.
5. Handle high-cardinality categorical features carefully.
6. Preserve the ability to make predictions when identity information is unavailable.
7. Keep preprocessing reproducible.
8. Ensure that transformations used during inference are identical to those used during training.

---

## 3. Dataset Splitting Strategy

The dataset will be divided into:

- Training set
- Validation set
- Test set

The split strategy must preserve the fraud class distribution.

A stratified split will initially be considered because the fraud class represents only 3.499% of transactions.

The test set will remain isolated until final model evaluation.

---

## 4. Leakage Prevention

No target-dependent transformation will be fitted using validation or test data.

For example, target encoding must not be calculated using the complete dataset before splitting.

Correct workflow:

```text
Raw Dataset
    |
    v
Train / Validation / Test Split
    |
    +--> Fit preprocessing on Train
    |
    +--> Transform Validation
    |
    +--> Transform Test


    # FinGuard AI - Feature Engineering & Preprocessing Plan