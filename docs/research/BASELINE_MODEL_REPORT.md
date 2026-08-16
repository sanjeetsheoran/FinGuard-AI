# FinGuard AI - Baseline Model Report

## 1. Experiment Overview

This experiment establishes the first machine learning baseline for FinGuard AI.

The purpose of the baseline is to provide a reproducible reference point against which future feature engineering, preprocessing, model, and threshold optimization experiments can be compared.

The baseline uses XGBoost with the leakage-safe preprocessing pipeline developed during the previous stages of the project.

---

## 2. Dataset

The project uses the IEEE-CIS Fraud Detection transaction dataset.

Total transactions:

590,540

Fraud distribution:

- Legitimate: 569,877
- Fraud: 20,663
- Fraud rate: 3.499%

The dataset is highly imbalanced, with legitimate transactions representing approximately 96.5% of all observations.

---

## 3. Dataset Split

The transaction dataset was split using stratification on the fraud target.

| Split | Transactions | Percentage |
|---|---:|---:|
| Train | 413,378 | 70% |
| Validation | 88,581 | 15% |
| Test | 88,581 | 15% |

The fraud rate remained approximately consistent across all splits:

| Split | Fraud Rate |
|---|---:|
| Train | 3.4990% |
| Validation | 3.4996% |
| Test | 3.4985% |

TransactionID overlap between the splits:

0

The test set was kept isolated and was not used during baseline model development.

---

## 4. Baseline Features

The baseline model uses a selected set of core transaction features.

### Transaction Features

- TransactionDT
- TransactionAmt
- ProductCD

### Card Features

- card1
- card2
- card3
- card4
- card5
- card6

### Address and Distance Features

- addr1
- addr2
- dist1
- dist2

### Email Features

- P_emaildomain
- R_emaildomain

### Verification Features

- M1
- M2
- M3
- M4
- M5
- M6
- M7
- M8
- M9

TransactionID was excluded because it is an identifier rather than a predictive feature.

The target column `isFraud` was also excluded from model inputs.

---

## 5. Preprocessing

The preprocessing pipeline was fitted only on training data.

### Numerical Features

The numerical preprocessing consists of:

- Median imputation
- Missing-value indicators

### Categorical Features

The categorical preprocessing consists of:

- Missing values replaced with `Unknown`
- One-hot encoding
- Unknown categories ignored during transformation

Processed feature dimensions:

```text
Train      : 413,378 × 181
Validation : 88,581 × 181

6. Leakage Prevention

The baseline follows the following workflow:

Raw Dataset
     |
     v
Train / Validation / Test Split
     |
     v
Fit preprocessing on TRAIN only
     |
     +------> Transform Validation
     |
     +------> Transform Test

The validation and test sets were never used to fit preprocessing parameters.

The test set was not used during baseline training or threshold analysis.

This ensures that baseline validation results are not based on information learned from the final test set.

7. Model

The baseline classifier is XGBoost.

Model configuration:

Parameter	Value
Model	XGBoost
Objective	Binary Logistic
Estimators	300
Maximum Depth	6
Learning Rate	0.08
Subsample	0.80
Column Sample by Tree	0.80
Random State	42
Evaluation Metric	PR-AUC
Tree Method	Histogram
8. Class Imbalance Handling

The dataset contains significantly fewer fraudulent transactions than legitimate transactions.

The baseline uses XGBoost's scale_pos_weight parameter.

Calculated value:

scale_pos_weight = 27.5798

This increases the importance of the minority fraud class during model training.

No SMOTE or random oversampling was used in the baseline experiment.

9. Validation Results

The baseline was evaluated on the validation set using a classification threshold of 0.50.

Metric	Result
PR-AUC	0.364275
ROC-AUC	0.904769
Precision	0.163441
Recall	0.803548
F1-Score	0.271632
10. Confusion Matrix

At a classification threshold of 0.50:

                  Predicted
              Legitimate   Fraud


Actual
Legitimate       72,731     12,750


Fraud               609      2,491

Interpretation:

True Negatives: 72,731
False Positives: 12,750
False Negatives: 609
True Positives: 2,491

The model detects a large proportion of fraudulent transactions but also produces a significant number of false-positive alerts.

11. Baseline Interpretation

The ROC-AUC of 0.904769 indicates strong ranking performance on the validation set.

The recall of 0.803548 indicates that approximately 80.35% of fraudulent transactions were detected at the 0.50 threshold.

However, precision was only 0.163441.

This means that many transactions classified as fraud were actually legitimate.

Therefore, the baseline demonstrates useful fraud-detection capability but is not yet suitable for production deployment.

12. Threshold Analysis

The baseline probabilities were evaluated using multiple classification thresholds.

Threshold	Precision	Recall	F1
0.10	0.0467	0.9919	0.0892
0.20	0.0673	0.9626	0.1257
0.30	0.0932	0.9242	0.1693
0.40	0.1247	0.8668	0.2181
0.50	0.1634	0.8035	0.2716
0.60	0.2112	0.7203	0.3266
0.70	0.2717	0.6161	0.3771
0.80	0.3436	0.4719	0.3977
0.90	0.4983	0.2365	0.3207
13. F1-Optimal Threshold

The highest F1-score in the evaluated threshold range occurred at:

Threshold: 0.80
Precision: 0.3436
Recall:    0.4719
F1:        0.3977

This represents the best F1 result among the tested thresholds.

However, this threshold is not automatically considered the production threshold.

The appropriate production threshold depends on the relative business cost of:

False positives
False negatives
14. Precision-Recall Trade-off

Increasing the classification threshold results in:

Higher precision
Lower recall

For example:

Threshold 0.50
Precision = 16.34%
Recall    = 80.35%

while:

Threshold 0.80
Precision = 34.36%
Recall    = 47.19%

Therefore, threshold selection is a business decision as well as a machine-learning optimization problem.

15. Primary Evaluation Metric

Accuracy is not considered the primary metric because the dataset is highly imbalanced.

The primary evaluation metrics for FinGuard AI are:

PR-AUC
Recall
Precision
F1-Score

Secondary metrics include:

ROC-AUC
False Positive Rate
False Negative Rate
Confusion Matrix
16. Test Set Policy

The test set has not been used during baseline development.

It will remain isolated until the final model and evaluation strategy are selected.

The final test evaluation will be performed only after:

Feature engineering
Model selection
Hyperparameter tuning
Threshold selection

have been completed.

17. Saved Artifacts

The baseline experiment generated the following artifacts:

ml/
├── models/
│   └── baseline/
│       ├── xgboost_baseline.json
│       └── preprocessor.joblib
│
└── metrics/
    ├── baseline_metrics.json
    └── threshold_analysis.csv

These artifacts represent the current baseline experiment.

18. Current Limitations

The baseline model has several limitations.

18.1 Limited Feature Set

The baseline uses a selected group of core transaction features rather than the complete IEEE-CIS feature space.

18.2 High False-Positive Rate

At threshold 0.50, the model produces 12,750 false-positive predictions on the validation set.

18.3 Threshold Not Optimized for a Business Objective

The 0.80 threshold provides the best F1 among the tested thresholds, but a production threshold requires a business cost analysis.

18.4 No Advanced Feature Engineering

The baseline does not yet include:

Frequency encoding
Target encoding
Advanced temporal features
Advanced interaction features
Identity-derived features
Feature selection
SHAP-based analysis
18.5 No Hyperparameter Optimization

The current XGBoost parameters are baseline parameters and have not been optimized through systematic experimentation.

19. Next Experiments

Future experiments will evaluate improvements systematically.

Planned experiments include:

Transaction amount feature engineering
Temporal feature engineering
Identity dataset integration
Missingness-based features
Frequency encoding
Selected interaction features
Feature selection
XGBoost hyperparameter optimization
Threshold optimization using business objectives
Model comparison

Each experiment will be compared against this baseline.

20. Experiment Tracking Principle

Every model experiment should record:

Dataset version
Feature set
Preprocessing configuration
Model configuration
Random seed
Validation metrics
Threshold
Confusion matrix
Experiment conclusion

This ensures that model improvements are measurable and reproducible.

21. Conclusion

The XGBoost baseline establishes a reliable starting point for FinGuard AI.

The model achieved:

PR-AUC    = 0.364275
ROC-AUC   = 0.904769
Recall    = 0.803548
Precision = 0.163441
F1        = 0.271632

The baseline demonstrates that the available transaction features contain meaningful fraud-related signals.

However, the current false-positive rate and precision indicate that additional feature engineering, model optimization, and threshold analysis are required before the system can be considered production-ready.

The baseline will therefore serve as the reference point for all subsequent experiments.