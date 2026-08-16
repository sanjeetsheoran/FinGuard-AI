# FinGuard AI - Feature Experiment 1 Report

## 1. Experiment Overview

This experiment evaluates whether transaction amount and relative transaction time features improve the FinGuard AI fraud detection baseline.

The experiment keeps the following components unchanged:

- Dataset split
- Training data
- Validation data
- XGBoost model configuration
- Class imbalance strategy
- Evaluation metrics

Only the feature set is changed.

This provides a controlled comparison between the baseline model and Experiment 1.

---

## 2. Experiment Objective

The objective is to determine whether the following engineered features improve fraud detection performance:

1. `TransactionAmtLog`
2. `TransactionAmtBucket`
3. `TransactionHour`
4. `TransactionDay`

The experiment does not use `isFraud` to construct these features.

Therefore, the engineered features do not introduce direct target leakage.

---

## 3. Feature Engineering

### 3.1 TransactionAmtLog

The original transaction amount is highly right-skewed.

A logarithmic transformation is applied:

```text
TransactionAmtLog = log(1 + TransactionAmt)

3.2 TransactionAmtBucket

Transaction amounts are grouped into categorical ranges:

Range	Category
≤ 25	very_low
25–50	low
50–100	medium
100–250	high
250–500	very_high
500–1000	premium
> 1000	extreme

This allows the model to learn non-linear relationships between transaction amount ranges and fraud risk.

3.3 TransactionHour

TransactionDT represents elapsed transaction time.

The relative hour of the transaction is derived using:

TransactionHour = (TransactionDT / 3600) % 24

This provides a time-of-day signal to the model.

3.4 TransactionDay

A relative transaction day is derived from TransactionDT:

TransactionDay = TransactionDT / 86400

This provides the model with a coarse temporal progression signal.

4. Dataset

The experiment uses the same IEEE-CIS transaction dataset as the baseline.

Total transactions:

590,540

Fraud distribution:

Legitimate = 569,877
Fraud      = 20,663

Fraud rate:

3.499%
5. Dataset Split

The predefined stratified split is reused.

Split	Transactions	Percentage
Train	413,378	70%
Validation	88,581	15%
Test	88,581	15%

The test set remains completely isolated.

6. Preprocessing

The preprocessing pipeline is fitted using training data only.

TRAIN
  ↓
fit preprocessing
  ↓
Validation → transform only
Test       → transform only

Numerical preprocessing:

Median imputation
Missing-value indicators

Categorical preprocessing:

Missing values → Unknown
One-hot encoding
Unknown categories ignored
7. Feature Dimensions

Baseline preprocessing produced:

Train      : 413,378 × 181
Validation : 88,581 × 181

Experiment 1 preprocessing produced:

Train      : 413,378 × 191
Validation : 88,581 × 191

The increase in processed features comes from the additional engineered features and their categorical representation.

8. Model Configuration

The same XGBoost configuration used by the baseline was retained.

Parameter	Value
Model	XGBoost
Estimators	300
Maximum Depth	6
Learning Rate	0.08
Subsample	0.80
Column Sample by Tree	0.80
Random State	42
Objective	Binary Logistic
Evaluation Metric	AUC-PR
Tree Method	Histogram

Class imbalance handling:

scale_pos_weight = 27.5798

No SMOTE or random oversampling was introduced.

9. Baseline Results

The baseline model produced the following validation results at threshold 0.50:

Metric	Baseline
PR-AUC	0.364275
ROC-AUC	0.904769
Precision	0.163441
Recall	0.803548
F1-Score	0.271632
10. Experiment 1 Results

Experiment 1 produced the following validation results at threshold 0.50:

Metric	Experiment 1
PR-AUC	0.372530
ROC-AUC	0.905054
Precision	0.165656
Recall	0.802258
F1-Score	0.274609
11. Baseline vs Experiment 1
Metric	Baseline	Experiment 1	Change
PR-AUC	0.364275	0.372530	+0.008256
ROC-AUC	0.904769	0.905054	+0.000285
Precision	0.163441	0.165656	+0.002216
Recall	0.803548	0.802258	-0.001290
F1-Score	0.271632	0.274609	+0.002978
12. Primary Result

The primary metric for the experiment is PR-AUC.

Experiment 1 improved PR-AUC:

Baseline:
0.364275


Experiment 1:
0.372530

Absolute improvement:

+0.008256

Approximate relative improvement:

+2.27%

Therefore, the engineered feature set provides a measurable improvement over the baseline.

13. ROC-AUC Result

ROC-AUC improved slightly:

0.904769 → 0.905054

Absolute improvement:

+0.000285

The improvement is small compared with the PR-AUC improvement.

This indicates that the new features provide a stronger improvement in the precision-recall region than in overall ranking measured by ROC-AUC.

14. Precision and Recall

At threshold 0.50:

Baseline
Precision = 16.34%
Recall    = 80.35%
Experiment 1
Precision = 16.57%
Recall    = 80.23%

Precision improved slightly while recall decreased slightly.

This represents a small shift toward reducing false-positive predictions while maintaining approximately the same fraud detection recall.

15. Confusion Matrix Comparison
Baseline
                  Predicted
              Legitimate   Fraud


Actual
Legitimate       72,731     12,750


Fraud               609      2,491
Experiment 1
                  Predicted
              Legitimate   Fraud


Actual
Legitimate       72,955     12,526


Fraud               613      2,487

Experiment 1 produced:

False positives:
12,750 → 12,526


Change:
-224

False negatives changed from:

609 → 613

Therefore, Experiment 1 reduced false positives at the cost of a small increase in false negatives.

16. Threshold Analysis

Experiment 1 probabilities were evaluated using multiple thresholds.

Threshold	Precision	Recall	F1
0.10	0.0470	0.9919	0.0898
0.20	0.0681	0.9603	0.1271
0.30	0.0936	0.9168	0.1699
0.40	0.1266	0.8684	0.2210
0.50	0.1657	0.8023	0.2746
0.60	0.2135	0.7139	0.3287
0.70	0.2782	0.6090	0.3819
0.80	0.3568	0.4639	0.4034
0.90	0.5302	0.2403	0.3307
17. Best F1 Threshold

The highest F1-score among the evaluated thresholds occurred at:

Threshold = 0.80

Results:

Precision = 35.68%
Recall    = 46.39%
F1        = 40.34%

The threshold of 0.80 should not automatically be considered the production threshold.

The production operating point should ultimately be selected using the business cost of:

False positives
False negatives
Manual review capacity
Customer friction
Fraud loss
18. Baseline vs Experiment 1 at Threshold 0.80
Baseline
Precision = 34.36%
Recall    = 47.19%
F1        = 39.77%
Experiment 1
Precision = 35.68%
Recall    = 46.39%
F1        = 40.34%

F1 improved:

39.77% → 40.34%

Precision improved:

34.36% → 35.68%

Recall decreased slightly:

47.19% → 46.39%
19. Experiment Decision
Decision: KEEP

Experiment 1 will be retained for future experiments.

Reason:

PR-AUC improved
ROC-AUC improved
Precision improved
F1 improved
Recall remained approximately stable
False positives decreased at threshold 0.50
No test-set information was used

The improvement is modest but measurable.

20. Leakage Check

The experiment follows the leakage-safe preprocessing policy.

Feature engineering features are deterministic transformations of transaction attributes and do not use isFraud.

The preprocessing transformer is fitted only on training data.

Validation data is transformed using the already-fitted transformer.

The test set was not used for:

Feature selection
Model training
Threshold selection
Performance comparison

Therefore, the experiment remains isolated from the final test evaluation.

21. Saved Experiment Artifacts

The experiment generated:

ml/
├── metrics/
│   ├── experiment_1_metrics.json
│   └── threshold_analysis_experiment_1.csv
│
└── models/
    └── experiment_1/
        ├── xgboost_experiment_1.json
        └── preprocessor.joblib

The model artifacts are excluded from normal Git tracking.

Experiment metrics and documentation are retained for reproducibility and comparison.

22. Limitations

The experiment has several limitations.

22.1 Modest Improvement

The PR-AUC improvement is:

+0.008256

Although positive, it is not a dramatic improvement.

Further experiments are required.

22.2 No Statistical Stability Analysis

The experiment uses a single predefined validation split.

Repeated validation or time-based validation may be required to assess stability.

22.3 No Business Cost Function

Threshold selection currently uses F1-score rather than an explicit financial cost function.

22.4 Limited Feature Engineering

Only transaction amount and relative time features were added.

Identity features, frequency features, behavioral features, and interaction features have not yet been evaluated.

23. Next Experiment

The next experiment should evaluate additional fraud-related signals while maintaining the same experimental methodology.

Potential directions include:

Identity dataset integration
Missingness indicators
Frequency-based features
Card-level behavioral features
Email-domain features
Device-related features
Interaction features

Each experiment should be compared against the current best model.

24. Current Best Model

Current best validation model:

Experiment 1

Primary metric:

PR-AUC = 0.372530

Current F1-optimal threshold from the tested range:

0.80

However, the threshold remains provisional and should not be treated as the final production threshold.

25. Experiment Status

Experiment: Feature Engineering Experiment 1

Status: KEEP

Validation: Complete

PR-AUC: 0.372530

Baseline PR-AUC: 0.364275

Improvement: +0.008256

Test Set: Untouched

Next Stage: Advanced Feature Engineering