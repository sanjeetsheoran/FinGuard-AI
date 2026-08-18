# FinGuard AI - Experiment 3 Report

## XGBoost Hyperparameter Experiment

---

## 1. Objective

The objective of Experiment 3 was to improve the current best fraud detection model by increasing the number of XGBoost estimators.

Experiment 2 was used as the baseline configuration.

The feature set remained unchanged.

Only the number of boosting estimators was increased:

```text
Experiment 2:
n_estimators = 300

Experiment 3:
n_estimators = 500

2. Experiment Configuration
Base Feature Set

Experiment 3 retained the complete feature set from Experiment 2:

Transaction Features
+
Transaction Feature Engineering
+
Identity Features
+
Identity Availability Features
Transaction Features

Selected transaction features include:

TransactionDT
TransactionAmt
ProductCD
card1
card2
card3
card4
card5
card6
addr1
addr2
dist1
dist2
P_emaildomain
R_emaildomain
M1
M2
M3
M4
M5
M6
M7
M8
M9
3. Engineered Features

The following features from previous experiments were retained:

TransactionAmtLog
TransactionAmtBucket
TransactionHour
TransactionDay

These features capture:

Transaction amount scale
Transaction amount ranges
Time of day
Transaction day
4. Identity Features

Experiment 3 retained the identity feature set introduced in Experiment 2.

Numerical Identity Features
id_01
id_02
id_03
id_04
id_05
id_06
id_07
id_08
id_09
id_10
id_11
id_13
id_14
id_15
id_17
id_19
id_20
id_21
id_22
id_24
id_25
id_26
id_32
Categorical Identity Features
id_12
id_16
id_18
id_23
id_27
id_28
id_29
id_30
id_31
id_33
id_34
id_35
id_36
id_37
id_38
DeviceType
DeviceInfo
5. Identity Availability Features

The following features were also retained:

HasIdentity

Indicates whether identity information is available for a transaction.

1 = identity information available
0 = identity information unavailable
IdentityMissingCount

Number of missing identity fields.

IdentityMissingRatio

Proportion of missing identity fields.

IdentityMissingRatio =
IdentityMissingCount / TotalIdentityFeatures
6. Dataset Split

The same dataset split was used as previous experiments.

Train      : 413,378
Validation : 88,581
Test       : 88,581

The test set was not used during Experiment 3.

7. Preprocessing

The preprocessing pipeline was fitted on training data only.

TRAIN
    |
    | fit_transform()
    v
Processed TRAIN


VALIDATION
    |
    | transform()
    v
Processed VALIDATION

This prevents validation information from influencing the preprocessing parameters.

The test set remained completely untouched.

8. Processed Feature Dimensions

Experiment 3 produced:

Train      : 413,378 × 2,328
Validation : 88,581 × 2,328

The feature dimensions remained identical to Experiment 2 because the feature set was unchanged.

Only the XGBoost hyperparameter n_estimators was changed.

9. Model Configuration

Model:

XGBoost Classifier

Experiment 3 configuration:

n_estimators      = 500
max_depth         = 6
learning_rate     = 0.08
subsample         = 0.8
colsample_bytree  = 0.8
scale_pos_weight  = 27.5798

Compared with Experiment 2:

Experiment 2:
n_estimators = 300


Experiment 3:
n_estimators = 500

All other major model settings remained unchanged.

10. Experiment 3 Validation Results

At the default classification threshold of 0.50:

PR-AUC    : 0.536558
ROC-AUC   : 0.921970
Precision : 0.200209
Recall    : 0.804839
F1-Score  : 0.320653

Confusion matrix:

[[75514  9967]
 [  605  2495]]
11. Experiment 2 vs Experiment 3
Metric	Experiment 2	Experiment 3	Change
PR-AUC	0.486660	0.536558	+0.049898
ROC-AUC	0.908566	0.921970	+0.013404
Precision	0.175046	0.200209	+0.025162
Recall	0.792903	0.804839	+0.011935
F1-Score	0.286781	0.320653	+0.033872
12. Performance Interpretation

Experiment 3 improved every reported validation metric compared with Experiment 2.

The primary metric, PR-AUC, increased from:

0.486660

to:

0.536558

Absolute improvement:

+0.049898

ROC-AUC also increased:

0.908566 → 0.921970

Precision improved:

0.175046 → 0.200209

Recall improved:

0.792903 → 0.804839

F1-Score improved:

0.286781 → 0.320653

Because both precision and recall improved simultaneously, the resulting F1-Score also increased.

13. Experiment Decision
Decision: KEEP

Experiment 3 becomes the current best model configuration.

The improvement in PR-AUC indicates that increasing the number of XGBoost estimators provided additional predictive capability on the validation dataset.

14. Threshold Analysis

Threshold analysis was performed on the Experiment 3 validation predictions.

Threshold	Precision	Recall	F1-Score	Predicted Fraud
0.10	0.0497	0.9868	0.0947	61,489
0.20	0.0743	0.9574	0.1379	39,937
0.30	0.1067	0.9132	0.1910	26,542
0.40	0.1477	0.8642	0.2523	18,134
0.50	0.2002	0.8048	0.3207	12,462
0.60	0.2764	0.7319	0.4012	8,210
0.70	0.3793	0.6352	0.4750	5,191
0.80	0.5308	0.5148	0.5227	3,007
0.90	0.7496	0.3265	0.4548	1,350
15. Best Threshold

The best F1-Score among the evaluated thresholds was obtained at:

Threshold : 0.80
Precision : 0.5308
Recall    : 0.5148
F1-Score  : 0.5227

Therefore:

Best evaluated validation threshold = 0.80

At this threshold, 3,007 validation transactions were classified as fraudulent.

16. Threshold Trade-off

Lower thresholds increase recall but also produce significantly more false positives.

For example:

Threshold = 0.50
Precision = 0.2002
Recall    = 0.8048
F1        = 0.3207

At threshold 0.80:

Threshold = 0.80
Precision = 0.5308
Recall    = 0.5148
F1        = 0.5227

Therefore, threshold 0.80 provides the best F1-Score among the evaluated thresholds.

The final production threshold should ultimately depend on the business cost of false positives versus false negatives.

17. Overall Model Progression

The fraud detection pipeline has progressed through three major experiments.

Baseline
PR-AUC = 0.364275
Experiment 1

Added transaction-level engineered features.

PR-AUC = 0.372530
Experiment 2

Added identity features.

PR-AUC = 0.486660
Experiment 3

Increased XGBoost estimators from 300 to 500.

PR-AUC = 0.536558

Overall:

0.364275
    ↓
0.372530
    ↓
0.486660
    ↓
0.536558

This represents a substantial improvement over the original baseline.

18. Current Best Model

The current best model is:

FinGuard AI - Experiment 3

Feature configuration:

Transaction Features
+
Transaction Engineered Features
+
Identity Features
+
Identity Availability Features

Model:

XGBoost
n_estimators = 500
max_depth = 6
learning_rate = 0.08

Validation performance:

PR-AUC  = 0.536558
ROC-AUC = 0.921970

Best evaluated validation threshold:

0.80

F1-Score at threshold 0.80:

0.5227
19. Saved Artifacts

Model:

ml/models/experiment_3/xgboost_experiment_3.json

Preprocessor:

ml/models/experiment_3/preprocessor.joblib

Metrics:

ml/metrics/experiment_3_metrics.json

Threshold analysis:

ml/metrics/threshold_analysis_experiment_3.csv

Training script:

scripts/train_experiment_3.py

Threshold analysis script:

scripts/analyze_thresholds_experiment_3.py
20. Leakage and Evaluation Policy

The following controls were maintained:

The preprocessing pipeline was fitted on training data only.
Validation data was used only for evaluation and threshold analysis.
The test set was not used.
isFraud was excluded from input features.
TransactionID was used only as a join key.
Identity features were joined using a one-to-one LEFT JOIN.
Threshold selection was performed only on validation predictions.
21. Conclusion

Experiment 3 successfully improved the current fraud detection model.

The primary metric increased from:

Experiment 2 PR-AUC = 0.486660

to:

Experiment 3 PR-AUC = 0.536558

The experiment was therefore retained as the current best configuration.

Threshold analysis identified:

Threshold = 0.80

as the best evaluated threshold by F1-Score, producing:

Precision = 0.5308
Recall    = 0.5148
F1-Score  = 0.5227

The test set remains untouched and will be reserved for final evaluation after model development and experimentation are complete.