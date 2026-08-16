# FinGuard AI - Experiment 2 Report

## Identity Feature Experiment

---

## 1. Objective

The objective of Experiment 2 was to determine whether identity-related information improves the fraud detection performance of the current best model.

Experiment 2 extends Experiment 1 by adding:

- Identity numerical features
- Identity categorical features
- HasIdentity
- IdentityMissingCount
- IdentityMissingRatio

The experiment uses the same train/validation split as previous experiments.

The test set was not used.

---

## 2. Experiment Configuration

### Dataset

Transaction dataset:

- Rows: 590,540
- Columns: 394

Identity dataset:

- Rows: 144,233
- Columns: 41

Identity integration:

- Join key: `TransactionID`
- Join strategy: `LEFT JOIN`
- Identity coverage: approximately 24.42%

---

## 3. Feature Groups

### Transaction Features

The model uses selected transaction-level features including:

- TransactionDT
- TransactionAmt
- ProductCD
- card features
- address features
- distance features
- email-domain features
- M1-M9 features

---

### Experiment 1 Engineered Features

The following features from Experiment 1 were retained:

- TransactionAmtLog
- TransactionAmtBucket
- TransactionHour
- TransactionDay

---

### Identity Features

Selected numerical identity features:

- id_01
- id_02
- id_03
- id_04
- id_05
- id_06
- id_07
- id_08
- id_09
- id_10
- id_11
- id_13
- id_14
- id_15
- id_17
- id_19
- id_20
- id_21
- id_22
- id_24
- id_25
- id_26
- id_32

Selected categorical identity features:

- id_12
- id_16
- id_18
- id_23
- id_27
- id_28
- id_29
- id_30
- id_31
- id_33
- id_34
- id_35
- id_36
- id_37
- id_38
- DeviceType
- DeviceInfo

---

## 4. Identity Availability Features

Three additional features were created.

### HasIdentity

Indicates whether at least one identity feature is available.

```text
1 = identity information available
0 = identity information unavailable

IdentityMissingCount

Number of missing identity fields for a transaction.

IdentityMissingRatio

Ratio of missing identity fields:

IdentityMissingRatio =
IdentityMissingCount / TotalIdentityFeatures

These features are calculated only from identity columns.

TransactionID and isFraud are not used to calculate identity availability.

5. Data Leakage Prevention

The experiment follows a leakage-safe workflow.

Training
TRAIN
  ↓
fit_transform()
  ↓
XGBoost
Validation
VALIDATION
  ↓
transform()
  ↓
prediction

The preprocessing pipeline is fitted using training data only.

The target variable isFraud is excluded from model features.

TransactionID is used only as the join key.

The test set was not used.

6. Processed Feature Dimensions

Experiment 2 produced:

Train      : 413,378 × 2,328
Validation : 88,581 × 2,328

Experiment 1 produced:

Train      : 413,378 × 191
Validation : 88,581 × 191

The increase in feature dimensions is primarily due to identity categorical variables being one-hot encoded.

7. Model Configuration

Model:

XGBoost Classifier

Parameters:

n_estimators      = 300
max_depth         = 6
learning_rate     = 0.08
scale_pos_weight  = 27.5798

The scale_pos_weight parameter was used to account for the class imbalance in the fraud dataset.

8. Experiment 2 Validation Results

At the default classification threshold of 0.50:

PR-AUC    : 0.486660
ROC-AUC   : 0.908566
Precision : 0.175046
Recall    : 0.792903
F1-Score  : 0.286781

Confusion matrix:

[[73897 11584]
 [  642  2458]]
9. Experiment 1 vs Experiment 2
Metric	Experiment 1	Experiment 2	Change
PR-AUC	0.372530	0.486660	+0.114129
ROC-AUC	0.905054	0.908566	+0.003512
Precision	0.165656	0.175046	+0.009390
Recall	0.802258	0.792903	-0.009355
F1-Score	0.274609	0.286781	+0.012172
Interpretation

Experiment 2 produced a substantial improvement in PR-AUC.

The PR-AUC increased from:

0.372530 → 0.486660

This represents an absolute improvement of:

+0.114129

ROC-AUC also improved slightly.

Precision and F1-Score improved, while recall decreased slightly.

Because PR-AUC is the primary metric for evaluating performance on this highly imbalanced fraud detection problem, the improvement is considered meaningful.

10. Experiment Decision
Decision: KEEP

Experiment 2 is retained as the current best model configuration.

Reason:

Experiment 2 PR-AUC
>
Experiment 1 PR-AUC

Therefore, identity information provides useful additional predictive signal.

11. Threshold Analysis

Threshold analysis was performed using the validation set.

Threshold	Precision	Recall	F1-Score	Predicted Fraud
0.10	0.0451	0.9910	0.0863	68,109
0.20	0.0651	0.9587	0.1220	45,639
0.30	0.0918	0.9158	0.1669	30,917
0.40	0.1283	0.8600	0.2233	20,782
0.50	0.1750	0.7929	0.2868	14,042
0.60	0.2412	0.7029	0.3591	9,035
0.70	0.3424	0.6052	0.4373	5,479
0.80	0.4867	0.4839	0.4853	3,082
0.90	0.7154	0.2903	0.4130	1,258
12. Best F1 Threshold

The best F1-Score among the evaluated thresholds was obtained at:

Threshold : 0.80
Precision : 0.4867
Recall    : 0.4839
F1-Score  : 0.4853

Therefore:

Best validation F1 threshold = 0.80

At this threshold, the model predicts 3,082 validation transactions as fraudulent.

13. Threshold Trade-off

The threshold controls the trade-off between precision and recall.

At lower thresholds:

Recall ↑
Precision ↓

At higher thresholds:

Precision ↑
Recall ↓

For example:

Threshold 0.50
Precision = 0.1750
Recall    = 0.7929

while:

Threshold 0.80
Precision = 0.4867
Recall    = 0.4839

Therefore, threshold 0.80 provides a substantially better balance according to F1-Score on the validation set.

14. Current Best Model

The current best model is:

FinGuard AI Experiment 2

Configuration:

XGBoost
+
Transaction Features
+
Experiment 1 Engineered Features
+
Identity Features
+
Identity Availability Features

Validation PR-AUC:

0.486660

Best evaluated F1 threshold:

0.80

Best evaluated F1:

0.4853
15. Artifacts

Experiment 2 model:

ml/models/experiment_2/xgboost_experiment_2.json

Preprocessor:

ml/models/experiment_2/preprocessor.joblib

Metrics:

ml/metrics/experiment_2_metrics.json

Threshold analysis:

ml/metrics/threshold_analysis_experiment_2.csv

Training script:

scripts/train_identity_experiment_2.py

Threshold analysis script:

scripts/analyze_thresholds_experiment_2.py
16. Validation Policy

The following policy was maintained throughout the experiment:

Training data was used to fit the preprocessing pipeline.
Validation data was used for model evaluation.
Threshold selection was performed using validation data.
The test set was not used.
isFraud was excluded from model features.
TransactionID was used only as a join key.
Identity data was integrated using a one-to-one LEFT JOIN.
Identity availability features were calculated only from identity features.
17. Conclusion

Experiment 2 demonstrates that identity-related information provides substantial additional predictive signal for fraud detection.

The primary metric improved from:

Experiment 1 PR-AUC = 0.372530

to:

Experiment 2 PR-AUC = 0.486660

Therefore, Experiment 2 is retained as the current best model.

The validation threshold analysis identifies:

Threshold = 0.80

as the best evaluated threshold by F1-Score.

The test set remains untouched and should only be used for final model evaluation after model development and experimentation are complete.