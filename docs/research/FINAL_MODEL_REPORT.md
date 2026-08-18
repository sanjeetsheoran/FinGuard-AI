FinGuard AI - Final Model Report
1. Project Overview
FinGuard AI is a machine learning–based fraud detection system designed to identify potentially fraudulent financial transactions.

The project emphasizes a leakage-safe pipeline, integrating transaction-level features, engineered features, and identity information.

The final model was selected through controlled experiments using a dedicated validation set, with the test set kept untouched until final evaluation.

2. Dataset
Transaction Dataset
Total transactions: 590,540

Columns: 394

Identity Dataset
Total identity records: 144,233

Columns: 41

Integration: LEFT JOIN on TransactionID

Duplicate IDs: 0 (both transaction and identity)

Identity coverage: 24.42% (144,233 transactions)

Transactions without identity info: 446,307

No identity records existed without a corresponding transaction.

3. Dataset Split
Train: 413,378 (70%)

Validation: 88,581 (15%)

Test: 88,581 (15%)

Fraud distribution:

Train: 14,464 fraud

Validation: 3,100 fraud

Test: 3,099 fraud

Split integrity: No TransactionID overlap.

4. Preprocessing
Numerical Features: Median imputation, missing-value indicators.

Categorical Features: Missing → “Unknown”, one-hot encoding, unknown categories ignored.

Preprocessing fitted only on training data.

Validation and test sets transformed using training-fitted preprocessor.

5. Feature Engineering
Transaction Features
TransactionAmtLog → log1p transformation

TransactionAmtBucket → categorical ranges (very_low → extreme)

TransactionHour → extracted from TransactionDT

TransactionDay → extracted from TransactionDT

Identity Features
Numerical: id_01 … id_32

Categorical: id_12, id_16, id_18, … DeviceType, DeviceInfo

Availability: HasIdentity, IdentityMissingCount, IdentityMissingRatio

Final Feature Dimensions
71 input columns

After preprocessing:

Train: 413,378 × 2,328

Validation: 88,581 × 2,328

Test: 88,581 × 2,328

6. Model
Algorithm: XGBoost Classifier

Final Configuration:

n_estimators = 500

max_depth = 6

learning_rate = 0.08

subsample = 0.8

colsample_bytree = 0.8

scale_pos_weight = 27.5798

Primary Metric: PR-AUC (due to class imbalance).

7. Experiment History
Experiment	Change	PR-AUC	Decision
Baseline	Transaction features only	0.364275	Reference
Exp 1	Added engineered transaction features	0.372530	KEEP
Exp 2	Added identity features	0.486660	KEEP
Exp 3	Increased estimators to 500	0.536558	KEEP
Exp 4	Shallower trees (max_depth=4)	0.441736	REJECT
Exp 5	More estimators + lower LR	0.530197	REJECT


Champion Model: Experiment 3

8. Threshold Selection
Best threshold: 0.80

Validation performance at 0.80:

Precision = 0.5308

Recall = 0.5148

F1 = 0.5227

9. Final Test Evaluation
Dataset: 88,581 transactions

Actual fraud: 3,099

Predicted fraud: 2,933

Metrics
PR-AUC = 0.541850

ROC-AUC = 0.923445

Precision = 0.532561

Recall = 0.504034

F1 = 0.517905

Confusion Matrix
Code
[[84111  1371]
 [ 1537  1562]]
True Negatives = 84,111

False Positives = 1,371

False Negatives = 1,537

True Positives = 1,562

Fraud recall: 50.40%  
Fraud precision: 53.26%

10. Validation vs Test
Metric	Validation	Test
PR-AUC	0.536558	0.541850
ROC-AUC	0.921970	0.923445
Precision	0.5308	0.532561
Recall	0.5148	0.504034
F1	0.5227	0.517905


Performance remained stable between validation and test sets.

11. Leakage Prevention
isFraud excluded from inputs.

TransactionID used only as join key.

Identity merged via one-to-one LEFT JOIN.

Preprocessing fitted only on training data.

Threshold selection done on validation only.

Test set untouched until final evaluation.

12. Final Model Artifacts
Model: ml/models/experiment_3/xgboost_experiment_3.json

Preprocessor: ml/models/experiment_3/preprocessor.joblib

Metrics: ml/metrics/final_test_metrics.json

Threshold analysis: ml/metrics/threshold_analysis_experiment_3.csv

Evaluation script: scripts/evaluate_final_model.py

13. Limitations
Fraud Recall: Only 50.40% at threshold 0.80.

False Positives: 1,371 legitimate transactions flagged.

Threshold Trade-off: Business scenarios may require different thresholds.

Dataset Dependence: Performance may vary if transaction patterns change in production.

14. Final Conclusion
FinGuard AI successfully developed a leakage-safe fraud detection pipeline using XGBoost.

The model evolved through:
Baseline → Transaction Feature Engineering → Identity Integration → Hyperparameter Optimization.

Final Model:

XGBoost, n_estimators = 500, max_depth = 6, learning_rate = 0.08

Threshold = 0.80

Final Test Performance:

PR-AUC = 0.541850

ROC-AUC = 0.923445

Precision = 0.532561

Recall = 0.504034

F1 = 0.517905

The test set was used only once, after full model development and threshold selection.