# FinGuard AI - Data Validation Report

## 1. Dataset Overview

FinGuard AI uses the IEEE-CIS Fraud Detection dataset as its primary dataset for developing and evaluating the fraud detection pipeline.

The dataset contains transaction-level information and a separate identity dataset. Both datasets are linked using the `TransactionID` field.

The raw datasets were validated programmatically before any preprocessing or model development.

## 2. Transaction Dataset Statistics

| Property | Result |
|---|---:|
| Rows | 590,540 |
| Columns | 394 |
| File Size | 651.69 MB |
| Duplicate Rows | 0 |
| Duplicate TransactionIDs | 0 |

The transaction dataset contains the target variable `isFraud` along with transaction, card, address, distance, and other anonymized features.

## 3. Identity Dataset Statistics

| Property | Result |
|---|---:|
| Rows | 144,233 |
| Columns | 41 |
| File Size | 25.30 MB |
| Duplicate Rows | 0 |

The identity dataset contains additional identity-related features associated with a subset of transactions.

## 4. Target Distribution

The target variable is `isFraud`.

| Class | Transactions | Percentage |
|---|---:|---:|
| Legitimate | 569,877 | 96.501% |
| Fraud | 20,663 | 3.499% |

The dataset is highly imbalanced toward legitimate transactions.

Therefore, accuracy will not be used as the sole evaluation metric during model development.

Primary evaluation metrics will include:

- Precision
- Recall
- F1-Score
- PR-AUC
- ROC-AUC
- False Positive Rate
- False Negative Rate

## 5. Missing Value Analysis

The transaction dataset contains missing values in 374 of its 394 columns.

Some of the columns with the highest missing-value percentages include:

| Feature | Missing Percentage |
|---|---:|
| dist2 | 93.63% |
| D7 | 93.41% |
| D13 | 89.51% |
| D14 | 89.47% |
| D12 | 89.04% |
| D6 | 87.61% |
| D8 | 87.31% |
| D9 | 87.31% |

The identity dataset contains missing values in 38 of its 41 columns.

Examples include:

| Feature | Missing Percentage |
|---|---:|
| id_24 | 96.71% |
| id_25 | 96.44% |
| id_07 | 96.43% |
| id_08 | 96.43% |
| id_21 | 96.42% |
| id_26 | 96.42% |
| id_22 | 96.42% |
| id_27 | 96.42% |
| id_23 | 96.42% |

Missing values will not be removed blindly. Feature-level analysis will be performed before deciding whether a feature should be retained, imputed, transformed, or removed.

## 6. Duplicate Analysis

The validation process identified:

- 0 duplicate transaction rows.
- 0 duplicate identity rows.
- 0 duplicate `TransactionID` values in the transaction dataset.

This indicates that no exact duplicate rows were detected in the validated raw datasets.

## 7. TransactionID Integrity

`TransactionID` was checked for duplication in the transaction dataset.

Result:

```text
Duplicate TransactionIDs: 0