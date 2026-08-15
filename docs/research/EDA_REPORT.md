# FinGuard AI - Exploratory Data Analysis Report

## 1. Objective

The objective of this exploratory data analysis (EDA) is to understand the structure, distribution, behavioural patterns, categorical relationships, transaction characteristics, and missing-value patterns present in the IEEE-CIS fraud detection dataset.

The analysis is performed before feature engineering and model development to identify potentially useful signals and data-quality considerations.

---

# 2. Dataset Overview

The transaction dataset contains:

| Property | Value |
|---|---:|
| Transactions | 590,540 |
| Features | 394 |
| Fraudulent Transactions | 20,663 |
| Legitimate Transactions | 569,877 |
| Overall Fraud Rate | 3.499% |

The dataset is highly imbalanced, with fraudulent transactions representing only 3.499% of all transactions.

Therefore, model evaluation should not rely on accuracy alone.

---

# 3. Transaction Amount Analysis

## 3.1 Overall Distribution

The transaction amount statistics are:

| Statistic | Value |
|---|---:|
| Mean | 135.03 |
| Median | 68.77 |
| Standard Deviation | 239.16 |
| Minimum | 0.25 |
| Maximum | 31,937.39 |

The large difference between the mean and median indicates a right-skewed transaction amount distribution.

---

## 3.2 Fraud vs Legitimate Transactions

| Metric | Legitimate | Fraud |
|---|---:|---:|
| Mean | 134.51 | 149.24 |
| Median | 68.50 | 75.00 |

Fraudulent transactions have a somewhat higher average and median transaction amount.

However, the difference is not sufficiently large to treat transaction amount as a standalone fraud detector.

`TransactionAmt` should therefore be considered as one component of a multi-feature fraud detection system.

---

# 4. Product Category Analysis

The observed fraud rate differs significantly across product categories.

| ProductCD | Transactions | Fraud Cases | Fraud Rate |
|---|---:|---:|---:|
| C | 68,519 | 8,008 | 11.687% |
| S | 11,628 | 686 | 5.900% |
| H | 33,024 | 1,574 | 4.766% |
| R | 37,699 | 1,426 | 3.783% |
| W | 439,670 | 8,969 | 2.040% |

Product C has an observed fraud rate of 11.687%, compared with the overall rate of 3.499%.

This makes `ProductCD` a potentially important predictive feature.

However, fraud rate represents association and does not establish causation.

---

# 5. Transaction Time Analysis

`TransactionDT` was analysed as a relative transaction-time feature.

The analysis showed that some relative transaction-time intervals have substantially higher observed fraud rates than the overall baseline.

The highest observed interval in the analysis was relative hour 7, with a fraud rate of 10.610%.

Other elevated intervals included:

- Relative hour 6: 7.774%
- Relative hour 8: 9.301%
- Relative hour 9: 8.996%

The overall fraud rate is 3.499%.

### Important Limitation

`TransactionDT` represents relative elapsed time rather than a verified real-world clock timestamp.

Therefore, the analysis does not claim that a particular real-world hour of the day has the highest fraud activity.

Future feature engineering should transform this variable carefully.

---

# 6. Card Feature Analysis

## 6.1 Card4

Observed fraud rates include:

| Card Type | Fraud Rate |
|---|---:|
| Discover | 7.728% |
| American Express | 5.848% |
| Visa | 3.476% |
| Mastercard | 3.433% |

Card type therefore shows potentially useful variation in fraud behaviour.

---

## 6.2 Card6

| Card Type | Fraud Rate |
|---|---:|
| Credit | 6.678% |
| Debit | 2.426% |

Credit transactions show a higher observed fraud rate than debit transactions.

These results indicate that card-related categorical features should be evaluated during feature engineering.

---

# 7. Email Domain Analysis

Email-domain analysis identified substantial variation in observed fraud rates.

Examples from `P_emaildomain` include:

| Domain | Fraud Rate |
|---|---:|
| mail.com | 18.962% |
| outlook.es | 13.014% |
| aim.com | 12.698% |
| outlook.com | 9.458% |

For `R_emaildomain`, some observed rates were even higher.

For example:

| Domain | Fraud Rate |
|---|---:|
| mail.com | 37.705% |
| outlook.com | 16.514% |
| outlook.es | 13.164% |
| icloud.com | 12.876% |
| gmail.com | 11.918% |

These results indicate that email-domain features may contain predictive information.

However, high observed fraud rates in small groups can be statistically unstable.

Therefore, production feature engineering should use appropriate minimum-support handling and leakage-safe encoding.

---

# 8. M1-M9 Analysis

The M1-M9 features showed different fraud rates depending on their categorical values and missingness.

One notable example is `M4`:

| Value | Fraud Rate |
|---|---:|
| M2 | 11.374% |
| M0 | 3.665% |
| M1 | 2.705% |
| Missing | 1.857% |

Another example is `M6`:

| Value | Fraud Rate |
|---|---:|
| Missing | 7.068% |
| F | 2.369% |
| T | 1.704% |

These results suggest that both categorical values and missingness patterns may contain useful information.

---

# 9. Product × Card Interactions

EDA Part 3 investigated interactions between product categories and card types.

## ProductCD × Card6

The strongest high-support combination observed was:

| ProductCD | Card6 | Transactions | Fraud Rate | Lift |
|---|---|---:|---:|---:|
| C | Credit | 27,551 | 16.925% | 4.837× |
| C | Debit | 40,763 | 8.162% | 2.333× |
| S | Debit | 5,100 | 6.176% | 1.765× |

The `C + Credit` combination has an observed fraud rate of 16.925%, substantially higher than the overall baseline of 3.499%.

This suggests that interaction features may provide additional predictive value beyond individual features.

---

# 10. Product × Card4 Interactions

Examples of higher-support combinations include:

| ProductCD | Card4 | Transactions | Fraud Rate |
|---|---|---:|---:|
| S | Discover | 580 | 13.276% |
| C | Visa | 40,904 | 12.018% |
| C | Mastercard | 27,418 | 11.204% |
| W | Discover | 4,379 | 7.673% |

Some extremely high percentages were observed in groups with very small sample sizes.

For example, one combination had a 100% observed fraud rate with only 2 transactions.

Such groups should not be treated as strong evidence.

Minimum-support filtering and smoothing will therefore be considered during feature engineering.

---

# 11. Product × Email Interactions

Several product-email combinations showed elevated observed fraud rates.

Examples include:

| ProductCD | P_emaildomain | Transactions | Fraud Rate |
|---|---|---:|---:|
| R | outlook.com | 256 | 21.484% |
| C | outlook.com | 2,026 | 17.177% |
| C | gmail.com | 27,654 | 16.923% |
| W | mail.com | 438 | 15.068% |
| C | icloud.com | 438 | 14.384% |

These interactions suggest that categorical combinations may capture behavioural information that individual features do not fully represent.

They should, however, be validated using leakage-safe feature engineering and holdout evaluation.

---

# 12. Transaction Amount Buckets

Transaction amounts were grouped into ranges to investigate non-linear relationships.

| Amount Range | Transactions | Fraud Cases | Fraud Rate | Fraud Share |
|---|---:|---:|---:|---:|
| 0-25 | 43,329 | 3,019 | 6.968% | 14.611% |
| 25-50 | 144,186 | 4,461 | 3.094% | 21.589% |
| 50-100 | 160,742 | 4,618 | 2.873% | 22.349% |
| 100-250 | 175,923 | 5,198 | 2.955% | 25.156% |
| 250-500 | 42,036 | 2,218 | 5.276% | 10.734% |
| 500-1000 | 16,744 | 962 | 5.745% | 4.656% |
| 1000-5000 | 7,562 | 186 | 2.460% | 0.900% |
| 5000+ | 18 | 1 | 5.556% | 0.005% |

The 0-25 range has the highest observed fraud rate at 6.968%.

However, the 100-250 range contributes the largest proportion of total fraud cases at 25.156%.

This demonstrates the distinction between:

- Fraud rate
- Fraud volume
- Fraud contribution

These should not be treated as the same metric.

---

# 13. Fraud Concentration by Product

Fraud cases are distributed unevenly across ProductCD categories.

| ProductCD | Fraud Cases | Fraud Share |
|---|---:|---:|
| W | 8,969 | 43.406% |
| C | 8,008 | 38.755% |
| H | 1,574 | 7.617% |
| R | 1,426 | 6.901% |
| S | 686 | 3.320% |

The cumulative fraud contribution is:

| Products | Cumulative Fraud Share |
|---|---:|
| W | 43.406% |
| W + C | 82.161% |
| W + C + H | 89.779% |
| W + C + H + R | 96.680% |
| All products | 100.000% |

Product W and C together account for 82.161% of all observed fraud cases.

This is a fraud-volume concentration finding and should not be confused with fraud rate.

---

# 14. Missingness Pattern Analysis

Missingness itself was investigated as a potential behavioural signal.

Selected features included:

- D6
- D7
- D8
- D9
- D12
- D13
- D14
- dist1
- dist2

The number of missing values among these selected features was grouped and compared against fraud rates.

| Missing Selected Features | Transactions | Fraud Cases | Fraud Rate | Fraud Share |
|---:|---:|---:|---:|---:|
| 1 | 9,947 | 1,614 | 16.226% | 7.811% |
| 2 | 12,360 | 2,358 | 19.078% | 11.412% |
| 3 | 10,728 | 1,285 | 11.978% | 6.219% |
| 4 | 13,162 | 1,070 | 8.129% | 5.178% |
| 5 | 16,845 | 1,057 | 6.275% | 5.115% |
| 6 | 15,219 | 1,154 | 7.583% | 5.585% |
| 7 | 36,304 | 2,023 | 5.572% | 9.790% |
| 8 | 244,789 | 5,082 | 2.076% | 24.595% |
| 9 | 231,186 | 5,020 | 2.171% | 24.295% |

The results indicate that missingness patterns may contain predictive information.

However, these observations represent associations and must be validated using leakage-safe train/validation/test procedures.

---

# 15. Key EDA Findings

The EDA identified the following potentially useful signal families:

### Transaction Behaviour

- Transaction amount
- Relative transaction time
- Distance-related features

### Product Information

- ProductCD
- Product-specific fraud concentration

### Card Information

- card4
- card6
- Product × card interactions

### Email Information

- P_emaildomain
- R_emaildomain
- Product × email interactions

### Verification Features

- M1-M9
- Missingness indicators
- Missingness patterns

### Identity Information

Identity information is available for only 24.42% of transactions.

Therefore, identity features cannot be treated as mandatory inputs for the complete fraud detection system.

---

# 16. Important Statistical and ML Considerations

The EDA findings are exploratory and do not establish that individual features cause fraud.

Several observed high fraud rates occur in small groups.

Therefore, the project will avoid naive target encoding based on the complete dataset.

The following principles will be applied during feature engineering:

1. Train/validation/test splitting before target-dependent transformations.
2. Target encoding, if used, will be fitted only on training data.
3. Minimum-support thresholds will be applied to categorical groups.
4. Missingness indicators may be retained where supported by validation.
5. Highly skewed numerical features may require transformations.
6. Feature importance will be evaluated using holdout data.
7. Model performance will be evaluated using fraud-focused metrics rather than accuracy alone.

---

# 17. EDA Limitations

The current analysis is exploratory.

It does not yet establish:

- Which features will perform best in a predictive model.
- Whether observed relationships generalize to unseen data.
- Whether categorical interactions improve model performance.
- Whether missingness patterns remain predictive after preprocessing.
- Whether identity features improve performance sufficiently to justify their conditional availability.

These questions will be addressed during feature engineering and model evaluation.

---

# 18. Conclusion

The exploratory analysis demonstrates that the IEEE-CIS fraud dataset contains several potentially useful fraud-related signals.

Important observations include:

- Strong class imbalance with a 3.499% fraud rate.
- Significant differences in fraud rates across ProductCD categories.
- Meaningful interactions between product and card features.
- Potential information in email-domain features.
- Non-linear relationships between transaction amount and fraud.
- Significant fraud concentration across product categories.
- Potential predictive information in missingness patterns.
- Limited availability of identity information.

The findings provide the foundation for the next stage of FinGuard AI:

**Leakage-safe feature engineering and preprocessing.**