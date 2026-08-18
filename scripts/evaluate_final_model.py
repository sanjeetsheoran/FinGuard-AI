"""
FinGuard AI - Final Model Evaluation

Final model:
    Experiment 3

Configuration:
    n_estimators = 500
    max_depth = 6
    learning_rate = 0.08

Final threshold:
    0.80

Purpose:
    Evaluate the final selected model on the untouched TEST set.

Important:
    - TEST set is used only for final evaluation.
    - No model fitting is performed on TEST.
    - No preprocessing fitting is performed on TEST.
    - Threshold was selected previously using VALIDATION only.
    - isFraud is excluded from model features.
    - TransactionID is only a join key.
"""

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PATHS
# ============================================================

TRANSACTION_FILE = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "raw"
    / "train_transaction.csv"
)

IDENTITY_FILE = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "raw"
    / "train_identity.csv"
)

SPLIT_FILE = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "splits"
    / "transaction_split.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "experiment_3"
    / "xgboost_experiment_3.json"
)

PREPROCESSOR_FILE = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "experiment_3"
    / "preprocessor.joblib"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "ml"
    / "metrics"
    / "final_test_metrics.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

ID_COLUMN = "TransactionID"
TARGET = "isFraud"

FINAL_THRESHOLD = 0.80


# ============================================================
# TRANSACTION FEATURES
# ============================================================

TRANSACTION_FEATURES = [
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
]


# ============================================================
# ENGINEERED FEATURES
# ============================================================

ENGINEERED_FEATURES = [
    "TransactionAmtLog",
    "TransactionAmtBucket",
    "TransactionHour",
    "TransactionDay",
]


# ============================================================
# IDENTITY FEATURES
# ============================================================

IDENTITY_NUMERICAL_FEATURES = [
    "id_01",
    "id_02",
    "id_03",
    "id_04",
    "id_05",
    "id_06",
    "id_07",
    "id_08",
    "id_09",
    "id_10",
    "id_11",
    "id_13",
    "id_14",
    "id_15",
    "id_17",
    "id_19",
    "id_20",
    "id_21",
    "id_22",
    "id_24",
    "id_25",
    "id_26",
    "id_32",
]


IDENTITY_CATEGORICAL_FEATURES = [
    "id_12",
    "id_16",
    "id_18",
    "id_23",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    "DeviceType",
    "DeviceInfo",
]


IDENTITY_AVAILABILITY_FEATURES = [
    "HasIdentity",
    "IdentityMissingCount",
    "IdentityMissingRatio",
]


# ============================================================
# UTILITY
# ============================================================

def section(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# TRANSACTION FEATURE ENGINEERING
# ============================================================

def engineer_transaction_features(df):

    result = df.copy()

    result["TransactionAmtLog"] = np.log1p(
        result["TransactionAmt"].clip(
            lower=0
        )
    )

    def amount_bucket(value):

        if pd.isna(value):
            return "unknown"

        if value <= 25:
            return "very_low"

        if value <= 50:
            return "low"

        if value <= 100:
            return "medium"

        if value <= 250:
            return "high"

        if value <= 500:
            return "very_high"

        if value <= 1000:
            return "premium"

        return "extreme"

    result["TransactionAmtBucket"] = (
        result["TransactionAmt"]
        .apply(amount_bucket)
    )

    result["TransactionHour"] = (
        (result["TransactionDT"] // 3600)
        % 24
    )

    result["TransactionDay"] = (
        result["TransactionDT"] // 86400
    )

    return result


# ============================================================
# IDENTITY FEATURE ENGINEERING
# ============================================================

def add_identity_features(
    transaction_df,
    identity_df,
):

    result = transaction_df.merge(
        identity_df,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    if len(result) != len(transaction_df):

        raise ValueError(
            "Identity merge changed transaction row count."
        )

    identity_columns = [
        column
        for column in (
            IDENTITY_NUMERICAL_FEATURES
            + IDENTITY_CATEGORICAL_FEATURES
        )
        if column in result.columns
    ]

    if not identity_columns:

        raise ValueError(
            "No identity columns found."
        )

    result["HasIdentity"] = (
        result[identity_columns]
        .notna()
        .any(axis=1)
        .astype("int8")
    )

    result["IdentityMissingCount"] = (
        result[identity_columns]
        .isna()
        .sum(axis=1)
    )

    result["IdentityMissingRatio"] = (
        result[identity_columns]
        .isna()
        .sum(axis=1)
        / len(identity_columns)
    )

    return result


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    section(
        "LOADING TEST DATA"
    )

    split_df = pd.read_csv(
        SPLIT_FILE,
        usecols=[
            ID_COLUMN,
            "split",
        ],
    )

    transaction_columns = [
        ID_COLUMN,
        TARGET,
        *TRANSACTION_FEATURES,
    ]

    transaction_columns = list(
        dict.fromkeys(
            transaction_columns
        )
    )

    transaction_df = pd.read_csv(
        TRANSACTION_FILE,
        usecols=transaction_columns,
    )

    print(
        f"Transaction rows: "
        f"{len(transaction_df):,}"
    )

    identity_columns = [
        ID_COLUMN,
        *IDENTITY_NUMERICAL_FEATURES,
        *IDENTITY_CATEGORICAL_FEATURES,
    ]

    identity_columns = list(
        dict.fromkeys(
            identity_columns
        )
    )

    identity_df = pd.read_csv(
        IDENTITY_FILE,
        usecols=identity_columns,
    )

    print(
        f"Identity rows: "
        f"{len(identity_df):,}"
    )

    transaction_df = transaction_df.merge(
        split_df,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    test_df = transaction_df[
        transaction_df["split"] == "test"
    ].copy()

    print(
        f"Test transactions: "
        f"{len(test_df):,}"
    )

    return (
        test_df,
        identity_df,
    )


# ============================================================
# PREPARE TEST FEATURES
# ============================================================

def prepare_test_data(
    test_df,
    identity_df,
):

    section(
        "PREPARING TEST FEATURES"
    )

    test_df = engineer_transaction_features(
        test_df
    )

    test_df = add_identity_features(
        test_df,
        identity_df,
    )

    feature_columns = [
        *TRANSACTION_FEATURES,
        *ENGINEERED_FEATURES,
        *IDENTITY_NUMERICAL_FEATURES,
        *IDENTITY_CATEGORICAL_FEATURES,
        *IDENTITY_AVAILABILITY_FEATURES,
    ]

    feature_columns = [
        column
        for column in feature_columns
        if column in test_df.columns
    ]

    forbidden_columns = {
        ID_COLUMN,
        TARGET,
        "split",
    }

    leakage = (
        set(feature_columns)
        & forbidden_columns
    )

    if leakage:

        raise ValueError(
            f"Leakage detected: {leakage}"
        )

    X_test = test_df[
        feature_columns
    ].copy()

    y_test = test_df[
        TARGET
    ].copy()

    print(
        f"Input feature columns: "
        f"{len(feature_columns)}"
    )

    print(
        "Target exclusion: PASSED"
    )

    return (
        X_test,
        y_test,
    )


# ============================================================
# LOAD FINAL MODEL
# ============================================================

def load_final_model():

    section(
        "LOADING FINAL MODEL"
    )

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_FILE}"
        )

    if not PREPROCESSOR_FILE.exists():

        raise FileNotFoundError(
            f"Preprocessor not found:\n"
            f"{PREPROCESSOR_FILE}"
        )

    model = xgb.XGBClassifier()

    model.load_model(
        MODEL_FILE
    )

    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    print(
        "Experiment 3 model loaded successfully."
    )

    print(
        "Experiment 3 preprocessor loaded successfully."
    )

    print(
        f"Final threshold: "
        f"{FINAL_THRESHOLD:.2f}"
    )

    return (
        model,
        preprocessor,
    )


# ============================================================
# FINAL EVALUATION
# ============================================================

def evaluate_final_model(
    model,
    preprocessor,
    X_test,
    y_test,
):

    section(
        "FINAL TEST EVALUATION"
    )

    print(
        "Transforming TEST data..."
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    print(
        f"Processed test shape: "
        f"{X_test_processed.shape}"
    )

    print(
        "\nGenerating TEST predictions..."
    )

    probabilities = (
        model.predict_proba(
            X_test_processed
        )[:, 1]
    )

    predictions = (
        probabilities >= FINAL_THRESHOLD
    ).astype(int)

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    print(
        "\nFINAL TEST RESULTS"
    )

    print(
        f"\nPR-AUC    : "
        f"{pr_auc:.6f}"
    )

    print(
        f"ROC-AUC   : "
        f"{roc_auc:.6f}"
    )

    print(
        f"Precision : "
        f"{precision:.6f}"
    )

    print(
        f"Recall    : "
        f"{recall:.6f}"
    )

    print(
        f"F1-Score  : "
        f"{f1:.6f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Legitimate",
                "Fraud",
            ],
            zero_division=0,
        )
    )

    fraud_predictions = int(
        predictions.sum()
    )

    actual_frauds = int(
        y_test.sum()
    )

    print(
        "\nPrediction Summary:"
    )

    print(
        f"Actual fraud transactions   : "
        f"{actual_frauds:,}"
    )

    print(
        f"Predicted fraud transactions: "
        f"{fraud_predictions:,}"
    )

    return {
        "model": "experiment_3",
        "evaluation_split": "test",
        "threshold": FINAL_THRESHOLD,
        "test_rows": int(len(y_test)),
        "actual_fraud_count": actual_frauds,
        "predicted_fraud_count": fraud_predictions,
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(metrics):

    section(
        "SAVING FINAL TEST METRICS"
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(
        f"Final test metrics saved to:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "FinGuard AI - Final Model Evaluation"
    )

    print(
        "FINAL MODEL: Experiment 3"
    )

    print(
        "FINAL THRESHOLD: 0.80"
    )

    print(
        "TEST SET WILL BE USED FOR FINAL EVALUATION ONLY."
    )

    (
        test_df,
        identity_df,
    ) = load_test_data()

    (
        X_test,
        y_test,
    ) = prepare_test_data(
        test_df,
        identity_df,
    )

    (
        model,
        preprocessor,
    ) = load_final_model()

    metrics = evaluate_final_model(
        model,
        preprocessor,
        X_test,
        y_test,
    )

    save_results(
        metrics
    )

    section(
        "FINAL MODEL EVALUATION COMPLETE"
    )

    print(
        "Experiment 3 evaluated on TEST set."
    )

    print(
        "No model fitting was performed on TEST."
    )

    print(
        "No preprocessing fitting was performed on TEST."
    )

    print(
        "Threshold 0.80 was selected using VALIDATION only."
    )

    print(
        "Final test metrics saved successfully."
    )


if __name__ == "__main__":
    main()