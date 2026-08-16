"""
FinGuard AI - Experiment 2 Threshold Analysis

Purpose:
    Analyze classification thresholds for the Identity Feature
    Experiment 2 model.

Experiment 2:
    Transaction features
    +
    Experiment 1 engineered features
    +
    Identity features
    +
    Identity availability features

Important:
    - Uses validation set only.
    - Test set is NOT used.
    - Uses the exact preprocessor saved during Experiment 2.
    - Threshold selection is performed only on validation data.
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

import xgboost as xgb


# ============================================================
# Project Setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Paths
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
    / "experiment_2"
    / "xgboost_experiment_2.json"
)

PREPROCESSOR_FILE = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "experiment_2"
    / "preprocessor.joblib"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "ml"
    / "metrics"
    / "threshold_analysis_experiment_2.csv"
)


# ============================================================
# Configuration
# ============================================================

ID_COLUMN = "TransactionID"
TARGET = "isFraud"


# ============================================================
# Transaction Features
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
# Experiment 1 Features
# ============================================================

ENGINEERED_FEATURES = [
    "TransactionAmtLog",
    "TransactionAmtBucket",
    "TransactionHour",
    "TransactionDay",
]


# ============================================================
# Identity Features
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
# Thresholds
# ============================================================

THRESHOLDS = np.arange(
    0.10,
    1.00,
    0.10,
)


# ============================================================
# Utility
# ============================================================

def section(title: str) -> None:

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Feature Engineering
# ============================================================

def engineer_transaction_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    # --------------------------------------------------------
    # Transaction amount log
    # --------------------------------------------------------

    result["TransactionAmtLog"] = np.log1p(
        result["TransactionAmt"].clip(
            lower=0
        )
    )

    # --------------------------------------------------------
    # Transaction amount bucket
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Transaction hour
    # --------------------------------------------------------

    result["TransactionHour"] = (
        (result["TransactionDT"] // 3600)
        % 24
    )

    # --------------------------------------------------------
    # Transaction day
    # --------------------------------------------------------

    result["TransactionDay"] = (
        result["TransactionDT"] // 86400
    )

    return result


# ============================================================
# Identity Feature Engineering
# ============================================================

def add_identity_features(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame,
) -> pd.DataFrame:

    result = transaction_df.merge(
        identity_df,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    if len(result) != len(
        transaction_df
    ):

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
            "No identity columns found after merge."
        )

    # --------------------------------------------------------
    # Identity availability
    # --------------------------------------------------------

    result["HasIdentity"] = (
        result[
            identity_columns
        ]
        .notna()
        .any(axis=1)
        .astype("int8")
    )

    result["IdentityMissingCount"] = (
        result[
            identity_columns
        ]
        .isna()
        .sum(axis=1)
    )

    result["IdentityMissingRatio"] = (
        result[
            identity_columns
        ]
        .isna()
        .sum(axis=1)
        / len(identity_columns)
    )

    return result


# ============================================================
# Load Validation Data
# ============================================================

def load_validation_data():

    section(
        "LOADING VALIDATION DATA"
    )

    if not TRANSACTION_FILE.exists():
        raise FileNotFoundError(
            f"Transaction dataset not found:\n"
            f"{TRANSACTION_FILE}"
        )

    if not IDENTITY_FILE.exists():
        raise FileNotFoundError(
            f"Identity dataset not found:\n"
            f"{IDENTITY_FILE}"
        )

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Split manifest not found:\n"
            f"{SPLIT_FILE}"
        )

    # --------------------------------------------------------
    # Load split manifest
    # --------------------------------------------------------

    split_df = pd.read_csv(
        SPLIT_FILE,
        usecols=[
            ID_COLUMN,
            "split",
        ],
    )

    # --------------------------------------------------------
    # Load transaction data
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Load identity data
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Add split
    # --------------------------------------------------------

    transaction_df = transaction_df.merge(
        split_df,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    transaction_df = engineer_transaction_features(
        transaction_df
    )

    transaction_df = add_identity_features(
        transaction_df,
        identity_df,
    )

    # --------------------------------------------------------
    # Select validation
    # --------------------------------------------------------

    validation_df = transaction_df[
        transaction_df["split"] == "validation"
    ].copy()

    print(
        f"Validation transactions: "
        f"{len(validation_df):,}"
    )

    return validation_df


# ============================================================
# Prepare Validation Features
# ============================================================

def prepare_validation_features(
    validation_df: pd.DataFrame,
):

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
        if column in validation_df.columns
    ]

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    forbidden = {
        ID_COLUMN,
        TARGET,
        "split",
    }

    leakage = (
        set(feature_columns)
        & forbidden
    )

    if leakage:

        raise ValueError(
            f"Leakage detected: {leakage}"
        )

    X_validation = validation_df[
        feature_columns
    ].copy()

    y_validation = validation_df[
        TARGET
    ].copy()

    return (
        X_validation,
        y_validation,
    )


# ============================================================
# Load Model
# ============================================================

def load_model():

    section(
        "LOADING EXPERIMENT 2 MODEL"
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
        "Experiment 2 model loaded successfully."
    )

    print(
        "Experiment 2 preprocessor loaded successfully."
    )

    return (
        model,
        preprocessor,
    )


# ============================================================
# Generate Predictions
# ============================================================

def generate_predictions(
    model,
    preprocessor,
    X_validation,
):

    section(
        "GENERATING EXPERIMENT 2 PREDICTIONS"
    )

    print(
        "Transforming validation data..."
    )

    X_processed = (
        preprocessor.transform(
            X_validation
        )
    )

    print(
        f"Processed validation shape: "
        f"{X_processed.shape}"
    )

    print(
        "Generating fraud probabilities..."
    )

    probabilities = (
        model.predict_proba(
            X_processed
        )[:, 1]
    )

    print(
        "Prediction generation completed."
    )

    return probabilities


# ============================================================
# Threshold Analysis
# ============================================================

def analyze_thresholds(
    y_validation,
    probabilities,
):

    section(
        "EXPERIMENT 2 THRESHOLD ANALYSIS"
    )

    pr_auc = average_precision_score(
        y_validation,
        probabilities,
    )

    results = []

    print(
        "\nThreshold Performance:\n"
    )

    print(
        f"{'threshold':>9} "
        f"{'precision':>9} "
        f"{'recall':>8} "
        f"{'f1_score':>9} "
        f"{'predicted_fraud_count':>23}"
    )

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        predicted_fraud_count = int(
            predictions.sum()
        )

        results.append(
            {
                "threshold": round(
                    float(threshold),
                    2,
                ),
                "precision": float(
                    precision
                ),
                "recall": float(
                    recall
                ),
                "f1_score": float(
                    f1
                ),
                "predicted_fraud_count": (
                    predicted_fraud_count
                ),
            }
        )

        print(
            f"{threshold:>9.2f} "
            f"{precision:>9.4f} "
            f"{recall:>8.4f} "
            f"{f1:>9.4f} "
            f"{predicted_fraud_count:>23,}"
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Best F1
    # --------------------------------------------------------

    best_row = results_df.loc[
        results_df[
            "f1_score"
        ].idxmax()
    ]

    print(
        "\nBest F1 threshold:"
    )

    print(
        f"Threshold : "
        f"{best_row['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best_row['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_row['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{best_row['f1_score']:.4f}"
    )

    print(
        f"\nExperiment 2 Validation PR-AUC: "
        f"{pr_auc:.6f}"
    )

    return (
        results_df,
        pr_auc,
        best_row,
    )


# ============================================================
# Save Results
# ============================================================

def save_results(
    results_df: pd.DataFrame,
):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\nThreshold analysis saved to:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# Main
# ============================================================

def main():

    section(
        "FinGuard AI - Experiment 2 Threshold Analysis"
    )

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    validation_df = (
        load_validation_data()
    )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    (
        X_validation,
        y_validation,
    ) = prepare_validation_features(
        validation_df
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    (
        model,
        preprocessor,
    ) = load_model()

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    probabilities = generate_predictions(
        model,
        preprocessor,
        X_validation,
    )

    # --------------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------------

    (
        results_df,
        pr_auc,
        best_row,
    ) = analyze_thresholds(
        y_validation,
        probabilities,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results_df
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    section(
        "EXPERIMENT 2 THRESHOLD ANALYSIS COMPLETE"
    )

    print(
        f"Best threshold: "
        f"{best_row['threshold']:.2f}"
    )

    print(
        f"Best F1: "
        f"{best_row['f1_score']:.4f}"
    )

    print(
        f"Validation PR-AUC: "
        f"{pr_auc:.6f}"
    )

    print(
        "Test set was NOT used."
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()