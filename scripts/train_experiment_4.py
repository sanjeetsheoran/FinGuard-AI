"""
FinGuard AI - Experiment 4

XGBoost Regularization Experiment

Experiment 4:
    Same feature set as Experiment 3
    Same n_estimators
    Same learning rate
    Reduced max_depth from 6 to 4

Purpose:
    Test whether shallower trees improve validation
    performance and reduce model complexity.

Important:
    - Same train/validation split.
    - Test set is NOT used.
    - Preprocessor is fitted on TRAIN only.
    - isFraud is excluded from features.
    - TransactionID is only a join key.
"""

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from xgboost import XGBClassifier


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

EXP3_METRICS_FILE = (
    PROJECT_ROOT
    / "ml"
    / "metrics"
    / "experiment_3_metrics.json"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "experiment_4"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "ml"
    / "metrics"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

ID_COLUMN = "TransactionID"
TARGET = "isFraud"

RANDOM_STATE = 42

N_ESTIMATORS = 500
MAX_DEPTH = 4
LEARNING_RATE = 0.08

SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8

THRESHOLD = 0.50


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
            "Identity merge changed row count."
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
# LOAD DATA
# ============================================================

def load_data():

    section("LOADING DATA")

    split_df = pd.read_csv(
        SPLIT_FILE,
        usecols=[
            ID_COLUMN,
            "split",
        ],
    )

    print(
        f"Split records: "
        f"{len(split_df):,}"
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

    transaction_df = engineer_transaction_features(
        transaction_df
    )

    transaction_df = add_identity_features(
        transaction_df,
        identity_df,
    )

    return transaction_df


# ============================================================
# PREPARE SPLITS
# ============================================================

def prepare_splits(df):

    section(
        "PREPARING TRAIN / VALIDATION SPLITS"
    )

    train_df = df[
        df["split"] == "train"
    ].copy()

    validation_df = df[
        df["split"] == "validation"
    ].copy()

    print(
        f"Train rows      : "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows : "
        f"{len(validation_df):,}"
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
        if column in df.columns
    ]

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

    X_train = train_df[
        feature_columns
    ].copy()

    X_validation = validation_df[
        feature_columns
    ].copy()

    y_train = train_df[
        TARGET
    ].copy()

    y_validation = validation_df[
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
        X_train,
        X_validation,
        y_train,
        y_validation,
    )


# ============================================================
# CLASS WEIGHT
# ============================================================

def calculate_scale_pos_weight(y):

    negative = int(
        (y == 0).sum()
    )

    positive = int(
        (y == 1).sum()
    )

    return negative / positive


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train_processed,
    y_train,
):

    section(
        "TRAINING EXPERIMENT 4"
    )

    scale_pos_weight = (
        calculate_scale_pos_weight(y_train)
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    print(
        f"n_estimators: "
        f"{N_ESTIMATORS}"
    )

    print(
        f"max_depth: "
        f"{MAX_DEPTH}"
    )

    print(
        f"learning_rate: "
        f"{LEARNING_RATE}"
    )

    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
    )

    model.fit(
        X_train_processed,
        y_train,
    )

    print(
        "\nExperiment 4 model training completed."
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_validation_processed,
    y_validation,
):

    section(
        "EXPERIMENT 4 VALIDATION"
    )

    probabilities = model.predict_proba(
        X_validation_processed
    )[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    pr_auc = average_precision_score(
        y_validation,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_validation,
        probabilities,
    )

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

    cm = confusion_matrix(
        y_validation,
        predictions,
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
            y_validation,
            predictions,
            target_names=[
                "Legitimate",
                "Fraud",
            ],
            zero_division=0,
        )
    )

    return {
        "experiment": "experiment_4",
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE_BYTREE,
        "threshold": THRESHOLD,
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "train_rows": 413378,
        "validation_rows": len(y_validation),
    }


# ============================================================
# COMPARE WITH EXPERIMENT 3
# ============================================================

def compare_with_experiment_3(
    experiment_4_metrics,
):

    section(
        "EXPERIMENT 3 VS EXPERIMENT 4"
    )

    if not EXP3_METRICS_FILE.exists():

        print(
            "Experiment 3 metrics file not found."
        )

        return

    with open(
        EXP3_METRICS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        experiment_3 = json.load(file)

    metrics = [
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1_score",
    ]

    print(
        f"{'Metric':<15}"
        f"{'Experiment 3':>18}"
        f"{'Experiment 4':>18}"
        f"{'Change':>15}"
    )

    print("-" * 66)

    for metric in metrics:

        exp3_value = float(
            experiment_3[metric]
        )

        exp4_value = float(
            experiment_4_metrics[metric]
        )

        change = (
            exp4_value
            - exp3_value
        )

        print(
            f"{metric:<15}"
            f"{exp3_value:>18.6f}"
            f"{exp4_value:>18.6f}"
            f"{change:>+15.6f}"
        )

    exp3_pr_auc = float(
        experiment_3["pr_auc"]
    )

    exp4_pr_auc = float(
        experiment_4_metrics["pr_auc"]
    )

    print()

    if exp4_pr_auc > exp3_pr_auc:

        print(
            "Experiment 4 improved PR-AUC."
        )

        print(
            "Decision: KEEP"
        )

    elif exp4_pr_auc < exp3_pr_auc:

        print(
            "Experiment 4 reduced PR-AUC."
        )

        print(
            "Decision: REJECT"
        )

    else:

        print(
            "Experiment 4 produced the same PR-AUC."
        )

        print(
            "Decision: REVIEW"
        )


# ============================================================
# SAVE ARTIFACTS
# ============================================================

def save_artifacts(
    model,
    preprocessor,
    metrics,
):

    section(
        "SAVING EXPERIMENT 4 ARTIFACTS"
    )

    model_file = (
        MODEL_DIR
        / "xgboost_experiment_4.json"
    )

    preprocessor_file = (
        MODEL_DIR
        / "preprocessor.joblib"
    )

    metrics_file = (
        METRICS_DIR
        / "experiment_4_metrics.json"
    )

    model.save_model(
        model_file
    )

    joblib.dump(
        preprocessor,
        preprocessor_file,
    )

    with open(
        metrics_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(
        f"Model saved:\n"
        f"{model_file}"
    )

    print(
        f"Preprocessor saved:\n"
        f"{preprocessor_file}"
    )

    print(
        f"Metrics saved:\n"
        f"{metrics_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "FinGuard AI - XGBoost Experiment 4"
    )

    df = load_data()

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = prepare_splits(df)

    section(
        "BUILDING PREPROCESSOR"
    )

    from ml.preprocessing.preprocessor import (
        build_preprocessor,
    )

    preprocessor = build_preprocessor(
        X_train
    )

    print(
        "Fitting preprocessing on TRAIN only..."
    )

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    print(
        f"Processed train shape: "
        f"{X_train_processed.shape}"
    )

    print(
        "Transforming validation data..."
    )

    X_validation_processed = (
        preprocessor.transform(
            X_validation
        )
    )

    print(
        f"Processed validation shape: "
        f"{X_validation_processed.shape}"
    )

    model = train_model(
        X_train_processed,
        y_train,
    )

    metrics = evaluate_model(
        model,
        X_validation_processed,
        y_validation,
    )

    compare_with_experiment_3(
        metrics
    )

    save_artifacts(
        model,
        preprocessor,
        metrics,
    )

    section(
        "EXPERIMENT 4 COMPLETE"
    )

    print(
        "Experiment 3 feature set retained."
    )

    print(
        "n_estimators remains 500."
    )

    print(
        "max_depth reduced from 6 to 4."
    )

    print(
        "Preprocessor fitted on TRAIN only."
    )

    print(
        "Validation used only for evaluation."
    )

    print(
        "Test set was NOT used."
    )


if __name__ == "__main__":
    main()