"""
FinGuard AI - Feature Experiment 1

Experiment:
    Baseline features
    +
    Transaction amount features
    +
    Relative time features

Purpose:
    Determine whether the engineered features improve the
    validation performance compared with the baseline model.

Important:
    - Same train/validation split as baseline.
    - Same XGBoost configuration as baseline.
    - Preprocessor is fitted on TRAIN only.
    - Validation is transformed only.
    - Test set remains completely untouched.
"""

from pathlib import Path
import json
import sys

import joblib
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
# Project Setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ml.preprocessing.preprocessor import (
    build_preprocessor,
    prepare_features,
    prepare_target,
)

from scripts.feature_engineering import (
    engineer_features,
)


# ============================================================
# Paths
# ============================================================

RAW_DATASET = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "raw"
    / "train_transaction.csv"
)

SPLIT_FILE = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "splits"
    / "transaction_split.csv"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "ml"
    / "metrics"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "experiment_1"
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 300
MAX_DEPTH = 6
LEARNING_RATE = 0.08
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8

THRESHOLD = 0.50

TARGET = "isFraud"
ID_COLUMN = "TransactionID"


# ============================================================
# Core Features
# ============================================================

CORE_FEATURES = [
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
# Engineered Features
# ============================================================

ENGINEERED_FEATURES = [
    "TransactionAmtLog",
    "TransactionAmtBucket",
    "TransactionHour",
    "TransactionDay",
]


# ============================================================
# Utility
# ============================================================

def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Load Dataset
# ============================================================

def load_data() -> pd.DataFrame:

    section("LOADING DATA")

    if not RAW_DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{RAW_DATASET}"
        )

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Split manifest not found:\n{SPLIT_FILE}"
        )

    split_df = pd.read_csv(
        SPLIT_FILE,
        usecols=[
            ID_COLUMN,
            "split",
        ],
    )

    columns = [
        ID_COLUMN,
        TARGET,
        *CORE_FEATURES,
    ]

    # Remove duplicate column names while
    # preserving their original order.
    columns = list(
        dict.fromkeys(columns)
    )

    print(
        "Loading transaction features..."
    )

    df = pd.read_csv(
        RAW_DATASET,
        usecols=columns,
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    # --------------------------------------------------------
    # Attach predefined split assignment
    # --------------------------------------------------------

    df = df.merge(
        split_df,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    if len(df) != len(split_df):
        raise ValueError(
            "Dataset and split manifest coverage mismatch."
        )

    # --------------------------------------------------------
    # Integrity checks
    # --------------------------------------------------------

    if df[ID_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate TransactionID detected."
        )

    if df[TARGET].isna().any():
        raise ValueError(
            "Missing target values detected."
        )

    return df


# ============================================================
# Prepare Splits
# ============================================================

def prepare_splits(
    df: pd.DataFrame,
):

    section("APPLYING FEATURE ENGINEERING")

    # --------------------------------------------------------
    # Deterministic feature engineering
    #
    # These features do not use isFraud and do not learn
    # statistics from the dataset.
    # --------------------------------------------------------

    df = engineer_features(
        df
    )

    print(
        "Engineered features added:"
    )

    for feature in ENGINEERED_FEATURES:
        print(
            f"- {feature}"
        )

    # --------------------------------------------------------
    # Separate train and validation
    # --------------------------------------------------------

    train_df = df[
        df["split"] == "train"
    ].copy()

    validation_df = df[
        df["split"] == "validation"
    ].copy()

    print(
        f"\nTrain rows      : {len(train_df):,}"
    )

    print(
        f"Validation rows : {len(validation_df):,}"
    )

    # --------------------------------------------------------
    # Prepare X
    # --------------------------------------------------------

    X_train = prepare_features(
        train_df.drop(
            columns=["split"]
        )
    )

    X_validation = prepare_features(
        validation_df.drop(
            columns=["split"]
        )
    )

    # --------------------------------------------------------
    # Prepare y
    # --------------------------------------------------------

    y_train = prepare_target(
        train_df.drop(
            columns=["split"]
        )
    )

    y_validation = prepare_target(
        validation_df.drop(
            columns=["split"]
        )
    )

    return (
        X_train,
        X_validation,
        y_train,
        y_validation,
    )


# ============================================================
# Class Weight
# ============================================================

def calculate_scale_pos_weight(
    y: pd.Series,
) -> float:

    negative = int(
        (y == 0).sum()
    )

    positive = int(
        (y == 1).sum()
    )

    if positive == 0:
        raise ValueError(
            "No fraud examples found in training data."
        )

    return negative / positive


# ============================================================
# Train Model
# ============================================================

def train_model(
    X_train_processed,
    y_train,
    scale_pos_weight,
):

    section("TRAINING EXPERIMENT 1")

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

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    print(
        f"n_estimators: {N_ESTIMATORS}"
    )

    print(
        f"max_depth: {MAX_DEPTH}"
    )

    print(
        f"learning_rate: {LEARNING_RATE}"
    )

    model.fit(
        X_train_processed,
        y_train,
    )

    print(
        "\nExperiment 1 model training completed."
    )

    return model


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(
    model,
    X_validation_processed,
    y_validation,
):

    section("EXPERIMENT 1 VALIDATION")

    probabilities = model.predict_proba(
        X_validation_processed
    )[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        f"\nPR-AUC    : {pr_auc:.6f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.6f}"
    )

    print(
        f"Precision : {precision:.6f}"
    )

    print(
        f"Recall    : {recall:.6f}"
    )

    print(
        f"F1-Score  : {f1:.6f}"
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

    # --------------------------------------------------------
    # Metrics dictionary
    # --------------------------------------------------------

    metrics = {
        "experiment": "experiment_1",
        "description": (
            "Baseline features + transaction amount "
            "and relative time features"
        ),
        "threshold": THRESHOLD,
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE_BYTREE,
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
    }

    return metrics


# ============================================================
# Compare With Baseline
# ============================================================

def compare_with_baseline(
    experiment_metrics: dict,
):

    baseline_file = (
        METRICS_DIR
        / "baseline_metrics.json"
    )

    if not baseline_file.exists():

        print(
            "\nBaseline metrics file not found."
        )

        return

    with open(
        baseline_file,
        "r",
        encoding="utf-8",
    ) as file:

        baseline = json.load(
            file
        )

    section(
        "BASELINE VS EXPERIMENT 1"
    )

    metrics = [
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1_score",
    ]

    print(
        f"{'Metric':<15}"
        f"{'Baseline':>15}"
        f"{'Experiment 1':>18}"
        f"{'Change':>15}"
    )

    print(
        "-" * 65
    )

    for metric in metrics:

        baseline_value = float(
            baseline[metric]
        )

        experiment_value = float(
            experiment_metrics[metric]
        )

        change = (
            experiment_value
            - baseline_value
        )

        print(
            f"{metric:<15}"
            f"{baseline_value:>15.6f}"
            f"{experiment_value:>18.6f}"
            f"{change:>+15.6f}"
        )

    baseline_pr_auc = float(
        baseline["pr_auc"]
    )

    experiment_pr_auc = float(
        experiment_metrics["pr_auc"]
    )

    print()

    if experiment_pr_auc > baseline_pr_auc:

        print(
            "Experiment 1 improved PR-AUC."
        )

    elif experiment_pr_auc < baseline_pr_auc:

        print(
            "Experiment 1 reduced PR-AUC."
        )

    else:

        print(
            "Experiment 1 produced the same PR-AUC."
        )


# ============================================================
# Save Artifacts
# ============================================================

def save_results(
    model,
    preprocessor,
    metrics,
):

    section(
        "SAVING EXPERIMENT ARTIFACTS"
    )

    model_file = (
        MODEL_DIR
        / "xgboost_experiment_1.json"
    )

    preprocessor_file = (
        MODEL_DIR
        / "preprocessor.joblib"
    )

    metrics_file = (
        METRICS_DIR
        / "experiment_1_metrics.json"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model.save_model(
        model_file
    )

    # --------------------------------------------------------
    # Save EXACT preprocessor fitted on training data
    # --------------------------------------------------------

    joblib.dump(
        preprocessor,
        preprocessor_file,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

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
        f"Model saved:\n{model_file}"
    )

    print(
        f"Preprocessor saved:\n"
        f"{preprocessor_file}"
    )

    print(
        f"Metrics saved:\n{metrics_file}"
    )


# ============================================================
# Main
# ============================================================

def main():

    section(
        "FinGuard AI - Feature Experiment 1"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Feature engineering + split preparation
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = prepare_splits(
        df
    )

    # --------------------------------------------------------
    # Build preprocessor
    # --------------------------------------------------------

    section(
        "BUILDING PREPROCESSOR"
    )

    preprocessor = build_preprocessor(
        X_train
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Fit preprocessing ONLY on training data.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Validation transformation only
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Class imbalance
    # --------------------------------------------------------

    scale_pos_weight = (
        calculate_scale_pos_weight(
            y_train
        )
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        X_train_processed,
        y_train,
        scale_pos_weight,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_model(
        model,
        X_validation_processed,
        y_validation,
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    compare_with_baseline(
        metrics
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        model,
        preprocessor,
        metrics,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    section(
        "EXPERIMENT 1 COMPLETE"
    )

    print(
        "Test set was NOT used."
    )

    print(
        "Training preprocessor was saved successfully."
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()