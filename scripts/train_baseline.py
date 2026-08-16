"""
FinGuard AI - XGBoost Baseline Model

Purpose:
    Train a reproducible baseline fraud detection model.

Important:
    - Uses the predefined train/validation split.
    - Preprocessing is fitted on TRAIN only.
    - Validation data is transformed only.
    - Test data is NOT used during baseline development.
    - Uses core features to establish a clean baseline.
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

MODEL_DIR = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "baseline"
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
# Configuration
# ============================================================

RANDOM_STATE = 42

# Initial baseline.
N_ESTIMATORS = 300
MAX_DEPTH = 6
LEARNING_RATE = 0.08
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8

THRESHOLD = 0.50


# ============================================================
# Core Features
# ============================================================

CORE_FEATURES = [
    # Transaction
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",

    # Card
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",

    # Address / distance
    "addr1",
    "addr2",
    "dist1",
    "dist2",

    # Email
    "P_emaildomain",
    "R_emaildomain",

    # Verification
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


TARGET = "isFraud"
ID_COLUMN = "TransactionID"


# ============================================================
# Utility
# ============================================================

def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Validate Columns
# ============================================================

def validate_columns(
    columns: list[str],
    available_columns: list[str],
) -> None:

    missing = [
        column
        for column in columns
        if column not in available_columns
    ]

    if missing:
        raise ValueError(
            "Required columns are missing:\n"
            + "\n".join(
                f"- {column}"
                for column in missing
            )
        )


# ============================================================
# Load Data
# ============================================================

def load_training_data() -> pd.DataFrame:

    section("LOADING DATA")

    if not RAW_DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{RAW_DATASET}"
        )

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Split manifest not found:\n{SPLIT_FILE}"
        )

    print("Loading split manifest...")

    split_df = pd.read_csv(
        SPLIT_FILE,
        usecols=[
            ID_COLUMN,
            "split",
        ],
    )

    print(
        f"Split records: {len(split_df):,}"
    )

    print("Loading core transaction features...")

    columns_to_load = [
        ID_COLUMN,
        TARGET,
        *CORE_FEATURES,
    ]

    # Remove duplicates while preserving order.
    columns_to_load = list(
        dict.fromkeys(columns_to_load)
    )

    df = pd.read_csv(
        RAW_DATASET,
        usecols=columns_to_load,
    )

    print(
        f"Loaded rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # Join split assignment
    # --------------------------------------------------------

    df = df.merge(
        split_df,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    if len(df) != len(split_df):
        raise ValueError(
            "Transaction/split manifest coverage mismatch."
        )

    # --------------------------------------------------------
    # Integrity
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
    section("PREPARING TRAIN / VALIDATION SPLITS")

    train_df = df[
        df["split"] == "train"
    ].copy()

    validation_df = df[
        df["split"] == "validation"
    ].copy()

    print(
        f"Train rows      : {len(train_df):,}"
    )

    print(
        f"Validation rows : {len(validation_df):,}"
    )

    # --------------------------------------------------------
    # Features
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
    # Targets
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
# Calculate Class Weight
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
            "Training set contains no fraud cases."
        )

    return negative / positive


# ============================================================
# Train Model
# ============================================================

def train_model(
    X_train_processed,
    y_train: pd.Series,
    scale_pos_weight: float,
) -> XGBClassifier:

    section("TRAINING XGBOOST BASELINE")

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
        "\nModel training completed."
    )

    return model


# ============================================================
# Evaluate
# ============================================================

def evaluate_model(
    model: XGBClassifier,
    X_validation_processed,
    y_validation: pd.Series,
):
    section("BASELINE VALIDATION")

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

    metrics = {
        "model": "XGBoost Baseline",
        "random_state": RANDOM_STATE,
        "threshold": THRESHOLD,
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "scale_pos_weight": float(
            (
                (y_validation == 0).sum()
                / (y_validation == 1).sum()
            )
        ),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
    }

    return metrics


# ============================================================
# Save Artifacts
# ============================================================

def save_artifacts(
    model: XGBClassifier,
    preprocessor,
    metrics: dict,
) -> None:

    section("SAVING BASELINE ARTIFACTS")

    model_file = (
        MODEL_DIR
        / "xgboost_baseline.json"
    )

    preprocessor_file = (
        MODEL_DIR
        / "preprocessor.joblib"
    )

    metrics_file = (
        METRICS_DIR
        / "baseline_metrics.json"
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

def main() -> None:

    section(
        "FinGuard AI - XGBoost Baseline"
    )

    df = load_training_data()

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = prepare_splits(df)

    section(
        "BUILDING PREPROCESSOR"
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

    # --------------------------------------------------------
    # Class imbalance
    # --------------------------------------------------------

    scale_pos_weight = (
        calculate_scale_pos_weight(
            y_train
        )
    )

    model = train_model(
        X_train_processed,
        y_train,
        scale_pos_weight,
    )

    metrics = evaluate_model(
        model,
        X_validation_processed,
        y_validation,
    )

    save_artifacts(
        model,
        preprocessor,
        metrics,
    )

    section(
        "BASELINE TRAINING COMPLETE"
    )

    print(
        "Validation metrics saved successfully."
    )

    print(
        "\nTest set was NOT used."
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()