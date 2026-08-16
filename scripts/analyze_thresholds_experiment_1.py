"""
FinGuard AI - Experiment 1 Threshold Analysis

Analyzes the precision-recall trade-off for Experiment 1.

Experiment 1:
    Baseline features
    +
    Transaction amount features
    +
    Relative time features

Important:
- Validation set only.
- Test set remains untouched.
- No model retraining.
"""

from pathlib import Path
import sys

import pandas as pd
import joblib

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from xgboost import XGBClassifier


# ============================================================
# Project Setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ml.preprocessing.preprocessor import (
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

MODEL_FILE = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "experiment_1"
    / "xgboost_experiment_1.json"
)

# Experiment 1 needs the preprocessing pipeline fitted
# specifically for the engineered feature set.
PREPROCESSOR_FILE = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "experiment_1"
    / "preprocessor.joblib"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "ml"
    / "metrics"
    / "threshold_analysis_experiment_1.csv"
)


# ============================================================
# Import Feature Engineering
# ============================================================

from scripts.feature_engineering import (
    engineer_features,
)


# ============================================================
# Configuration
# ============================================================

TARGET = "isFraud"
ID_COLUMN = "TransactionID"

THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]


# ============================================================
# Utility
# ============================================================

def section(title: str) -> None:

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Load Validation Data
# ============================================================

def load_validation_data() -> pd.DataFrame:

    section("LOADING VALIDATION DATA")

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

    validation_ids = split_df.loc[
        split_df["split"] == "validation",
        ID_COLUMN,
    ]

    print(
        f"Validation transactions: "
        f"{len(validation_ids):,}"
    )

    selected_ids = set(
        validation_ids
    )

    chunks = []

    for chunk in pd.read_csv(
        RAW_DATASET,
        chunksize=50000,
    ):

        selected = chunk[
            chunk[ID_COLUMN].isin(
                selected_ids
            )
        ]

        if not selected.empty:
            chunks.append(selected)

        if sum(
            len(part)
            for part in chunks
        ) >= len(selected_ids):
            break

    if not chunks:
        raise ValueError(
            "Validation transactions could not be loaded."
        )

    df = pd.concat(
        chunks,
        ignore_index=True,
    )

    if len(df) != len(validation_ids):
        raise ValueError(
            "Validation data coverage mismatch."
        )

    df = df.merge(
        split_df,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    if not (
        df["split"] == "validation"
    ).all():
        raise ValueError(
            "Non-validation records detected."
        )

    return df


# ============================================================
# Load Model
# ============================================================

def load_model() -> XGBClassifier:

    section("LOADING EXPERIMENT 1 MODEL")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    model = XGBClassifier()

    model.load_model(
        MODEL_FILE
    )

    print(
        "Experiment 1 model loaded successfully."
    )

    return model


# ============================================================
# Generate Probabilities
# ============================================================

def generate_predictions(
    df: pd.DataFrame,
    model: XGBClassifier,
):

    section(
        "GENERATING EXPERIMENT 1 PREDICTIONS"
    )

    if not PREPROCESSOR_FILE.exists():
        raise FileNotFoundError(
            f"Experiment 1 preprocessor not found:\n"
            f"{PREPROCESSOR_FILE}"
        )

    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    # --------------------------------------------------------
    # Apply EXACTLY the same deterministic feature engineering
    # used during Experiment 1 training.
    # --------------------------------------------------------

    df = engineer_features(
        df
    )

    X_validation = prepare_features(
        df.drop(
            columns=["split"]
        )
    )

    y_validation = prepare_target(
        df.drop(
            columns=["split"]
        )
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

    print(
        "Generating fraud probabilities..."
    )

    probabilities = model.predict_proba(
        X_validation_processed
    )[:, 1]

    print(
        "Prediction generation completed."
    )

    return (
        y_validation,
        probabilities,
    )


# ============================================================
# Threshold Analysis
# ============================================================

def analyze_thresholds(
    y_true,
    probabilities,
) -> pd.DataFrame:

    section(
        "EXPERIMENT 1 THRESHOLD ANALYSIS"
    )

    results = []

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        predicted_fraud_count = int(
            predictions.sum()
        )

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "predicted_fraud_count":
                    predicted_fraud_count,
            }
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# Display Results
# ============================================================

def display_results(
    results_df: pd.DataFrame,
    y_true,
    probabilities,
) -> None:

    print(
        "\nThreshold Performance:"
    )

    print()

    print(
        results_df.to_string(
            index=False,
            formatters={
                "threshold":
                    "{:.2f}".format,
                "precision":
                    "{:.4f}".format,
                "recall":
                    "{:.4f}".format,
                "f1_score":
                    "{:.4f}".format,
            },
        )
    )

    # --------------------------------------------------------
    # Best F1
    # --------------------------------------------------------

    best_f1_row = results_df.loc[
        results_df["f1_score"].idxmax()
    ]

    print(
        "\nBest F1 threshold:"
    )

    print(
        f"Threshold : "
        f"{best_f1_row['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best_f1_row['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_f1_row['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{best_f1_row['f1_score']:.4f}"
    )

    # --------------------------------------------------------
    # PR-AUC
    # --------------------------------------------------------

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    print(
        f"\nExperiment 1 Validation PR-AUC: "
        f"{pr_auc:.6f}"
    )


# ============================================================
# Save Results
# ============================================================

def save_results(
    results_df: pd.DataFrame,
) -> None:

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nThreshold analysis saved to:"
        f"\n{OUTPUT_FILE}"
    )


# ============================================================
# Main
# ============================================================

def main():

    section(
        "FinGuard AI - Experiment 1 Threshold Analysis"
    )

    df = load_validation_data()

    model = load_model()

    (
        y_validation,
        probabilities,
    ) = generate_predictions(
        df,
        model,
    )

    results_df = analyze_thresholds(
        y_validation,
        probabilities,
    )

    display_results(
        results_df,
        y_validation,
        probabilities,
    )

    save_results(
        results_df
    )

    section(
        "EXPERIMENT 1 THRESHOLD ANALYSIS COMPLETE"
    )

    print(
        "Test set was NOT used."
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()