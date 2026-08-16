"""
FinGuard AI - Feature Engineering

Experiment 1:
- Transaction amount transformations
- Transaction time features

This module creates deterministic features from the original
transaction columns without using the fraud target.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# Project Setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Feature Engineering
# ============================================================

def add_transaction_amount_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add features derived from TransactionAmt.

    Features:
    - TransactionAmtLog
    - TransactionAmtBucket
    """

    result = df.copy()

    if "TransactionAmt" not in result.columns:
        raise ValueError(
            "TransactionAmt column not found."
        )

    # Log transformation reduces the effect of extreme values.
    result["TransactionAmtLog"] = np.log1p(
        result["TransactionAmt"]
    )

    # Simple transaction amount buckets.
    result["TransactionAmtBucket"] = pd.cut(
        result["TransactionAmt"],
        bins=[
            -np.inf,
            25,
            50,
            100,
            250,
            500,
            1000,
            np.inf,
        ],
        labels=[
            "very_low",
            "low",
            "medium",
            "high",
            "very_high",
            "premium",
            "extreme",
        ],
    ).astype("string")

    return result


# ============================================================
# Time Features
# ============================================================

def add_transaction_time_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add time-derived features from TransactionDT.

    TransactionDT represents elapsed time rather than a
    standard calendar timestamp.

    Therefore, these are relative time features.
    """

    result = df.copy()

    if "TransactionDT" not in result.columns:
        raise ValueError(
            "TransactionDT column not found."
        )

    # TransactionDT is measured in seconds.
    seconds_per_hour = 60 * 60
    seconds_per_day = 24 * seconds_per_hour

    result["TransactionHour"] = (
        result["TransactionDT"]
        // seconds_per_hour
    ) % 24

    result["TransactionDay"] = (
        result["TransactionDT"]
        // seconds_per_day
    )

    return result


# ============================================================
# Complete Feature Engineering Pipeline
# ============================================================

def engineer_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply all currently approved deterministic feature
    engineering transformations.
    """

    result = df.copy()

    result = add_transaction_amount_features(
        result
    )

    result = add_transaction_time_features(
        result
    )

    return result


# ============================================================
# Validation
# ============================================================

def validate_features(
    original_df: pd.DataFrame,
    engineered_df: pd.DataFrame,
) -> None:
    """
    Validate that feature engineering:

    - Preserves row count.
    - Preserves original columns.
    - Does not modify the target.
    """

    if len(original_df) != len(engineered_df):
        raise ValueError(
            "Feature engineering changed row count."
        )

    missing_original_columns = [
        column
        for column in original_df.columns
        if column not in engineered_df.columns
    ]

    if missing_original_columns:
        raise ValueError(
            "Original columns were removed: "
            f"{missing_original_columns}"
        )

    if "isFraud" in original_df.columns:

        if not engineered_df[
            "isFraud"
        ].equals(
            original_df[
                "isFraud"
            ]
        ):
            raise ValueError(
                "Target column was modified."
            )


# ============================================================
# Main Validation
# ============================================================

def main() -> None:

    print("=" * 70)
    print(
        "FinGuard AI - Feature Engineering Validation"
    )
    print("=" * 70)

    dataset_path = (
        PROJECT_ROOT
        / "ml"
        / "datasets"
        / "raw"
        / "train_transaction.csv"
    )

    print(
        f"\nLoading sample from:\n{dataset_path}"
    )

    # Only a representative sample is needed to validate
    # deterministic feature generation.
    df = pd.read_csv(
        dataset_path,
        nrows=10000,
        usecols=[
            "TransactionID",
            "isFraud",
            "TransactionDT",
            "TransactionAmt",
        ],
    )

    print(
        f"Original rows: {len(df):,}"
    )

    print(
        f"Original columns: {len(df.columns):,}"
    )

    # --------------------------------------------------------
    # Apply feature engineering
    # --------------------------------------------------------

    engineered_df = engineer_features(
        df
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_features(
        df,
        engineered_df,
    )

    # --------------------------------------------------------
    # Verify new features
    # --------------------------------------------------------

    expected_features = [
        "TransactionAmtLog",
        "TransactionAmtBucket",
        "TransactionHour",
        "TransactionDay",
    ]

    missing_features = [
        feature
        for feature in expected_features
        if feature not in engineered_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Expected engineered features missing: "
            f"{missing_features}"
        )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ENGINEERED FEATURES")
    print("=" * 70)

    for feature in expected_features:
        print(
            f"- {feature}"
        )

    print(
        f"\nEngineered columns: "
        f"{len(engineered_df.columns):,}"
    )

    print(
        "\nSample engineered data:"
    )

    print(
        engineered_df[
            [
                "TransactionAmt",
                "TransactionAmtLog",
                "TransactionAmtBucket",
                "TransactionDT",
                "TransactionHour",
                "TransactionDay",
            ]
        ].head(10).to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("VALIDATION RESULT")
    print("=" * 70)

    print(
        "Row count preserved: PASSED"
    )

    print(
        "Original columns preserved: PASSED"
    )

    print(
        "Target preserved: PASSED"
    )

    print(
        "New feature generation: PASSED"
    )

    print(
        "\nFeature engineering validation: PASSED"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()