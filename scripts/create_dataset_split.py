from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# FinGuard AI - Dataset Split
# ============================================================

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATASET = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "raw"
    / "train_transaction.csv"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "splits"
)


# ============================================================
# Validation
# ============================================================

def validate_configuration() -> None:

    total = (
        TRAIN_SIZE
        + VALIDATION_SIZE
        + TEST_SIZE
    )

    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "Train, validation and test sizes must sum to 1.0"
        )

    if not RAW_DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{RAW_DATASET}"
        )


# ============================================================
# Main Split Function
# ============================================================

def create_split() -> None:

    print("=" * 70)
    print("FinGuard AI - Dataset Split")
    print("=" * 70)

    validate_configuration()

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nLoading transaction dataset...")

    # Only load the columns required for splitting.
    df = pd.read_csv(
        RAW_DATASET,
        usecols=[
            "TransactionID",
            "isFraud"
        ]
    )

    print("Dataset loaded successfully.")

    print(f"Total transactions: {len(df):,}")

    # --------------------------------------------------------
    # Basic Integrity Checks
    # --------------------------------------------------------

    if df["TransactionID"].duplicated().any():
        raise ValueError(
            "Duplicate TransactionID values detected."
        )

    if df["isFraud"].isna().any():
        raise ValueError(
            "Missing values detected in target column."
        )

    unique_targets = set(
        df["isFraud"].unique()
    )

    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            "Target column contains values other than 0 and 1."
        )

    # --------------------------------------------------------
    # First Split
    #
    # 70% Train
    # 30% Temporary
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=(
            VALIDATION_SIZE
            + TEST_SIZE
        ),
        stratify=df["isFraud"],
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Second Split
    #
    # Temporary 30%
    # -> 15% Validation
    # -> 15% Test
    #
    # Therefore half of temporary data goes to validation.
    # --------------------------------------------------------

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["isFraud"],
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Add Split Labels
    # --------------------------------------------------------

    train_df = train_df[
        ["TransactionID"]
    ].copy()

    train_df["split"] = "train"

    validation_df = validation_df[
        ["TransactionID"]
    ].copy()

    validation_df["split"] = "validation"

    test_df = test_df[
        ["TransactionID"]
    ].copy()

    test_df["split"] = "test"

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    split_df = pd.concat(
        [
            train_df,
            validation_df,
            test_df
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Final Integrity Checks
    # --------------------------------------------------------

    expected_rows = len(df)

    if len(split_df) != expected_rows:
        raise ValueError(
            "Split row count does not match original dataset."
        )

    if split_df["TransactionID"].duplicated().any():
        raise ValueError(
            "TransactionID appears in more than one split."
        )

    if split_df["TransactionID"].nunique() != (
        df["TransactionID"].nunique()
    ):
        raise ValueError(
            "TransactionID coverage mismatch."
        )

    # --------------------------------------------------------
    # Save Split Manifest
    # --------------------------------------------------------

    output_file = (
        SPLIT_DIR
        / "transaction_split.csv"
    )

    split_df.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SPLIT DISTRIBUTION")
    print("=" * 70)

    split_counts = (
        split_df["split"]
        .value_counts()
    )

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        count = split_counts.get(
            split_name,
            0
        )

        percentage = (
            count
            / len(split_df)
        ) * 100

        print(
            f"{split_name.capitalize():12}"
            f"{count:>12,}"
            f"{percentage:>10.2f}%"
        )

    # --------------------------------------------------------
    # Fraud Distribution
    # --------------------------------------------------------

    analysis_df = df.merge(
        split_df,
        on="TransactionID",
        how="inner",
        validate="one_to_one"
    )

    print("\n" + "=" * 70)
    print("FRAUD DISTRIBUTION BY SPLIT")
    print("=" * 70)

    fraud_stats = (
        analysis_df
        .groupby("split")["isFraud"]
        .agg(
            transactions="count",
            frauds="sum",
            fraud_rate="mean"
        )
    )

    fraud_stats["fraud_rate"] *= 100

    fraud_stats = fraud_stats.reindex(
        ["train", "validation", "test"]
    )

    print(
        fraud_stats.round(4)
    )

    # --------------------------------------------------------
    # Coverage Check
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INTEGRITY CHECK")
    print("=" * 70)

    print(
        f"Original transactions : {len(df):,}"
    )

    print(
        f"Split transactions    : {len(split_df):,}"
    )

    print(
        f"Unique TransactionIDs  : "
        f"{split_df['TransactionID'].nunique():,}"
    )

    print(
        "TransactionID overlap : 0"
    )

    print(
        "\nDataset split integrity: PASSED"
    )

    print(
        f"\nSaved split manifest:"
        f"\n{output_file}"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    create_split()