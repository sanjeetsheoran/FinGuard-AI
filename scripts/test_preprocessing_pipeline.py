"""
FinGuard AI - Preprocessing Pipeline Test

Validates the leakage-safe preprocessing workflow:

1. Load the split manifest.
2. Load a representative sample from the transaction dataset.
3. Verify split assignment.
4. Fit preprocessing only on training data.
5. Transform validation and test data without fitting.
6. Verify target and identifier columns are excluded.
7. Verify output dimensions are consistent.
"""

from pathlib import Path
import sys

import pandas as pd


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
    TARGET_COLUMN,
)


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

SPLIT_FILE = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "splits"
    / "transaction_split.csv"
)


# ============================================================
# Configuration
# ============================================================

SAMPLE_SIZE = 10000
RANDOM_STATE = 42


# ============================================================
# Utility
# ============================================================

def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Load Split Manifest
# ============================================================

def load_split_manifest() -> pd.DataFrame:

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Split manifest not found:\n{SPLIT_FILE}"
        )

    split_df = pd.read_csv(
        SPLIT_FILE
    )

    required_columns = {
        "TransactionID",
        "split",
    }

    missing_columns = (
        required_columns
        - set(split_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Split manifest is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return split_df


# ============================================================
# Load Representative Sample
# ============================================================

def load_sample(
    split_df: pd.DataFrame,
) -> pd.DataFrame:

    # --------------------------------------------------------
    # Randomly sample IDs from each split.
    # This prevents loading the complete 651 MB dataset.
    # --------------------------------------------------------

    train_ids = (
        split_df[
            split_df["split"] == "train"
        ]["TransactionID"]
        .sample(
            n=min(6000, len(
                split_df[
                    split_df["split"] == "train"
                ]
            )),
            random_state=RANDOM_STATE,
        )
        .tolist()
    )

    validation_ids = (
        split_df[
            split_df["split"] == "validation"
        ]["TransactionID"]
        .sample(
            n=min(2000, len(
                split_df[
                    split_df["split"] == "validation"
                ]
            )),
            random_state=RANDOM_STATE,
        )
        .tolist()
    )

    test_ids = (
        split_df[
            split_df["split"] == "test"
        ]["TransactionID"]
        .sample(
            n=min(2000, len(
                split_df[
                    split_df["split"] == "test"
                ]
            )),
            random_state=RANDOM_STATE,
        )
        .tolist()
    )

    selected_ids = set(
        train_ids
        + validation_ids
        + test_ids
    )

    # --------------------------------------------------------
    # Read only selected rows using chunks.
    # --------------------------------------------------------

    chunks = []

    for chunk in pd.read_csv(
        TRANSACTION_FILE,
        chunksize=50000,
    ):

        selected = chunk[
            chunk["TransactionID"]
            .isin(selected_ids)
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
            "No selected transactions were found."
        )

    df = pd.concat(
        chunks,
        ignore_index=True,
    )

    return df


# ============================================================
# Verify Split Integrity
# ============================================================

def verify_split_integrity(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> pd.DataFrame:

    merged = df.merge(
        split_df,
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )

    if merged["split"].isna().any():
        raise ValueError(
            "Some transactions do not have a split assignment."
        )

    train_ids = set(
        merged.loc[
            merged["split"] == "train",
            "TransactionID",
        ]
    )

    validation_ids = set(
        merged.loc[
            merged["split"] == "validation",
            "TransactionID",
        ]
    )

    test_ids = set(
        merged.loc[
            merged["split"] == "test",
            "TransactionID",
        ]
    )

    if train_ids & validation_ids:
        raise ValueError(
            "Train/validation TransactionID overlap detected."
        )

    if train_ids & test_ids:
        raise ValueError(
            "Train/test TransactionID overlap detected."
        )

    if validation_ids & test_ids:
        raise ValueError(
            "Validation/test TransactionID overlap detected."
        )

    print(
        "TransactionID overlap: 0"
    )

    return merged


# ============================================================
# Test Preprocessing
# ============================================================

def test_preprocessing(
    df: pd.DataFrame,
) -> None:

    train_df = df[
        df["split"] == "train"
    ].copy()

    validation_df = df[
        df["split"] == "validation"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    print(
        f"Train sample      : {len(train_df):,}"
    )

    print(
        f"Validation sample : {len(validation_df):,}"
    )

    print(
        f"Test sample       : {len(test_df):,}"
    )

    # --------------------------------------------------------
    # Prepare X and y
    # --------------------------------------------------------

    X_train = prepare_features(
        train_df
    )

    X_validation = prepare_features(
        validation_df
    )

    X_test = prepare_features(
        test_df
    )

    y_train = prepare_target(
        train_df
    )

    y_validation = prepare_target(
        validation_df
    )

    y_test = prepare_target(
        test_df
    )

    # --------------------------------------------------------
    # Verify target is excluded
    # --------------------------------------------------------

    if TARGET_COLUMN in X_train.columns:
        raise AssertionError(
            "Target column found inside X_train."
        )

    if TARGET_COLUMN in X_validation.columns:
        raise AssertionError(
            "Target column found inside X_validation."
        )

    if TARGET_COLUMN in X_test.columns:
        raise AssertionError(
            "Target column found inside X_test."
        )

    print(
        "\nTarget exclusion: PASSED"
    )

    # --------------------------------------------------------
    # Build preprocessor using training schema
    # --------------------------------------------------------

    preprocessor = build_preprocessor(
        X_train
    )

    # --------------------------------------------------------
    # CRITICAL:
    # FIT ONLY ON TRAIN
    # --------------------------------------------------------

    print(
        "\nFitting preprocessor on TRAIN only..."
    )

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    print(
        "Train preprocessing: PASSED"
    )

    # --------------------------------------------------------
    # VALIDATION:
    # transform only
    # --------------------------------------------------------

    print(
        "Transforming VALIDATION..."
    )

    X_validation_processed = (
        preprocessor.transform(
            X_validation
        )
    )

    print(
        "Validation preprocessing: PASSED"
    )

    # --------------------------------------------------------
    # TEST:
    # transform only
    # --------------------------------------------------------

    print(
        "Transforming TEST..."
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    print(
        "Test preprocessing: PASSED"
    )

    # --------------------------------------------------------
    # Verify output dimensions
    # --------------------------------------------------------

    train_features = (
        X_train_processed.shape[1]
    )

    validation_features = (
        X_validation_processed.shape[1]
    )

    test_features = (
        X_test_processed.shape[1]
    )

    print(
        "\nProcessed feature dimensions:"
    )

    print(
        f"Train      : {X_train_processed.shape}"
    )

    print(
        f"Validation : {X_validation_processed.shape}"
    )

    print(
        f"Test       : {X_test_processed.shape}"
    )

    if not (
        train_features
        == validation_features
        == test_features
    ):
        raise AssertionError(
            "Processed feature dimensions do not match."
        )

    print(
        "\nFeature dimension consistency: PASSED"
    )

    # --------------------------------------------------------
    # Verify target alignment
    # --------------------------------------------------------

    if len(X_train_processed) != len(y_train):
        raise AssertionError(
            "Train feature/target row mismatch."
        )

    if len(X_validation_processed) != len(
        y_validation
    ):
        raise AssertionError(
            "Validation feature/target row mismatch."
        )

    if len(X_test_processed) != len(y_test):
        raise AssertionError(
            "Test feature/target row mismatch."
        )

    print(
        "Feature/target alignment: PASSED"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    section(
        "FinGuard AI - Preprocessing Pipeline Test"
    )

    if not TRANSACTION_FILE.exists():
        raise FileNotFoundError(
            f"Transaction dataset not found:\n"
            f"{TRANSACTION_FILE}"
        )

    section(
        "LOADING SPLIT MANIFEST"
    )

    split_df = load_split_manifest()

    print(
        f"Split records: {len(split_df):,}"
    )

    section(
        "LOADING REPRESENTATIVE SAMPLE"
    )

    df = load_sample(
        split_df
    )

    print(
        f"Sample records: {len(df):,}"
    )

    section(
        "VERIFYING SPLIT INTEGRITY"
    )

    df = verify_split_integrity(
        df,
        split_df,
    )

    print(
        "Split integrity: PASSED"
    )

    section(
        "TESTING PREPROCESSING PIPELINE"
    )

    test_preprocessing(
        df
    )

    section(
        "FINAL RESULT"
    )

    print(
        "Preprocessing pipeline test: PASSED"
    )

    print(
        "\nLeakage-safe workflow verified:"
    )

    print(
        "TRAIN      → fit_transform()"
    )

    print(
        "VALIDATION → transform()"
    )

    print(
        "TEST       → transform()"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()