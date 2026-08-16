"""
FinGuard AI - Identity Feature Engineering

Day 11 - Experiment 2

Purpose:
    Create useful features from train_identity.csv and
    integrate them with transaction-level data.

Features:
    - HasIdentity
    - IdentityMissingCount
    - IdentityMissingRatio
    - Selected identity numerical features
    - Selected identity categorical features

Important:
    - TransactionID is used only as the join key.
    - isFraud is NEVER used to create identity features.
    - Identity data is LEFT JOINED to transaction data.
    - No target encoding is performed.
    - Identity availability is calculated ONLY from identity columns.
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


# ============================================================
# Dataset Paths
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


# ============================================================
# Configuration
# ============================================================

ID_COLUMN = "TransactionID"
TARGET = "isFraud"


# ============================================================
# Identity Feature Groups
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


# ============================================================
# Utility
# ============================================================

def section(title: str) -> None:

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Load Identity Dataset
# ============================================================

def load_identity_dataset() -> pd.DataFrame:

    section(
        "LOADING IDENTITY DATASET"
    )

    if not IDENTITY_FILE.exists():

        raise FileNotFoundError(
            f"Identity dataset not found:\n"
            f"{IDENTITY_FILE}"
        )

    columns = [
        ID_COLUMN,
        *IDENTITY_NUMERICAL_FEATURES,
        *IDENTITY_CATEGORICAL_FEATURES,
    ]

    # Remove duplicate column names while
    # preserving order.
    columns = list(
        dict.fromkeys(columns)
    )

    identity_df = pd.read_csv(
        IDENTITY_FILE,
        usecols=columns,
    )

    print(
        f"Identity rows: "
        f"{len(identity_df):,}"
    )

    print(
        f"Selected identity columns: "
        f"{len(identity_df.columns):,}"
    )

    return identity_df


# ============================================================
# Validate Identity Key
# ============================================================

def validate_identity_key(
    identity_df: pd.DataFrame,
) -> None:

    section(
        "VALIDATING IDENTITY KEY"
    )

    if ID_COLUMN not in identity_df.columns:

        raise ValueError(
            "TransactionID missing from identity dataset."
        )

    duplicate_count = int(
        identity_df[
            ID_COLUMN
        ].duplicated().sum()
    )

    print(
        f"Duplicate TransactionIDs: "
        f"{duplicate_count:,}"
    )

    if duplicate_count != 0:

        raise ValueError(
            "Identity dataset contains duplicate "
            "TransactionIDs."
        )

    print(
        "Identity key validation: PASSED"
    )


# ============================================================
# Add Identity Availability Features
# ============================================================

def add_identity_availability_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Only actual identity columns are used here.
    #
    # TransactionID and isFraud are intentionally excluded.
    #
    # This prevents the target column from incorrectly making
    # every transaction appear to have identity information.
    # --------------------------------------------------------

    identity_feature_columns = [
        column
        for column in (
            IDENTITY_NUMERICAL_FEATURES
            + IDENTITY_CATEGORICAL_FEATURES
        )
        if column in result.columns
    ]

    if not identity_feature_columns:

        raise ValueError(
            "No identity features available."
        )

    # --------------------------------------------------------
    # HasIdentity
    #
    # 1 = at least one identity feature is available
    # 0 = all identity features are missing
    # --------------------------------------------------------

    result["HasIdentity"] = (
        result[
            identity_feature_columns
        ]
        .notna()
        .any(axis=1)
        .astype("int8")
    )

    # --------------------------------------------------------
    # IdentityMissingCount
    #
    # Number of missing identity fields.
    # --------------------------------------------------------

    result["IdentityMissingCount"] = (
        result[
            identity_feature_columns
        ]
        .isna()
        .sum(axis=1)
    )

    # --------------------------------------------------------
    # IdentityMissingRatio
    #
    # Fraction of identity fields that are missing.
    # --------------------------------------------------------

    feature_count = len(
        identity_feature_columns
    )

    result["IdentityMissingRatio"] = (
        result[
            identity_feature_columns
        ]
        .isna()
        .sum(axis=1)
        / feature_count
    )

    return result


# ============================================================
# Add Identity Features
# ============================================================

def add_identity_features(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame,
) -> pd.DataFrame:

    section(
        "MERGING IDENTITY FEATURES"
    )

    # --------------------------------------------------------
    # Validate transaction key
    # --------------------------------------------------------

    if ID_COLUMN not in transaction_df.columns:

        raise ValueError(
            "TransactionID missing from transaction dataset."
        )

    # --------------------------------------------------------
    # Validate identity key
    # --------------------------------------------------------

    if ID_COLUMN not in identity_df.columns:

        raise ValueError(
            "TransactionID missing from identity dataset."
        )

    if identity_df[
        ID_COLUMN
    ].duplicated().any():

        raise ValueError(
            "Duplicate TransactionIDs detected "
            "in identity dataset."
        )

    # --------------------------------------------------------
    # Preserve original row count
    # --------------------------------------------------------

    original_rows = len(
        transaction_df
    )

    # --------------------------------------------------------
    # LEFT JOIN
    # --------------------------------------------------------

    result = transaction_df.merge(
        identity_df,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    print(
        f"Transaction rows : "
        f"{original_rows:,}"
    )

    print(
        f"Merged rows      : "
        f"{len(result):,}"
    )

    # --------------------------------------------------------
    # Row count integrity
    # --------------------------------------------------------

    if len(result) != original_rows:

        raise ValueError(
            "Identity merge changed transaction row count."
        )

    print(
        "LEFT JOIN row preservation: PASSED"
    )

    # --------------------------------------------------------
    # Add availability features
    # --------------------------------------------------------

    result = add_identity_availability_features(
        result
    )

    return result


# ============================================================
# Validate Target Integrity
# ============================================================

def validate_target(
    original_df: pd.DataFrame,
    engineered_df: pd.DataFrame,
) -> None:

    section(
        "VALIDATING TARGET INTEGRITY"
    )

    if TARGET not in original_df.columns:

        raise ValueError(
            "isFraud missing from transaction dataset."
        )

    if TARGET not in engineered_df.columns:

        raise ValueError(
            "isFraud missing after identity feature engineering."
        )

    if not engineered_df[
        TARGET
    ].equals(
        original_df[
            TARGET
        ]
    ):

        raise ValueError(
            "Target changed during identity feature engineering."
        )

    print(
        "Target preservation: PASSED"
    )


# ============================================================
# Validate Generated Features
# ============================================================

def validate_generated_features(
    df: pd.DataFrame,
) -> None:

    section(
        "VALIDATING GENERATED FEATURES"
    )

    required_features = [
        "HasIdentity",
        "IdentityMissingCount",
        "IdentityMissingRatio",
    ]

    missing_features = [
        feature
        for feature in required_features
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing generated identity features: "
            f"{missing_features}"
        )

    # --------------------------------------------------------
    # HasIdentity validation
    # --------------------------------------------------------

    has_identity_values = set(
        df[
            "HasIdentity"
        ]
        .dropna()
        .unique()
    )

    if not has_identity_values.issubset(
        {0, 1}
    ):

        raise ValueError(
            "HasIdentity contains invalid values."
        )

    # --------------------------------------------------------
    # IdentityMissingCount validation
    # --------------------------------------------------------

    if (
        df[
            "IdentityMissingCount"
        ] < 0
    ).any():

        raise ValueError(
            "IdentityMissingCount contains "
            "negative values."
        )

    # --------------------------------------------------------
    # IdentityMissingRatio validation
    # --------------------------------------------------------

    invalid_ratio = (
        (
            df[
                "IdentityMissingRatio"
            ] < 0
        )
        |
        (
            df[
                "IdentityMissingRatio"
            ] > 1
        )
    )

    if invalid_ratio.any():

        raise ValueError(
            "IdentityMissingRatio contains invalid values."
        )

    print(
        "HasIdentity validation: PASSED"
    )

    print(
        "IdentityMissingCount validation: PASSED"
    )

    print(
        "IdentityMissingRatio validation: PASSED"
    )


# ============================================================
# Display Statistics
# ============================================================

def display_identity_statistics(
    df: pd.DataFrame,
) -> None:

    section(
        "IDENTITY FEATURE STATISTICS"
    )

    total_rows = len(df)

    identity_available = int(
        df[
            "HasIdentity"
        ].sum()
    )

    identity_missing = (
        total_rows
        - identity_available
    )

    coverage = (
        identity_available
        / total_rows
        * 100
    )

    print(
        f"Total transactions      : "
        f"{total_rows:,}"
    )

    print(
        f"Identity available      : "
        f"{identity_available:,}"
    )

    print(
        f"Identity unavailable    : "
        f"{identity_missing:,}"
    )

    print(
        f"Identity coverage       : "
        f"{coverage:.2f}%"
    )

    print(
        "\nMissing identity statistics:"
    )

    print(
        df[
            [
                "IdentityMissingCount",
                "IdentityMissingRatio",
            ]
        ]
        .describe()
        .to_string()
    )


# ============================================================
# Display Feature List
# ============================================================

def display_feature_list() -> None:

    section(
        "IDENTITY FEATURES"
    )

    print(
        "Selected numerical identity features:"
    )

    for feature in IDENTITY_NUMERICAL_FEATURES:

        print(
            f"- {feature}"
        )

    print(
        "\nSelected categorical identity features:"
    )

    for feature in IDENTITY_CATEGORICAL_FEATURES:

        print(
            f"- {feature}"
        )

    print(
        "\nGenerated availability features:"
    )

    print(
        "- HasIdentity"
    )

    print(
        "- IdentityMissingCount"
    )

    print(
        "- IdentityMissingRatio"
    )


# ============================================================
# Main Validation
# ============================================================

def main():

    section(
        "FinGuard AI - Identity Feature Engineering"
    )

    # --------------------------------------------------------
    # Load identity dataset
    # --------------------------------------------------------

    identity_df = (
        load_identity_dataset()
    )

    # --------------------------------------------------------
    # Validate identity key
    # --------------------------------------------------------

    validate_identity_key(
        identity_df
    )

    # --------------------------------------------------------
    # Load transaction sample
    #
    # 10,000 rows are sufficient for validating the feature
    # engineering logic.
    # --------------------------------------------------------

    section(
        "LOADING TRANSACTION SAMPLE"
    )

    if not TRANSACTION_FILE.exists():

        raise FileNotFoundError(
            f"Transaction dataset not found:\n"
            f"{TRANSACTION_FILE}"
        )

    transaction_df = pd.read_csv(
        TRANSACTION_FILE,
        usecols=[
            ID_COLUMN,
            TARGET,
        ],
        nrows=10000,
    )

    print(
        f"Transaction sample rows: "
        f"{len(transaction_df):,}"
    )

    # --------------------------------------------------------
    # Merge identity features
    # --------------------------------------------------------

    engineered_df = add_identity_features(
        transaction_df,
        identity_df,
    )

    # --------------------------------------------------------
    # Validate target
    # --------------------------------------------------------

    validate_target(
        transaction_df,
        engineered_df,
    )

    # --------------------------------------------------------
    # Validate generated features
    # --------------------------------------------------------

    validate_generated_features(
        engineered_df
    )

    # --------------------------------------------------------
    # Display statistics
    # --------------------------------------------------------

    display_identity_statistics(
        engineered_df
    )

    # --------------------------------------------------------
    # Display feature list
    # --------------------------------------------------------

    display_feature_list()

    # --------------------------------------------------------
    # Display sample
    # --------------------------------------------------------

    section(
        "SAMPLE IDENTITY FEATURES"
    )

    display_columns = [
        ID_COLUMN,
        TARGET,
        "HasIdentity",
        "IdentityMissingCount",
        "IdentityMissingRatio",
        "id_01",
        "id_02",
        "id_12",
        "DeviceType",
        "DeviceInfo",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in engineered_df.columns
    ]

    print(
        engineered_df[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    section(
        "IDENTITY FEATURE VALIDATION COMPLETE"
    )

    print(
        "Identity features generated: PASSED"
    )

    print(
        "Identity availability features: PASSED"
    )

    print(
        "Target integrity: PASSED"
    )

    print(
        "Test set was NOT used."
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()