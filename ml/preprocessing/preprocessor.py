"""
FinGuard AI - Preprocessing Pipeline

Provides a reusable preprocessing pipeline for the
IEEE-CIS fraud detection dataset.

Design goals:
- Leakage-safe preprocessing
- Separate numerical and categorical handling
- Unknown-category protection
- Missing-value handling
- Reusable for training and inference
"""

from pathlib import Path
import sys

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# Project Import
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ml.preprocessing.feature_config import (
    TARGET_COLUMN,
    IDENTIFIER_COLUMNS,
)


# ============================================================
# Columns That Must Never Be Used as Model Features
# ============================================================

EXCLUDED_COLUMNS = {
    TARGET_COLUMN,
    *IDENTIFIER_COLUMNS,
}


# ============================================================
# Feature Detection
# ============================================================

def detect_feature_types(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Detect numerical and categorical columns.

    Target and identifier columns are excluded.
    """

    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    numerical_columns = (
        df[feature_columns]
        .select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        df[feature_columns]
        .select_dtypes(
            include=["object", "category", "string"]
        )
        .columns
        .tolist()
    )

    return (
        numerical_columns,
        categorical_columns,
    )


# ============================================================
# Numerical Pipeline
# ============================================================

def build_numerical_pipeline() -> Pipeline:
    """
    Numerical preprocessing:

    1. Median imputation
    2. Missing-value indicators
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
        ]
    )


# ============================================================
# Categorical Pipeline
# ============================================================

def build_categorical_pipeline() -> Pipeline:
    """
    Categorical preprocessing:

    1. Replace missing values with explicit category
    2. One-hot encode categories
    3. Ignore unseen categories during inference
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )


# ============================================================
# Complete Preprocessor
# ============================================================

def build_preprocessor(
    df: pd.DataFrame,
) -> ColumnTransformer:
    """
    Build the complete preprocessing transformer.

    IMPORTANT:
    This function only constructs the transformer.
    It does not fit it.

    Fitting must happen on training data only.
    """

    (
        numerical_columns,
        categorical_columns,
    ) = detect_feature_types(df)

    numerical_pipeline = (
        build_numerical_pipeline()
    )

    categorical_pipeline = (
        build_categorical_pipeline()
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ============================================================
# Feature Preparation
# ============================================================

def prepare_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove target and identifier columns.

    Returns only model input features.
    """

    columns_to_drop = [
        column
        for column in EXCLUDED_COLUMNS
        if column in df.columns
    ]

    return df.drop(
        columns=columns_to_drop
    )


# ============================================================
# Target Preparation
# ============================================================

def prepare_target(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Extract the target column.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "not found in dataset."
        )

    return df[TARGET_COLUMN]


# ============================================================
# Pipeline Summary
# ============================================================

def describe_preprocessor(
    df: pd.DataFrame,
) -> None:
    """
    Print detected feature groups and preprocessing summary.
    """

    (
        numerical_columns,
        categorical_columns,
    ) = detect_feature_types(df)

    print("=" * 70)
    print("FinGuard AI - Preprocessing Configuration")
    print("=" * 70)

    print(
        f"\nTotal input columns: "
        f"{len(df.columns):,}"
    )

    print(
        f"Numerical features: "
        f"{len(numerical_columns):,}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_columns):,}"
    )

    print(
        f"Excluded columns: "
        f"{len(EXCLUDED_COLUMNS):,}"
    )

    print("\nNumerical preprocessing:")
    print("- Median imputation")
    print("- Missing-value indicators")

    print("\nCategorical preprocessing:")
    print("- Missing values → Unknown")
    print("- One-hot encoding")
    print("- Unknown categories ignored")

    print("\nLeakage policy:")
    print(
        "- Preprocessor must be fitted on training data only."
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    raw_dataset = (
        PROJECT_ROOT
        / "ml"
        / "datasets"
        / "raw"
        / "train_transaction.csv"
    )

    print("=" * 70)
    print("FinGuard AI - Preprocessor Validation")
    print("=" * 70)

    print(
        f"\nLoading schema from:\n{raw_dataset}"
    )

    # Only a small sample is required to detect dtypes.
    sample = pd.read_csv(
        raw_dataset,
        nrows=1000,
    )

    describe_preprocessor(sample)

    preprocessor = build_preprocessor(
        sample
    )

    print(
        "\nPreprocessor object created successfully."
    )

    print(
        "\nTransformer configuration:"
    )

    print(
        preprocessor
    )

    print(
        "\nValidation complete."
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()