from pathlib import Path

import pandas as pd


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "ml" / "datasets" / "raw"

TRANSACTION_FILE = RAW_DATA_DIR / "train_transaction.csv"
IDENTITY_FILE = RAW_DATA_DIR / "train_identity.csv"


# ============================================================
# Utility Functions
# ============================================================

def print_section(title: str) -> None:
    """Print a formatted section heading."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Dataset Validation
# ============================================================

def validate_file(file_path: Path, dataset_name: str):

    print_section(f"{dataset_name} VALIDATION")

    # --------------------------------------------------------
    # File existence
    # --------------------------------------------------------

    if not file_path.exists():

        print("ERROR: File not found.")
        print(f"Expected path: {file_path}")

        return None

    print(f"File: {file_path.name}")

    # --------------------------------------------------------
    # File size
    # --------------------------------------------------------

    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    print(f"Size: {file_size_mb:.2f} MB")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    try:

        df = pd.read_csv(file_path)

    except Exception as error:

        print("ERROR while reading file:")
        print(error)

        return None

    print("Dataset loaded successfully.")

    # --------------------------------------------------------
    # Dataset shape
    # --------------------------------------------------------

    print(f"\nRows:    {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")

    # --------------------------------------------------------
    # Column information
    # --------------------------------------------------------

    print_section(f"{dataset_name} COLUMN INFORMATION")

    print("First 15 columns:")

    for column in df.columns[:15]:

        print(f" - {column}")

    # --------------------------------------------------------
    # Data types
    # --------------------------------------------------------

    print_section(f"{dataset_name} DATA TYPES")

    print(df.dtypes.value_counts())

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print_section(f"{dataset_name} MISSING VALUES")

    missing = df.isnull().sum()

    missing = missing[missing > 0].sort_values(
        ascending=False
    )

    if missing.empty:

        print("No missing values found.")

    else:

        print(
            f"Columns containing missing values: "
            f"{len(missing)}"
        )

        print("\nTop 15 columns by missing values:")

        for column, count in missing.head(15).items():

            percentage = (count / len(df)) * 100

            print(
                f"{column:<25} "
                f"{count:>10,} "
                f"({percentage:>6.2f}%)"
            )

    # --------------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------------

    print_section(f"{dataset_name} DUPLICATE ROWS")

    duplicate_rows = df.duplicated().sum()

    print(f"Duplicate rows: {duplicate_rows:,}")

    return df


# ============================================================
# Transaction Dataset Analysis
# ============================================================

def analyze_transaction_dataset(
    df: pd.DataFrame
) -> None:

    print_section("TRANSACTION DATASET ANALYSIS")

    # --------------------------------------------------------
    # TransactionID validation
    # --------------------------------------------------------

    if "TransactionID" in df.columns:

        duplicate_ids = (
            df["TransactionID"]
            .duplicated()
            .sum()
        )

        print(
            f"TransactionID duplicates: "
            f"{duplicate_ids:,}"
        )

    else:

        print("TransactionID column not found.")

    # --------------------------------------------------------
    # Fraud target validation
    # --------------------------------------------------------

    if "isFraud" not in df.columns:

        print("isFraud column not found.")

        return

    print_section("FRAUD CLASS DISTRIBUTION")

    fraud_counts = df["isFraud"].value_counts()

    fraud_percentages = (
        df["isFraud"]
        .value_counts(normalize=True)
        .mul(100)
    )

    for class_value in sorted(
        fraud_counts.index
    ):

        count = fraud_counts[class_value]

        percentage = fraud_percentages[
            class_value
        ]

        if class_value == 0:

            label = "Legitimate"

        elif class_value == 1:

            label = "Fraud"

        else:

            label = f"Class {class_value}"

        print(
            f"{label:<15} "
            f"{count:>10,} "
            f"({percentage:>6.3f}%)"
        )

    # --------------------------------------------------------
    # Fraud rate
    # --------------------------------------------------------

    fraud_count = (
        df["isFraud"] == 1
    ).sum()

    total_transactions = len(df)

    fraud_rate = (
        fraud_count /
        total_transactions
    ) * 100

    print("\nFraud Rate:")

    print(f"{fraud_rate:.4f}%")

    # --------------------------------------------------------
    # Target validation
    # --------------------------------------------------------

    print("\nTarget Validation:")

    unique_values = sorted(
        df["isFraud"]
        .dropna()
        .unique()
    )

    print(
        f"Unique values: "
        f"{unique_values}"
    )

    if set(unique_values).issubset({0, 1}):

        print(
            "Target contains only 0 and 1."
        )

    else:

        print(
            "WARNING: Unexpected target "
            "values detected."
        )


# ============================================================
# Transaction ↔ Identity Coverage Analysis
# ============================================================

def analyze_identity_coverage(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame
) -> None:

    print_section(
        "TRANSACTION ↔ IDENTITY COVERAGE"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    if "TransactionID" not in transaction_df.columns:

        print(
            "TransactionID missing from "
            "transaction dataset."
        )

        return

    if "TransactionID" not in identity_df.columns:

        print(
            "TransactionID missing from "
            "identity dataset."
        )

        return

    # --------------------------------------------------------
    # Transaction IDs
    # --------------------------------------------------------

    transaction_ids = (
        transaction_df["TransactionID"]
    )

    # --------------------------------------------------------
    # Identity IDs
    # --------------------------------------------------------

    identity_ids = (
        identity_df["TransactionID"]
    )

    # --------------------------------------------------------
    # Check identity availability
    # --------------------------------------------------------

    has_identity = (
        transaction_ids.isin(identity_ids)
    )

    identity_count = has_identity.sum()

    no_identity_count = (
        ~has_identity
    ).sum()

    total_transactions = (
        len(transaction_df)
    )

    # --------------------------------------------------------
    # Coverage percentages
    # --------------------------------------------------------

    identity_percentage = (
        identity_count /
        total_transactions
    ) * 100

    no_identity_percentage = (
        no_identity_count /
        total_transactions
    ) * 100

    # --------------------------------------------------------
    # Display coverage
    # --------------------------------------------------------

    print(
        f"Total transactions:       "
        f"{total_transactions:,}"
    )

    print(
        f"With identity data:       "
        f"{identity_count:,}"
    )

    print(
        f"Without identity data:    "
        f"{no_identity_count:,}"
    )

    print(
        f"\nIdentity coverage:        "
        f"{identity_percentage:.2f}%"
    )

    print(
        f"No identity coverage:     "
        f"{no_identity_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # Orphan identity records
    # --------------------------------------------------------

    orphan_identity_ids = (
        ~identity_ids.isin(transaction_ids)
    ).sum()

    print(
        f"\nIdentity records without "
        f"matching transaction:     "
        f"{orphan_identity_ids:,}"
    )

    # --------------------------------------------------------
    # Integrity check
    # --------------------------------------------------------

    if orphan_identity_ids == 0:

        print(
            "\nIdentity integrity check: "
            "PASSED"
        )

    else:

        print(
            "\nIdentity integrity check: "
            "WARNING"
        )


# ============================================================
# Main
# ============================================================

def main():

    print_section(
        "FinGuard AI - Dataset Validation"
    )

    print("Project root:")

    print(PROJECT_ROOT)

    print("\nRaw dataset directory:")

    print(RAW_DATA_DIR)

    # --------------------------------------------------------
    # Transaction dataset
    # --------------------------------------------------------

    transaction_df = validate_file(
        TRANSACTION_FILE,
        "TRANSACTION DATASET"
    )

    # --------------------------------------------------------
    # Identity dataset
    # --------------------------------------------------------

    identity_df = validate_file(
        IDENTITY_FILE,
        "IDENTITY DATASET"
    )

    # --------------------------------------------------------
    # Transaction analysis
    # --------------------------------------------------------

    if transaction_df is not None:

        analyze_transaction_dataset(
            transaction_df
        )

    # --------------------------------------------------------
    # Identity coverage analysis
    # --------------------------------------------------------

    if (
        transaction_df is not None
        and identity_df is not None
    ):

        analyze_identity_coverage(
            transaction_df,
            identity_df
        )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print_section(
        "VALIDATION COMPLETE"
    )

    if (
        transaction_df is not None
        and identity_df is not None
    ):

        print(
            "Both datasets loaded successfully."
        )

        print(
            "Basic validation completed."
        )

    else:

        print(
            "Dataset validation failed."
        )

        print(
            "Check the dataset files and paths."
        )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()