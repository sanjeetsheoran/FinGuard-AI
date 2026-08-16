"""
FinGuard AI - Identity Dataset Integration Validation

Day 11

Purpose:
    Validate safe integration of the IEEE-CIS identity dataset
    with the transaction dataset using TransactionID.

Important:
    - Transaction dataset remains the primary dataset.
    - LEFT JOIN is used.
    - No target information is taken from identity data.
    - Duplicate TransactionIDs are checked before merging.
    - Row count must remain unchanged after integration.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# Project Setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
# Utility
# ============================================================

def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Main Validation
# ============================================================

def main():

    section(
        "FinGuard AI - Identity Integration Validation"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not TRANSACTION_FILE.exists():
        raise FileNotFoundError(
            f"Transaction dataset not found:\n"
            f"{TRANSACTION_FILE}"
        )

    if not IDENTITY_FILE.exists():
        raise FileNotFoundError(
            f"Identity dataset not found:\n"
            f"{IDENTITY_FILE}"
        )

    # --------------------------------------------------------
    # Load Transaction Dataset
    # --------------------------------------------------------

    section(
        "LOADING TRANSACTION DATASET"
    )

    transaction_df = pd.read_csv(
        TRANSACTION_FILE
    )

    print(
        f"Transaction rows   : "
        f"{len(transaction_df):,}"
    )

    print(
        f"Transaction columns: "
        f"{len(transaction_df.columns):,}"
    )

    # --------------------------------------------------------
    # Load Identity Dataset
    # --------------------------------------------------------

    section(
        "LOADING IDENTITY DATASET"
    )

    identity_df = pd.read_csv(
        IDENTITY_FILE
    )

    print(
        f"Identity rows   : "
        f"{len(identity_df):,}"
    )

    print(
        f"Identity columns: "
        f"{len(identity_df.columns):,}"
    )

    # --------------------------------------------------------
    # Verify TransactionID exists
    # --------------------------------------------------------

    section(
        "VERIFYING JOIN KEY"
    )

    if "TransactionID" not in transaction_df.columns:
        raise ValueError(
            "TransactionID missing from transaction dataset."
        )

    if "TransactionID" not in identity_df.columns:
        raise ValueError(
            "TransactionID missing from identity dataset."
        )

    print(
        "TransactionID exists in both datasets."
    )

    # --------------------------------------------------------
    # Check duplicate TransactionIDs
    # --------------------------------------------------------

    section(
        "CHECKING IDENTITY KEY UNIQUENESS"
    )

    transaction_duplicates = (
        transaction_df["TransactionID"]
        .duplicated()
        .sum()
    )

    identity_duplicates = (
        identity_df["TransactionID"]
        .duplicated()
        .sum()
    )

    print(
        f"Transaction duplicate IDs: "
        f"{transaction_duplicates:,}"
    )

    print(
        f"Identity duplicate IDs   : "
        f"{identity_duplicates:,}"
    )

    if transaction_duplicates != 0:
        raise ValueError(
            "Duplicate TransactionIDs found "
            "in transaction dataset."
        )

    if identity_duplicates != 0:
        raise ValueError(
            "Duplicate TransactionIDs found "
            "in identity dataset."
        )

    print(
        "TransactionID uniqueness: PASSED"
    )

    # --------------------------------------------------------
    # Calculate Identity Coverage
    # --------------------------------------------------------

    section(
        "IDENTITY COVERAGE"
    )

    transaction_ids = set(
        transaction_df["TransactionID"]
    )

    identity_ids = set(
        identity_df["TransactionID"]
    )

    matched_ids = (
        transaction_ids
        & identity_ids
    )

    identity_only_ids = (
        identity_ids
        - transaction_ids
    )

    without_identity = (
        transaction_ids
        - identity_ids
    )

    print(
        f"Total transactions          : "
        f"{len(transaction_ids):,}"
    )

    print(
        f"With identity data           : "
        f"{len(matched_ids):,}"
    )

    print(
        f"Without identity data        : "
        f"{len(without_identity):,}"
    )

    print(
        f"Identity records without "
        f"transaction                : "
        f"{len(identity_only_ids):,}"
    )

    coverage = (
        len(matched_ids)
        / len(transaction_ids)
        * 100
    )

    no_coverage = (
        len(without_identity)
        / len(transaction_ids)
        * 100
    )

    print(
        f"\nIdentity coverage            : "
        f"{coverage:.2f}%"
    )

    print(
        f"No identity coverage        : "
        f"{no_coverage:.2f}%"
    )

    if len(identity_only_ids) != 0:
        raise ValueError(
            "Identity dataset contains TransactionIDs "
            "not present in transaction dataset."
        )

    # --------------------------------------------------------
    # Perform LEFT JOIN
    # --------------------------------------------------------

    section(
        "PERFORMING LEFT JOIN"
    )

    merged_df = transaction_df.merge(
        identity_df,
        on="TransactionID",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_identity",
        ),
    )

    print(
        f"Original transaction rows : "
        f"{len(transaction_df):,}"
    )

    print(
        f"Merged rows               : "
        f"{len(merged_df):,}"
    )

    # --------------------------------------------------------
    # Row Count Integrity
    # --------------------------------------------------------

    section(
        "MERGE INTEGRITY"
    )

    if len(merged_df) != len(
        transaction_df
    ):
        raise ValueError(
            "LEFT JOIN changed transaction row count."
        )

    print(
        "Row count preservation: PASSED"
    )

    # --------------------------------------------------------
    # TransactionID Integrity
    # --------------------------------------------------------

    merged_duplicate_ids = (
        merged_df["TransactionID"]
        .duplicated()
        .sum()
    )

    print(
        f"Merged duplicate IDs: "
        f"{merged_duplicate_ids:,}"
    )

    if merged_duplicate_ids != 0:
        raise ValueError(
            "Duplicate TransactionIDs detected "
            "after merge."
        )

    print(
        "TransactionID integrity: PASSED"
    )

    # --------------------------------------------------------
    # Identity Column Check
    # --------------------------------------------------------

    identity_columns = [
        column
        for column in identity_df.columns
        if column != "TransactionID"
    ]

    missing_identity_columns = [
        column
        for column in identity_columns
        if column not in merged_df.columns
    ]

    if missing_identity_columns:
        raise ValueError(
            "Identity columns missing after merge: "
            f"{missing_identity_columns}"
        )

    print(
        "Identity feature preservation: PASSED"
    )

    # --------------------------------------------------------
    # Identity Availability Check
    # --------------------------------------------------------

    if identity_columns:

        identity_available = (
            merged_df[identity_columns]
            .notna()
            .any(axis=1)
        )

        available_count = int(
            identity_available.sum()
        )

        unavailable_count = int(
            (~identity_available).sum()
        )

        print(
            f"\nRows with identity features : "
            f"{available_count:,}"
        )

        print(
            f"Rows without identity data  : "
            f"{unavailable_count:,}"
        )

    # --------------------------------------------------------
    # Target Integrity
    # --------------------------------------------------------

    section(
        "TARGET INTEGRITY"
    )

    if "isFraud" not in transaction_df.columns:
        raise ValueError(
            "isFraud target missing."
        )

    if "isFraud" not in merged_df.columns:
        raise ValueError(
            "isFraud target missing after merge."
        )

    if not merged_df["isFraud"].equals(
        transaction_df["isFraud"]
    ):
        raise ValueError(
            "Target changed during identity merge."
        )

    print(
        "Target preservation: PASSED"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    section(
        "IDENTITY INTEGRATION SUMMARY"
    )

    print(
        f"Transaction rows : "
        f"{len(transaction_df):,}"
    )

    print(
        f"Identity rows    : "
        f"{len(identity_df):,}"
    )

    print(
        f"Identity coverage: "
        f"{coverage:.2f}%"
    )

    print(
        f"Merged rows      : "
        f"{len(merged_df):,}"
    )

    print(
        "\nJoin strategy: LEFT JOIN"
    )

    print(
        "Join key: TransactionID"
    )

    print(
        "\nIdentity integration validation: PASSED"
    )

    print(
        "Test set was NOT used."
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()