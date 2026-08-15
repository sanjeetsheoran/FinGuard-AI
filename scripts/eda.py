from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "raw"
    / "train_transaction.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "eda"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42


# ============================================================
# Utility
# ============================================================

def print_section(title: str) -> None:

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Load Dataset
# ============================================================

def load_dataset() -> pd.DataFrame:

    print_section("LOADING DATASET")

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    print(f"Loading: {DATA_FILE.name}")

    df = pd.read_csv(DATA_FILE)

    print("Dataset loaded successfully.")

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    return df


# ============================================================
# Basic Statistics
# ============================================================

def basic_statistics(df: pd.DataFrame) -> None:

    print_section("BASIC STATISTICS")

    print("\nTransaction Amount Statistics:")

    print(
        df["TransactionAmt"]
        .describe()
        .round(2)
    )

    print("\nFraud Distribution:")

    print(
        df["isFraud"]
        .value_counts()
    )

    print("\nFraud Percentage:")

    print(
        df["isFraud"]
        .value_counts(normalize=True)
        .mul(100)
        .round(4)
    )


# ============================================================
# Fraud vs Transaction Amount
# ============================================================

def analyze_transaction_amount(
    df: pd.DataFrame
) -> None:

    print_section(
        "TRANSACTION AMOUNT ANALYSIS"
    )

    fraud_amount = (
        df.loc[
            df["isFraud"] == 1,
            "TransactionAmt"
        ]
    )

    legitimate_amount = (
        df.loc[
            df["isFraud"] == 0,
            "TransactionAmt"
        ]
    )

    print(
        f"Average legitimate transaction: "
        f"{legitimate_amount.mean():.2f}"
    )

    print(
        f"Average fraudulent transaction: "
        f"{fraud_amount.mean():.2f}"
    )

    print(
        f"Median legitimate transaction: "
        f"{legitimate_amount.median():.2f}"
    )

    print(
        f"Median fraudulent transaction: "
        f"{fraud_amount.median():.2f}"
    )

    # Box plot
    plt.figure(figsize=(10, 6))

    df.boxplot(
        column="TransactionAmt",
        by="isFraud"
    )

    plt.title(
        "Transaction Amount by Fraud Status"
    )

    plt.suptitle("")

    plt.xlabel("Fraud Status")

    plt.ylabel("Transaction Amount")

    output_file = (
        OUTPUT_DIR
        / "transaction_amount_by_fraud.png"
    )

    plt.savefig(
        output_file,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved: {output_file}"
    )


# ============================================================
# Product Category Analysis
# ============================================================

def analyze_product_category(
    df: pd.DataFrame
) -> None:

    print_section(
        "PRODUCT CATEGORY ANALYSIS"
    )

    if "ProductCD" not in df.columns:

        print("ProductCD not found.")

        return

    product_stats = (
        df.groupby("ProductCD")["isFraud"]
        .agg(
            transactions="count",
            frauds="sum",
            fraud_rate="mean"
        )
    )

    product_stats["fraud_rate"] *= 100

    product_stats = (
        product_stats
        .sort_values(
            "fraud_rate",
            ascending=False
        )
    )

    print(
        product_stats.round(3)
    )

    # Plot fraud rate
    plt.figure(figsize=(10, 6))

    product_stats["fraud_rate"].plot(
        kind="bar"
    )

    plt.title(
        "Fraud Rate by Product Category"
    )

    plt.xlabel("Product Category")

    plt.ylabel("Fraud Rate (%)")

    plt.xticks(rotation=0)

    output_file = (
        OUTPUT_DIR
        / "fraud_rate_by_product.png"
    )

    plt.savefig(
        output_file,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved: {output_file}"
    )


# ============================================================
# Transaction Time Analysis
# ============================================================

def analyze_transaction_time(
    df: pd.DataFrame
) -> None:

    print_section(
        "TRANSACTION TIME ANALYSIS"
    )

    if "TransactionDT" not in df.columns:

        print(
            "TransactionDT not found."
        )

        return

    # TransactionDT represents elapsed
    # time from a reference point.
    #
    # Convert into hours within the
    # relative transaction timeline.

    seconds_in_day = 24 * 60 * 60

    df = df.copy()

    df["TransactionHour"] = (
        df["TransactionDT"]
        % seconds_in_day
    ) / 3600

    df["TransactionHour"] = (
        df["TransactionHour"]
        .astype(int)
    )

    hourly_stats = (
        df.groupby("TransactionHour")[
            "isFraud"
        ]
        .agg(
            transactions="count",
            frauds="sum",
            fraud_rate="mean"
        )
    )

    hourly_stats["fraud_rate"] *= 100

    print(
        hourly_stats.round(3)
    )

    plt.figure(figsize=(12, 6))

    hourly_stats["fraud_rate"].plot()

    plt.title(
        "Fraud Rate by Transaction Hour"
    )

    plt.xlabel(
        "Relative Transaction Hour"
    )

    plt.ylabel(
        "Fraud Rate (%)"
    )

    plt.grid(True)

    output_file = (
        OUTPUT_DIR
        / "fraud_rate_by_hour.png"
    )

    plt.savefig(
        output_file,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved: {output_file}"
    )


# ============================================================
# Missing Value Analysis
# ============================================================

def analyze_missing_values(
    df: pd.DataFrame
) -> None:

    print_section(
        "MISSING VALUE ANALYSIS"
    )

    missing_percentage = (
        df.isnull()
        .mean()
        .mul(100)
        .sort_values(
            ascending=False
        )
    )

    print(
        "\nTop 20 features by missing percentage:"
    )

    print(
        missing_percentage
        .head(20)
        .round(2)
    )

    # Save report
    output_file = (
        OUTPUT_DIR
        / "missing_value_report.csv"
    )

    missing_percentage.to_csv(
        output_file,
        header=["missing_percentage"]
    )

    print(
        f"\nSaved: {output_file}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print_section(
        "FinGuard AI - Exploratory Data Analysis"
    )

    df = load_dataset()

    basic_statistics(df)

    analyze_transaction_amount(df)

    analyze_product_category(df)

    analyze_transaction_time(df)

    analyze_missing_values(df)

    print_section(
        "EDA COMPLETE"
    )

    print(
        f"EDA outputs saved to: {OUTPUT_DIR}"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()