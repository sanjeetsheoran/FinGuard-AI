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
    / "part3"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Configuration
# ============================================================

OVERALL_FRAUD_RATE = 3.499


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

    df = pd.read_csv(DATA_FILE)

    print("Dataset loaded successfully.")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    return df


# ============================================================
# Product × Card6 Analysis
# ============================================================

def analyze_product_card6(
    df: pd.DataFrame
) -> None:

    print_section(
        "PRODUCTCD × CARD6 ANALYSIS"
    )

    required = [
        "ProductCD",
        "card6",
        "isFraud"
    ]

    if not all(
        column in df.columns
        for column in required
    ):
        print("Required columns not found.")
        return

    stats = (
        df.groupby(
            ["ProductCD", "card6"],
            dropna=False
        )["isFraud"]
        .agg(
            transactions="count",
            frauds="sum",
            fraud_rate="mean"
        )
    )

    stats["fraud_rate"] *= 100

    stats["lift_vs_baseline"] = (
        stats["fraud_rate"]
        / OVERALL_FRAUD_RATE
    )

    stats = stats.sort_values(
        "fraud_rate",
        ascending=False
    )

    print(
        stats.round(3)
    )

    output_file = (
        OUTPUT_DIR
        / "product_card6_fraud_analysis.csv"
    )

    stats.to_csv(output_file)

    print(
        f"\nSaved: {output_file}"
    )


# ============================================================
# Product × Card4 Analysis
# ============================================================

def analyze_product_card4(
    df: pd.DataFrame
) -> None:

    print_section(
        "PRODUCTCD × CARD4 ANALYSIS"
    )

    required = [
        "ProductCD",
        "card4",
        "isFraud"
    ]

    if not all(
        column in df.columns
        for column in required
    ):
        print("Required columns not found.")
        return

    stats = (
        df.groupby(
            ["ProductCD", "card4"],
            dropna=False
        )["isFraud"]
        .agg(
            transactions="count",
            frauds="sum",
            fraud_rate="mean"
        )
    )

    stats["fraud_rate"] *= 100

    stats["lift_vs_baseline"] = (
        stats["fraud_rate"]
        / OVERALL_FRAUD_RATE
    )

    stats = stats.sort_values(
        "fraud_rate",
        ascending=False
    )

    print(
        stats.round(3)
    )

    output_file = (
        OUTPUT_DIR
        / "product_card4_fraud_analysis.csv"
    )

    stats.to_csv(output_file)

    print(
        f"\nSaved: {output_file}"
    )


# ============================================================
# Product × Email Analysis
# ============================================================

def analyze_product_email(
    df: pd.DataFrame
) -> None:

    print_section(
        "PRODUCTCD × EMAIL DOMAIN ANALYSIS"
    )

    required = [
        "ProductCD",
        "P_emaildomain",
        "isFraud"
    ]

    if not all(
        column in df.columns
        for column in required
    ):
        print("Required columns not found.")
        return

    stats = (
        df.groupby(
            [
                "ProductCD",
                "P_emaildomain"
            ],
            dropna=False
        )["isFraud"]
        .agg(
            transactions="count",
            frauds="sum",
            fraud_rate="mean"
        )
    )

    stats["fraud_rate"] *= 100

    # Ignore tiny groups.
    stats = stats[
        stats["transactions"] >= 100
    ]

    stats["lift_vs_baseline"] = (
        stats["fraud_rate"]
        / OVERALL_FRAUD_RATE
    )

    stats = stats.sort_values(
        "fraud_rate",
        ascending=False
    )

    print(
        "\nTop 30 Product × Email combinations:"
    )

    print(
        stats.head(30).round(3)
    )

    output_file = (
        OUTPUT_DIR
        / "product_email_fraud_analysis.csv"
    )

    stats.to_csv(output_file)

    print(
        f"\nSaved: {output_file}"
    )


# ============================================================
# Transaction Amount Binning
# ============================================================

def analyze_amount_buckets(
    df: pd.DataFrame
) -> None:

    print_section(
        "TRANSACTION AMOUNT BUCKET ANALYSIS"
    )

    if "TransactionAmt" not in df.columns:
        print("TransactionAmt not found.")
        return

    bins = [
        0,
        25,
        50,
        100,
        250,
        500,
        1000,
        5000,
        float("inf")
    ]

    labels = [
        "0-25",
        "25-50",
        "50-100",
        "100-250",
        "250-500",
        "500-1000",
        "1000-5000",
        "5000+"
    ]

    df = df.copy()

    df["AmountBucket"] = pd.cut(
        df["TransactionAmt"],
        bins=bins,
        labels=labels,
        right=False
    )

    stats = (
        df.groupby(
            "AmountBucket",
            observed=False
        )["isFraud"]
        .agg(
            transactions="count",
            frauds="sum",
            fraud_rate="mean"
        )
    )

    stats["fraud_rate"] *= 100

    stats["fraud_share"] = (
        stats["frauds"]
        / stats["frauds"].sum()
    ) * 100

    print(
        stats.round(3)
    )

    output_file = (
        OUTPUT_DIR
        / "transaction_amount_buckets.csv"
    )

    stats.to_csv(output_file)

    print(
        f"\nSaved: {output_file}"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    stats["fraud_rate"].plot(
        kind="bar"
    )

    plt.axhline(
        OVERALL_FRAUD_RATE,
        linestyle="--",
        label="Overall Fraud Rate"
    )

    plt.title(
        "Fraud Rate by Transaction Amount"
    )

    plt.xlabel(
        "Transaction Amount Bucket"
    )

    plt.ylabel(
        "Fraud Rate (%)"
    )

    plt.legend()

    plt.xticks(rotation=45)

    plt.tight_layout()

    plot_file = (
        OUTPUT_DIR
        / "fraud_rate_by_amount_bucket.png"
    )

    plt.savefig(
        plot_file,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {plot_file}"
    )


# ============================================================
# Fraud Concentration Analysis
# ============================================================

def analyze_fraud_concentration(
    df: pd.DataFrame
) -> None:

    print_section(
        "FRAUD CONCENTRATION ANALYSIS"
    )

    if "ProductCD" not in df.columns:
        print("ProductCD not found.")
        return

    stats = (
        df.groupby("ProductCD")["isFraud"]
        .agg(
            transactions="count",
            frauds="sum"
        )
    )

    total_frauds = stats["frauds"].sum()

    stats["fraud_share"] = (
        stats["frauds"]
        / total_frauds
    ) * 100

    stats = stats.sort_values(
        "frauds",
        ascending=False
    )

    print(
        stats.round(3)
    )

    # --------------------------------------------------------
    # Cumulative fraud share
    # --------------------------------------------------------

    stats["cumulative_fraud_share"] = (
        stats["fraud_share"]
        .cumsum()
    )

    print(
        "\nCumulative fraud contribution:"
    )

    print(
        stats[
            [
                "frauds",
                "fraud_share",
                "cumulative_fraud_share"
            ]
        ].round(3)
    )

    output_file = (
        OUTPUT_DIR
        / "fraud_concentration_product.csv"
    )

    stats.to_csv(output_file)

    print(
        f"\nSaved: {output_file}"
    )


# ============================================================
# Missingness Pattern Analysis
# ============================================================

def analyze_missingness_patterns(
    df: pd.DataFrame
) -> None:

    print_section(
        "MISSINGNESS PATTERN ANALYSIS"
    )

    selected_features = [
        "D6",
        "D7",
        "D8",
        "D9",
        "D12",
        "D13",
        "D14",
        "dist1",
        "dist2"
    ]

    available_features = [
        feature
        for feature in selected_features
        if feature in df.columns
    ]

    if not available_features:
        print("No selected features available.")
        return

    # Number of missing selected features
    df = df.copy()

    df["missing_selected_count"] = (
        df[available_features]
        .isna()
        .sum(axis=1)
    )

    stats = (
        df.groupby(
            "missing_selected_count"
        )["isFraud"]
        .agg(
            transactions="count",
            frauds="sum",
            fraud_rate="mean"
        )
    )

    stats["fraud_rate"] *= 100

    stats["fraud_share"] = (
        stats["frauds"]
        / stats["frauds"].sum()
    ) * 100

    print(
        stats.round(3)
    )

    output_file = (
        OUTPUT_DIR
        / "missingness_pattern_analysis.csv"
    )

    stats.to_csv(output_file)

    print(
        f"\nSaved: {output_file}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print_section(
        "FinGuard AI - EDA Part 3"
    )

    df = load_dataset()

    # --------------------------------------------------------
    # Feature interactions
    # --------------------------------------------------------

    analyze_product_card6(df)

    analyze_product_card4(df)

    analyze_product_email(df)

    # --------------------------------------------------------
    # Transaction amount
    # --------------------------------------------------------

    analyze_amount_buckets(df)

    # --------------------------------------------------------
    # Fraud concentration
    # --------------------------------------------------------

    analyze_fraud_concentration(df)

    # --------------------------------------------------------
    # Missingness patterns
    # --------------------------------------------------------

    analyze_missingness_patterns(df)

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print_section(
        "EDA PART 3 COMPLETE"
    )

    print(
        f"Outputs saved to: {OUTPUT_DIR}"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()