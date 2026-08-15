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
    / "part2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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
# Categorical Feature Analysis
# ============================================================

def analyze_categorical_feature(
    df: pd.DataFrame,
    feature: str
) -> None:

    if feature not in df.columns:
        print(f"Skipping {feature}: column not found.")
        return

    print_section(
        f"CATEGORICAL ANALYSIS: {feature}"
    )

    # Keep categories with at least 100 transactions
    # to avoid unstable rates from extremely small groups.

    stats = (
        df.groupby(feature, dropna=False)["isFraud"]
        .agg(
            transactions="count",
            frauds="sum",
            fraud_rate="mean"
        )
    )

    stats["fraud_rate"] *= 100

    stats = stats[
        stats["transactions"] >= 100
    ].sort_values(
        "fraud_rate",
        ascending=False
    )

    print(stats.round(3))

    # Save statistics
    output_file = (
        OUTPUT_DIR
        / f"{feature}_fraud_rate.csv"
    )

    stats.to_csv(output_file)

    # Plot
    plt.figure(figsize=(10, 6))

    stats["fraud_rate"].plot(
        kind="bar"
    )

    plt.title(
        f"Fraud Rate by {feature}"
    )

    plt.xlabel(feature)

    plt.ylabel("Fraud Rate (%)")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plot_file = (
        OUTPUT_DIR
        / f"{feature}_fraud_rate.png"
    )

    plt.savefig(
        plot_file,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nSaved: {output_file}")
    print(f"Saved: {plot_file}")


# ============================================================
# Selected Numerical Feature Analysis
# ============================================================

def analyze_numerical_feature(
    df: pd.DataFrame,
    feature: str
) -> None:

    if feature not in df.columns:
        print(f"Skipping {feature}: column not found.")
        return

    print_section(
        f"NUMERICAL ANALYSIS: {feature}"
    )

    stats = (
        df.groupby("isFraud")[feature]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max"
        )
    )

    print(
        stats.round(3)
    )

    # --------------------------------------------------------
    # Fraud vs legitimate boxplot
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    df.boxplot(
        column=feature,
        by="isFraud"
    )

    plt.title(
        f"{feature} by Fraud Status"
    )

    plt.suptitle("")

    plt.xlabel("Fraud Status")

    plt.ylabel(feature)

    plt.tight_layout()

    plot_file = (
        OUTPUT_DIR
        / f"{feature}_by_fraud.png"
    )

    plt.savefig(
        plot_file,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {plot_file}")


# ============================================================
# Missingness vs Fraud Analysis
# ============================================================

def analyze_missingness_signal(
    df: pd.DataFrame,
    features: list[str]
) -> None:

    print_section(
        "MISSINGNESS VS FRAUD ANALYSIS"
    )

    results = []

    for feature in features:

        if feature not in df.columns:
            continue

        missing_mask = df[feature].isna()

        missing_count = missing_mask.sum()

        present_count = (~missing_mask).sum()

        if missing_count == 0 or present_count == 0:
            continue

        fraud_rate_missing = (
            df.loc[
                missing_mask,
                "isFraud"
            ].mean() * 100
        )

        fraud_rate_present = (
            df.loc[
                ~missing_mask,
                "isFraud"
            ].mean() * 100
        )

        results.append(
            {
                "feature": feature,
                "missing_count": missing_count,
                "present_count": present_count,
                "fraud_rate_when_missing":
                    fraud_rate_missing,
                "fraud_rate_when_present":
                    fraud_rate_present
            }
        )

    if not results:
        print("No suitable features found.")
        return

    result_df = pd.DataFrame(results)

    result_df["absolute_difference"] = (
        result_df["fraud_rate_when_missing"]
        - result_df["fraud_rate_when_present"]
    ).abs()

    result_df = result_df.sort_values(
        "absolute_difference",
        ascending=False
    )

    print(
        result_df.head(20).round(3)
    )

    output_file = (
        OUTPUT_DIR
        / "missingness_vs_fraud.csv"
    )

    result_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved: {output_file}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print_section(
        "FinGuard AI - EDA Part 2"
    )

    df = load_dataset()

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    categorical_features = [
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "M6",
        "M7",
        "M8",
        "M9"
    ]

    print_section(
        "CATEGORICAL FEATURES"
    )

    for feature in categorical_features:

        analyze_categorical_feature(
            df,
            feature
        )

    # --------------------------------------------------------
    # Numerical features
    # --------------------------------------------------------

    numerical_features = [
        "TransactionAmt",
        "card1",
        "card2",
        "card3",
        "card5",
        "addr1",
        "addr2",
        "dist1"
    ]

    print_section(
        "NUMERICAL FEATURES"
    )

    for feature in numerical_features:

        analyze_numerical_feature(
            df,
            feature
        )

    # --------------------------------------------------------
    # Missingness signal
    # --------------------------------------------------------

    missingness_features = [
        "dist1",
        "dist2",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "D8",
        "D9",
        "D10",
        "D11",
        "D12",
        "D13",
        "D14",
        "D15"
    ]

    analyze_missingness_signal(
        df,
        missingness_features
    )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print_section(
        "EDA PART 2 COMPLETE"
    )

    print(
        f"Outputs saved to: {OUTPUT_DIR}"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()