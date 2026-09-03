from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TARGET_COLUMN = "Exited"


def load_data(file_path: str | Path) -> pd.DataFrame:
    churn_df = pd.read_csv(file_path)
    churn_df.columns = churn_df.columns.str.strip()
    return churn_df


def clean_data(churn_df: pd.DataFrame) -> pd.DataFrame:
    churn_df = churn_df.drop_duplicates().reset_index(drop=True).copy()

    for column in churn_df.columns:
        if churn_df[column].isna().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(churn_df[column]):
            churn_df[column] = churn_df[column].fillna(churn_df[column].median())
        else:
            churn_df[column] = churn_df[column].fillna(churn_df[column].mode()[0])

    return churn_df


def cap_outliers(churn_df: pd.DataFrame) -> pd.DataFrame:
    churn_df = churn_df.copy()
    ignored_columns = {TARGET_COLUMN, "CustomerId"}
    numeric_columns = churn_df.select_dtypes(include="number").columns

    for column in numeric_columns:
        if column in ignored_columns:
            continue

        first_quartile = churn_df[column].quantile(0.25)
        third_quartile = churn_df[column].quantile(0.75)
        interquartile_range = third_quartile - first_quartile
        lower_limit = first_quartile - 1.5 * interquartile_range
        upper_limit = third_quartile + 1.5 * interquartile_range
        churn_df[column] = churn_df[column].clip(lower_limit, upper_limit)

    return churn_df


def show_eda(churn_df: pd.DataFrame) -> None:
    """Print a few useful checks before training a model."""
    print("\n=== Step 3: Exploratory Data Analysis ===")
    print(f"Rows and columns: {churn_df.shape}")
    print("\nDescriptive summary:")
    print(churn_df.describe(include="all").transpose())
    print("\nChurn distribution:")
    print(churn_df[TARGET_COLUMN].value_counts())
    print("\nChurn percentage:")
    print((churn_df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2))
    print("\nNumeric correlations:")
    print(churn_df.select_dtypes(include="number").corr().round(2))


def add_features(churn_df: pd.DataFrame) -> pd.DataFrame:
    churn_df = churn_df.copy()

    churn_df["BalanceToSalaryRatio"] = churn_df["Balance"] / (
        churn_df["EstimatedSalary"] + 1e-6
    )
    churn_df["CreditScoreToAgeRatio"] = churn_df["CreditScore"] / (
        churn_df["Age"] + 1e-6
    )
    churn_df["ProductDensity"] = churn_df["NumOfProducts"] / (churn_df["Tenure"] + 1)
    return churn_df


def prepare_data(file_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    print("=== Step 1: Loading Data ===")
    churn_df = load_data(file_path)
    print(f"Loaded {len(churn_df)} customer records.")

    print("\n=== Step 2: Data Cleaning ===")
    churn_df = clean_data(churn_df)
    churn_df = cap_outliers(churn_df)
    print(f"Cleaned data shape: {churn_df.shape}")

    show_eda(churn_df)

    print("\n=== Step 4: Feature Engineering ===")
    churn_df = add_features(churn_df)
    churn_target = churn_df.pop(TARGET_COLUMN).astype(int)

    churn_df = churn_df.drop(columns=["CustomerId", "Surname"], errors="ignore")
    return churn_df, churn_target


def plot_histograms(churn_df: pd.DataFrame, output_dir: str | Path | None = None) -> None:
    preferred_features = [
        column
        for column in [
            "Age",
            "Balance",
            "CreditScore",
            "EstimatedSalary",
            "Tenure",
            "NumOfProducts",
            "BalanceToSalaryRatio",
            "CreditScoreToAgeRatio",
            "ProductDensity",
        ]
        if column in churn_df.columns
    ]
    numeric_features = churn_df.select_dtypes(include="number").columns.tolist()
    selected_features = preferred_features or numeric_features[:6]

    output_path = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / "eda_visuals"
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(selected_features), 1, figsize=(12, 4 * len(selected_features)))
    if len(selected_features) == 1:
        axes = [axes]

    for axis, feature in zip(axes, selected_features):
        axis.hist(churn_df[feature], bins=30, color="#4C72B0", edgecolor="black")
        axis.set_title(f"Histogram of {feature}")
        axis.set_xlabel(feature)
        axis.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(output_path / "histograms.png", dpi=300)
    plt.close(fig)


def plot_target_histogram(churn_df: pd.DataFrame, output_dir: str | Path | None = None) -> None:
    output_path = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / "eda_visuals"
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(8, 5))
    counts = churn_df[TARGET_COLUMN].value_counts().sort_index()
    axis.bar(
        ["No Churn", "Churn"],
        counts.values,
        color=["#4C72B0", "#C44E52"],
        edgecolor="black",
    )
    axis.set_title("Histogram of Churn Target (Exited)")
    axis.set_xlabel("Customer Outcome")
    axis.set_ylabel("Number of Customers")
    for value, count in zip(["No Churn", "Churn"], counts.values):
        axis.text(value, count + 50, str(count), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path / "target_histogram.png", dpi=300)
    plt.close(fig)


def plot_correlation_heatmap(churn_df: pd.DataFrame, output_dir: str | Path | None = None) -> None:
    numeric_df = churn_df.select_dtypes(include="number")
    if numeric_df.empty:
        return

    output_path = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / "eda_visuals"
    output_path.mkdir(parents=True, exist_ok=True)

    correlation_matrix = numeric_df.corr().round(2)
    fig, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        fmt=".2f",
        ax=axis,
    )
    axis.set_title("Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(output_path / "correlation_heatmap.png", dpi=300)
    plt.close(fig)


def main() -> None:
    dataset_path = Path(__file__).resolve().parent / "Bank_Churn.csv"

    churn_df = load_data(dataset_path)
    churn_df = clean_data(churn_df)
    churn_df = cap_outliers(churn_df)
    churn_df = add_features(churn_df)

    plot_histograms(churn_df)
    plot_target_histogram(churn_df)
    plot_correlation_heatmap(churn_df)
    print("\nPreprocessing completed successfully.")
    print("EDA visualizations saved in the 'eda_visuals' folder.")


if __name__ == "__main__":
    main()
