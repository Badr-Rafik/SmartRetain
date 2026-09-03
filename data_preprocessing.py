from pathlib import Path

import pandas as pd

TARGET_COLUMN = "Exited"


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load the CSV file and tidy up the column names."""
    churn_df = pd.read_csv(file_path)
    churn_df.columns = churn_df.columns.str.strip()
    return churn_df


def clean_data(churn_df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and fill missing values with simple sensible defaults."""
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
    """Limit extreme feature values without changing the target or ID columns."""
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
    """Add a few ratios that describe a customer's banking situation."""
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
    """Run cleaning and feature engineering, then separate X and y."""
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


def main() -> None:
    """Run the preprocessing and EDA steps by themselves."""
    dataset_path = Path(__file__).resolve().parent / "Bank_Churn.csv"
    prepare_data(dataset_path)
    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()
