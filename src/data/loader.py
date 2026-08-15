from pathlib import Path

import pandas as pd


# Project root:
# .../sri_sai_arun_genar_challenge/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data" / "Bisoprolol_icsr_sample_1068rows.xlsx"


def load_dataset() -> pd.DataFrame:
    """
    Load the supplied Bisoprolol ICSR dataset.

    Returns:
        pandas.DataFrame: Raw dataset exactly as supplied.
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_FILE}"
        )

    df = pd.read_excel(DATA_FILE)

    if df.empty:
        raise ValueError("Dataset was loaded but contains no rows.")

    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    """Print basic information about the dataset."""

    print("\n" + "=" * 70)
    print("GENAR DATASET INSPECTION")
    print("=" * 70)

    print(f"\nDataset path:")
    print(DATA_FILE)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nColumn names:")
    for index, column in enumerate(df.columns, start=1):
        print(f"{index:>3}. {column}")

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("No missing values found.")
    else:
        for column, count in missing.items():
            percentage = (count / len(df)) * 100
            print(
                f"{column}: {count:,} "
                f"({percentage:.1f}%)"
            )

    if "safetyreportid" in df.columns:
        unique_cases = df["safetyreportid"].nunique()

        print("\nCase identity:")
        print(f"Rows: {len(df):,}")
        print(f"Unique safetyreportid values: {unique_cases:,}")
        print(
            f"Rows per unique case: "
            f"{len(df) / unique_cases:.2f}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    dataset = load_dataset()
    inspect_dataset(dataset)