from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "safetyreportid",
    "safetyreportversion",
    "serious",
    "fulfillexpeditecriteria",
    "receivedate",
    "occurcountry",
    "patient_patientonsetage",
    "patient_patientonsetageunit",
    "patient_patientsex",
    "patient_reaction_reactionmeddrapt",
    "patient_reaction_reactionoutcome",
]


def validate_required_columns(df: pd.DataFrame) -> None:
    """
    Verify that all fields required for Version 0 analysis exist.
    """

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing)
        )


def normalize_cases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the raw line listing into one canonical row per case.

    A case is identified by safetyreportid.
    If multiple versions exist, the highest
    safetyreportversion is treated as the latest case version.
    """

    validate_required_columns(df)

    working = df.copy()

    # Make sure version is numeric.
    working["safetyreportversion"] = pd.to_numeric(
        working["safetyreportversion"],
        errors="coerce",
    )

    if working["safetyreportversion"].isna().any():
        raise ValueError(
            "Some rows have an invalid safetyreportversion."
        )

    # Sort so the latest version of every case is last.
    working = working.sort_values(
        ["safetyreportid", "safetyreportversion"]
    )

    # Keep exactly one canonical/latest record per case.
    cases = (
        working
        .groupby("safetyreportid", as_index=False)
        .tail(1)
        .copy()
    )

    # Sort back into reporting-date order.
    cases["report_date"] = pd.to_datetime(
        cases["report_date"],
        errors="coerce",
    )

    cases = cases.sort_values(
        ["report_date", "safetyreportid"]
    ).reset_index(drop=True)

    return cases


def print_case_summary(
    raw_df: pd.DataFrame,
    cases_df: pd.DataFrame,
) -> None:
    """Print normalization results for verification."""

    print("\n" + "=" * 70)
    print("CASE NORMALIZATION")
    print("=" * 70)

    print(f"Raw rows: {len(raw_df):,}")
    print(f"Unique raw case IDs: {raw_df['safetyreportid'].nunique():,}")
    print(f"Canonical cases: {len(cases_df):,}")

    if len(cases_df) != raw_df["safetyreportid"].nunique():
        raise AssertionError(
            "Canonical case count does not match "
            "unique safetyreportid count."
        )

    serious_counts = cases_df["serious"].value_counts(
        dropna=False
    )

    expedited_counts = cases_df[
        "fulfillexpeditecriteria"
    ].value_counts(dropna=False)

    print("\nSeriousness:")
    print(serious_counts.to_string())

    print("\nExpedited criteria:")
    print(expedited_counts.to_string())

    print("\nReporting period:")

    valid_dates = cases_df["report_date"].dropna()

    if not valid_dates.empty:
        print(f"Start: {valid_dates.min().date()}")
        print(f"End:   {valid_dates.max().date()}")

    print("\n" + "=" * 70)