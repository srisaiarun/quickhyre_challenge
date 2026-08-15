from __future__ import annotations

import pandas as pd


def prepare_dates(cases: pd.DataFrame) -> pd.DataFrame:
    """Create normalized date fields for temporal analysis."""

    df = cases.copy()

    df["report_date"] = pd.to_datetime(
        df["report_date"],
        errors="coerce",
    )

    df = df.dropna(subset=["report_date"]).copy()

    df["month"] = df["report_date"].dt.to_period("M").astype(str)

    return df


def analyze_monthly_trends(
    cases: pd.DataFrame,
) -> list[dict]:
    """Calculate case volume by reporting month."""

    df = prepare_dates(cases)

    if df.empty:
        return []

    monthly = (
        df.groupby("month")
        .agg(
            cases=("safetyreportid", "nunique"),
            serious_cases=(
                "serious",
                lambda x: (
                    x.astype(str)
                    .str.lower()
                    .eq("serious")
                    .sum()
                ),
            ),
        )
        .reset_index()
    )

    monthly["serious_percentage"] = (
        monthly["serious_cases"]
        / monthly["cases"]
        * 100
    ).round(1)

    return [
        {
            "month": row["month"],
            "cases": int(row["cases"]),
            "serious_cases": int(row["serious_cases"]),
            "serious_percentage": float(
                row["serious_percentage"]
            ),
        }
        for _, row in monthly.iterrows()
    ]


def analyze_15_day_windows(
    cases: pd.DataFrame,
) -> list[dict]:
    """
    Calculate rolling 15-day case-volume windows.

    Each calendar day is used as a window start.
    The window includes the start date and the following
    14 calendar days.

    This is descriptive monitoring only and does not
    constitute a safety-signal determination.
    """

    df = prepare_dates(cases)

    if df.empty:
        return []

    min_date = df["report_date"].min().normalize()
    max_date = df["report_date"].max().normalize()

    results = []

    current = min_date

    while current <= max_date:

        window_end = min(
            current + pd.Timedelta(days=14),
            max_date,
        )

        window = df[
            (df["report_date"] >= current)
            & (df["report_date"] <= window_end)
        ]

        case_count = window[
            "safetyreportid"
        ].nunique()

        serious_count = (
            window["serious"]
            .astype(str)
            .str.lower()
            .eq("serious")
            .sum()
        )

        results.append(
            {
                "start_date": current.date().isoformat(),
                "end_date": window_end.date().isoformat(),
                "cases": int(case_count),
                "serious_cases": int(serious_count),
            }
        )

        current += pd.Timedelta(days=1)

    return results


def detect_15_day_spikes(
    windows: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """
    Identify the highest-volume rolling 15-day windows.

    This is a descriptive ranking, not a safety-signal
    determination.

    The purpose is to direct human review toward periods
    with relatively high observed case volume.
    """

    if not windows:
        return []

    ranked = sorted(
        windows,
        key=lambda item: (
            item["cases"],
            item["serious_cases"],
        ),
        reverse=True,
    )

    results = []

    for window in ranked[:top_n]:

        results.append(
            {
                **window,
                "observation": (
                    "High observed case volume relative "
                    "to other 15-day windows; requires "
                    "human review and contextual assessment."
                ),
            }
        )

    return results