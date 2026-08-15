from __future__ import annotations

from typing import Iterable

import pandas as pd


def clean_text(value) -> str | None:
    """Normalize a cell value into clean text."""

    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def normalize_yes_no(value) -> str | None:
    """Normalize yes/no style fields."""

    text = clean_text(value)

    if text is None:
        return None

    text = text.lower()

    if text in {"yes", "y", "true", "1"}:
        return "yes"

    if text in {"no", "n", "false", "0"}:
        return "no"

    return text


def safe_percentage(
    numerator: int,
    denominator: int,
) -> float:
    """Calculate a percentage without division-by-zero errors."""

    if denominator == 0:
        return 0.0

    return round((numerator / denominator) * 100, 1)


def top_counts(
    series: pd.Series,
    limit: int = 10,
) -> list[dict]:
    """
    Return the most frequent non-empty values.

    Output format:
    [
        {"value": "...", "count": 10, "percentage": 1.2},
        ...
    ]
    """

    cleaned = series.map(clean_text).dropna()

    counts = cleaned.value_counts().head(limit)

    total = len(cleaned)

    results = []

    for value, count in counts.items():
        results.append(
            {
                "value": value,
                "count": int(count),
                "percentage": safe_percentage(
                    int(count),
                    total,
                ),
            }
        )

    return results


def unique_non_empty(
    series: pd.Series,
) -> list[str]:
    """Return sorted unique non-empty values."""

    values = (
        series
        .map(clean_text)
        .dropna()
        .unique()
        .tolist()
    )

    return sorted(values)