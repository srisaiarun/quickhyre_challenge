from __future__ import annotations

import pandas as pd

from analysis.analysis_utils import (
    normalize_yes_no,
    safe_percentage,
    top_counts,
)


def analyze_case_volume(
    cases: pd.DataFrame,
) -> dict:
    """Calculate case-level volume and seriousness."""

    total_cases = len(cases)

    serious = (
        cases["serious"]
        .map(normalize_yes_no)
    )

    serious_cases = int(
        (serious == "serious").sum()
    )

    non_serious_cases = int(
        (serious == "not serious").sum()
    )

    expedited = (
        cases["fulfillexpeditecriteria"]
        .map(normalize_yes_no)
    )

    expedited_cases = int(
        (expedited == "yes").sum()
    )

    return {
        "total_cases": total_cases,
        "serious_cases": serious_cases,
        "serious_percentage": safe_percentage(
            serious_cases,
            total_cases,
        ),
        "non_serious_cases": non_serious_cases,
        "non_serious_percentage": safe_percentage(
            non_serious_cases,
            total_cases,
        ),
        "expedited_cases": expedited_cases,
        "expedited_percentage": safe_percentage(
            expedited_cases,
            total_cases,
        ),
    }


def analyze_demographics(
    cases: pd.DataFrame,
) -> dict:
    """Analyze age, sex, and country."""

    age = pd.to_numeric(
        cases["patient_patientonsetage"],
        errors="coerce",
    )

    age_groups = pd.cut(
        age,
        bins=[
            -float("inf"),
            17,
            44,
            64,
            74,
            float("inf"),
        ],
        labels=[
            "<18",
            "18-44",
            "45-64",
            "65-74",
            "75+",
        ],
    )

    age_counts = (
        age_groups
        .value_counts(sort=False)
    )

    age_results = []

    for group, count in age_counts.items():
        age_results.append(
            {
                "age_group": str(group),
                "count": int(count),
                "percentage": safe_percentage(
                    int(count),
                    int(age.notna().sum()),
                ),
            }
        )

    return {
        "age_groups": age_results,
        "sex": top_counts(
            cases["patient_patientsex"]
        ),
        "country": top_counts(
            cases["occurcountry"],
            limit=15,
        ),
    }


