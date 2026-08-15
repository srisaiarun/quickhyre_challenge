from __future__ import annotations

import pandas as pd

from analysis.analysis_utils import (
    clean_text,
    safe_percentage,
)


def split_reactions(value) -> list[str]:
    """
    Split a comma-separated MedDRA Preferred Term field.

    The dataset can contain multiple reactions for one case.
    """
    text = clean_text(value)

    if text is None:
        return []

    return [
        part.strip()
        for part in text.split(",")
        if part.strip()
    ]


def split_outcomes(value) -> list[str]:
    """
    Split a comma-separated reaction outcome field.

    Outcomes are positionally associated with reactions.
    """
    text = clean_text(value)

    if text is None:
        return []

    return [
        part.strip()
        for part in text.split(",")
        if part.strip()
    ]


def build_reaction_rows(
    cases: pd.DataFrame,
) -> pd.DataFrame:
    """
    Expand case-level data into reaction-level records.

    Each reaction gets the corresponding outcome by position.

    If the dataset contains fewer outcomes than reactions,
    the missing outcome is explicitly recorded as
    'unreported' rather than inferred.
    """

    records = []

    for _, row in cases.iterrows():

        reactions = split_reactions(
            row["patient_reaction_reactionmeddrapt"]
        )

        outcomes = split_outcomes(
            row["patient_reaction_reactionoutcome"]
        )

        for index, reaction in enumerate(reactions):

            if index < len(outcomes):
                outcome = outcomes[index]
            else:
                outcome = "unreported"

            records.append(
                {
                    "safetyreportid": row["safetyreportid"],
                    "reaction": reaction,
                    "outcome": outcome,
                    "serious": row["serious"],
                    "country": row["occurcountry"],
                    "sex": row["patient_patientsex"],
                    "age": row["patient_patientonsetage"],
                    "receivedate": row["receivedate"],
                }
            )

    return pd.DataFrame(records)


def analyze_reactions(
    cases: pd.DataFrame,
    limit: int = 20,
) -> dict:
    """
    Analyze overall and serious reactions.

    Reaction counts are case counts:
    one case contributes at most once to a given
    Preferred Term.
    """

    reaction_rows = build_reaction_rows(cases)

    if reaction_rows.empty:
        return {
            "total_reaction_records": 0,
            "top_reactions": [],
            "top_serious_reactions": [],
        }

    total_cases = len(cases)

    # Count unique cases per Preferred Term.
    reaction_case_counts = (
        reaction_rows
        .groupby("reaction")["safetyreportid"]
        .nunique()
        .sort_values(ascending=False)
        .head(limit)
    )

    serious_rows = reaction_rows[
        reaction_rows["serious"]
        .astype(str)
        .str.lower()
        == "serious"
    ]

    serious_reaction_case_counts = (
        serious_rows
        .groupby("reaction")["safetyreportid"]
        .nunique()
        .sort_values(ascending=False)
        .head(limit)
    )

    top_reactions = [
        {
            "reaction": reaction,
            "case_count": int(count),
            "case_percentage": safe_percentage(
                int(count),
                total_cases,
            ),
        }
        for reaction, count
        in reaction_case_counts.items()
    ]

    top_serious_reactions = [
        {
            "reaction": reaction,
            "case_count": int(count),
            "case_percentage_of_all_cases": safe_percentage(
                int(count),
                total_cases,
            ),
            "case_percentage_of_serious_cases": safe_percentage(
                int(count),
                len(cases),
            ),
        }
        for reaction, count
        in serious_reaction_case_counts.items()
    ]

    return {
        "total_reaction_records": len(reaction_rows),
        "unique_reaction_terms": int(
            reaction_rows["reaction"].nunique()
        ),
        "top_reactions": top_reactions,
        "top_serious_reactions": top_serious_reactions,
    }


def analyze_outcomes(
    cases: pd.DataFrame,
) -> list[dict]:
    """
    Analyze reaction-level outcomes.

    Outcomes are counted from the expanded reaction records,
    not from the raw comma-separated case field.
    """

    reaction_rows = build_reaction_rows(cases)

    if reaction_rows.empty:
        return []

    counts = (
        reaction_rows["outcome"]
        .value_counts()
    )

    total = len(reaction_rows)

    return [
        {
            "outcome": outcome,
            "reaction_count": int(count),
            "percentage_of_reaction_records": safe_percentage(
                int(count),
                total,
            ),
        }
        for outcome, count
        in counts.items()
    ]