from __future__ import annotations

from datetime import date

import pandas as pd

from analysis.case_analysis import (
    analyze_case_volume,
    analyze_demographics,
)

from analysis.reaction_analysis import (
    analyze_outcomes,
    analyze_reactions,
)


def _date_to_string(value) -> str | None:
    """Convert a pandas timestamp/date to ISO text."""

    if pd.isna(value):
        return None

    return pd.Timestamp(value).date().isoformat()


def build_reporting_period(
    cases: pd.DataFrame,
) -> dict:
    """Build reporting-period evidence."""

    dates = pd.to_datetime(
        cases["report_date"],
        errors="coerce",
    ).dropna()

    if dates.empty:
        return {
            "start_date": None,
            "end_date": None,
        }

    return {
        "start_date": _date_to_string(dates.min()),
        "end_date": _date_to_string(dates.max()),
    }


def build_evidence_pack(
    cases: pd.DataFrame,
) -> dict:
    """
    Build the approved evidence packet.

    This packet contains deterministic facts only.
    The LLM should use this packet as its source of truth.
    """

    case_volume = analyze_case_volume(cases)

    demographics = analyze_demographics(cases)

    reactions = analyze_reactions(cases)

    outcomes = analyze_outcomes(cases)

    reporting_period = build_reporting_period(cases)

    evidence = {
        "metadata": {
            "product": "Bisoprolol",
            "report_type": "PADER-style safety report",
            "application_identifier": "B-1",
            "data_source": "Bisoprolol ICSR supplied challenge dataset",
            "analysis_method": "Deterministic Python analysis",
        },

        "reporting_period": reporting_period,

        "case_volume": case_volume,

        "demographics": demographics,

        "reaction_analysis": {
            "total_reaction_records": reactions[
                "total_reaction_records"
            ],
            "unique_reaction_terms": reactions[
                "unique_reaction_terms"
            ],
            "top_reactions": reactions[
                "top_reactions"
            ],
            "top_serious_reactions": reactions[
                "top_serious_reactions"
            ],
        },

        "outcomes": outcomes,

        "limitations": {
            "system_organ_class": (
                "SOC-level analysis is unavailable because "
                "the supplied dataset contains MedDRA "
                "Preferred Terms but no SOC field."
            ),
            "expectedness": (
                "Expectedness cannot be determined because "
                "no product label or CCDS was supplied."
            ),
            "history_of_actions": (
                "No history-of-actions data was supplied "
                "for this exercise."
            ),
        },
    }

    return evidence