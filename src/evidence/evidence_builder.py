from __future__ import annotations

import pandas as pd

from analysis.case_analysis import (
    analyze_case_volume,
    analyze_demographics,
)

from analysis.reaction_analysis import (
    analyze_outcomes,
    analyze_reactions,
)

from analysis.time_analysis import (
    analyze_monthly_trends,
    analyze_15_day_windows,
    detect_15_day_spikes,
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
    Build the approved deterministic evidence packet.

    The LLM should use this packet as its source of truth.
    """

    # ---------------------------------------------------------
    # CASE ANALYSIS
    # ---------------------------------------------------------
    case_volume = analyze_case_volume(cases)

    demographics = analyze_demographics(cases)

    # ---------------------------------------------------------
    # REACTION ANALYSIS
    # ---------------------------------------------------------
    reactions = analyze_reactions(cases)

    outcomes = analyze_outcomes(cases)

    # ---------------------------------------------------------
    # TEMPORAL ANALYSIS
    # ---------------------------------------------------------
    monthly_trends = analyze_monthly_trends(cases)

    rolling_15_day_windows = analyze_15_day_windows(
        cases
    )

    high_volume_15_day_windows = detect_15_day_spikes(
        rolling_15_day_windows
    )

    # ---------------------------------------------------------
    # REPORTING PERIOD
    # ---------------------------------------------------------
    reporting_period = build_reporting_period(cases)

    # ---------------------------------------------------------
    # EVIDENCE PACK
    # ---------------------------------------------------------
    evidence = {
        "metadata": {
            "product": "Bisoprolol",
            "report_type": "PADER-style safety report",
            "application_identifier": "B-1",
            "data_source": (
                "Bisoprolol ICSR supplied challenge dataset"
            ),
            "analysis_method": (
                "Deterministic Python analysis"
            ),
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

        "temporal_analysis": {
            "monthly_trends": monthly_trends,

            "high_volume_15_day_windows": (
                high_volume_15_day_windows
            ),

            "methodology": {
                "monthly_analysis": (
                    "Cases grouped by reporting month."
                ),
                "rolling_15_day_analysis": (
                    "Each calendar day was evaluated as "
                    "the start of a 15-day window."
                ),
                "high_volume_observation": (
                    "Top complete 15-day windows ranked "
                    "by observed case volume."
                ),
                "interpretation": (
                    "These are descriptive observations "
                    "for human review and are not "
                    "automatic safety-signal determinations."
                ),
            },
        },

        "limitations": {
            "system_organ_class": (
                "SOC-level analysis is unavailable "
                "because the supplied dataset contains "
                "MedDRA Preferred Terms but no SOC field."
            ),

            "expectedness": (
                "Expectedness cannot be determined because "
                "no product label or CCDS was supplied."
            ),

            "history_of_actions": (
                "No history-of-actions data was supplied "
                "for this exercise."
            ),

            "causality": (
                "The dataset-level analysis does not "
                "establish causality between bisoprolol "
                "and reported reactions."
            ),
        },
    }

    return evidence