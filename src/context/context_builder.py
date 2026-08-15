from __future__ import annotations

import json


def _pretty_json(data: dict) -> str:
    """
    Convert evidence into readable JSON for the LLM.
    """

    return json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
    )


def build_overview_context(
    evidence: dict,
) -> dict:
    """
    Build context specifically for the Overview section.
    """

    return {
        "section": "overview",

        "reporting_period": evidence[
            "reporting_period"
        ],

        "case_volume": evidence[
            "case_volume"
        ],
    }


def build_demographics_context(
    evidence: dict,
) -> dict:
    """
    Build context specifically for the Demographics section.
    """

    demographics = evidence[
        "demographics"
    ]

    return {
        "section": "demographics",

        "reporting_period": evidence[
            "reporting_period"
        ],

        "demographics": demographics,
    }


def build_safety_findings_context(
    evidence: dict,
) -> dict:
    """
    Build context specifically for Safety Findings.
    """

    reaction_analysis = evidence[
        "reaction_analysis"
    ]

    outcomes = evidence[
        "outcomes"
    ]

    return {
        "section": "safety_findings",

        "reporting_period": evidence[
            "reporting_period"
        ],

        "reaction_analysis": reaction_analysis,

        "outcomes": outcomes,
    }


def build_trends_context(
    evidence: dict,
) -> dict:
    """
    Build context specifically for temporal trends.
    """

    temporal_analysis = evidence[
        "temporal_analysis"
    ]

    return {
        "section": "trends",

        "reporting_period": evidence[
            "reporting_period"
        ],

        "temporal_analysis": temporal_analysis,
    }


def build_limitations_context(
    evidence: dict,
) -> dict:
    """
    Build context specifically for limitations.
    """

    return {
        "section": "limitations",

        "limitations": evidence[
            "limitations"
        ],
    }


def build_conclusion_context(
    evidence: dict,
) -> dict:
    """
    Build a compact context for the conclusion.

    The conclusion receives high-level evidence only.
    """

    return {
        "section": "conclusion",

        "reporting_period": evidence[
            "reporting_period"
        ],

        "case_volume": evidence[
            "case_volume"
        ],

        "top_reactions": evidence[
            "reaction_analysis"
        ]["top_reactions"],

        "top_serious_reactions": evidence[
            "reaction_analysis"
        ]["top_serious_reactions"],

        "temporal_analysis": {
            "highest_volume_month": (
                evidence[
                    "temporal_analysis"
                ]["monthly_trends"][
                    "highest_volume_month"
                ]
            ),

            "high_volume_15_day_windows": (
                evidence[
                    "temporal_analysis"
                ][
                    "high_volume_15_day_windows"
                ]
            ),
        },

        "limitations": evidence[
            "limitations"
        ],
    }


def build_all_contexts(
    evidence: dict,
) -> dict:
    """
    Build all section-specific contexts.
    """

    return {
        "overview": build_overview_context(
            evidence
        ),

        "demographics": build_demographics_context(
            evidence
        ),

        "safety_findings": (
            build_safety_findings_context(
                evidence
            )
        ),

        "trends": build_trends_context(
            evidence
        ),

        "limitations": build_limitations_context(
            evidence
        ),

        "conclusion": build_conclusion_context(
            evidence
        ),
    }