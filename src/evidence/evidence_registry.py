from __future__ import annotations

from copy import deepcopy


def create_evidence_registry(
    evidence: dict,
) -> dict:
    """
    Convert deterministic evidence into a traceable
    evidence registry.

    Each important evidence item receives a stable ID.
    """

    registry = {}
        # =========================================================
    # REPORTING PERIOD
    # =========================================================

    registry["EV-REPORT-001"] = {
        "evidence_id": "EV-REPORT-001",
        "category": "reporting_period",
        "metric": "reporting_period",
        "value": deepcopy(
            evidence["reporting_period"]
        ),
        "description": (
            "Start and end dates of the "
            "analysis reporting period."
        ),
    }

    # =========================================================
    # CASE VOLUME
    # =========================================================

    case_volume = evidence["case_volume"]

    registry["EV-CASE-001"] = {
        "evidence_id": "EV-CASE-001",
        "category": "case_volume",
        "metric": "total_cases",
        "value": case_volume["total_cases"],
        "description": (
            "Total number of canonical cases "
            "in the reporting period."
        ),
    }

    registry["EV-CASE-002"] = {
        "evidence_id": "EV-CASE-002",
        "category": "case_volume",
        "metric": "serious_cases",
        "value": case_volume["serious_cases"],
        "description": (
            "Number of serious canonical cases."
        ),
    }

    registry["EV-CASE-003"] = {
        "evidence_id": "EV-CASE-003",
        "category": "case_volume",
        "metric": "serious_percentage",
        "value": case_volume["serious_percentage"],
        "description": (
            "Percentage of canonical cases "
            "classified as serious."
        ),
    }

    registry["EV-CASE-004"] = {
        "evidence_id": "EV-CASE-004",
        "category": "case_volume",
        "metric": "non_serious_cases",
        "value": case_volume["non_serious_cases"],
        "description": (
            "Number of non-serious canonical cases."
        ),
    }

    registry["EV-CASE-005"] = {
        "evidence_id": "EV-CASE-005",
        "category": "case_volume",
        "metric": "expedited_cases",
        "value": case_volume["expedited_cases"],
        "description": (
            "Number of cases meeting expedited "
            "reporting criteria."
        ),
    }
    registry["EV-CASE-006"] = {
        "evidence_id": "EV-CASE-006",
        "category": "case_volume",
        "metric": "expedited_percentage",
        "value": case_volume[
            "expedited_percentage"
        ],
        "description": (
            "Percentage of canonical cases "
            "meeting expedited reporting criteria."
        ),
    }

    # =========================================================
    # DEMOGRAPHICS
    # =========================================================

    demographics = evidence["demographics"]

    registry["EV-DEMO-001"] = {
        "evidence_id": "EV-DEMO-001",
        "category": "demographics",
        "metric": "age_groups",
        "value": deepcopy(
            demographics["age_groups"]
        ),
        "description": (
            "Case distribution across age groups."
        ),
    }

    registry["EV-DEMO-002"] = {
        "evidence_id": "EV-DEMO-002",
        "category": "demographics",
        "metric": "sex",
        "value": deepcopy(
            demographics["sex"]
        ),
        "description": (
            "Case distribution by reported sex."
        ),
    }

    registry["EV-DEMO-003"] = {
        "evidence_id": "EV-DEMO-003",
        "category": "demographics",
        "metric": "country",
        "value": deepcopy(
            demographics["country"]
        ),
        "description": (
            "Case distribution by reporting country."
        ),
    }

    # =========================================================
    # REACTIONS
    # =========================================================

    reaction_analysis = evidence[
        "reaction_analysis"
    ]

    registry["EV-REACTION-001"] = {
        "evidence_id": "EV-REACTION-001",
        "category": "reaction_analysis",
        "metric": "total_reaction_records",
        "value": reaction_analysis[
            "total_reaction_records"
        ],
        "description": (
            "Total reaction records represented "
            "in the normalized case dataset."
        ),
    }

    registry["EV-REACTION-002"] = {
        "evidence_id": "EV-REACTION-002",
        "category": "reaction_analysis",
        "metric": "unique_reaction_terms",
        "value": reaction_analysis[
            "unique_reaction_terms"
        ],
        "description": (
            "Number of unique reaction terms."
        ),
    }

    # Top reactions get individual evidence IDs.
    for index, reaction in enumerate(
        reaction_analysis["top_reactions"],
        start=1,
    ):
        evidence_id = (
            f"EV-REACTION-TOP-{index:03d}"
        )

        registry[evidence_id] = {
            "evidence_id": evidence_id,
            "category": "reaction_analysis",
            "metric": "top_reaction",
            "value": deepcopy(reaction),
            "description": (
                f"Top reaction ranking item #{index}."
            ),
        }

    # Serious reactions get individual IDs.
    for index, reaction in enumerate(
        reaction_analysis["top_serious_reactions"],
        start=1,
    ):
        evidence_id = (
            f"EV-REACTION-SERIOUS-{index:03d}"
        )

        registry[evidence_id] = {
            "evidence_id": evidence_id,
            "category": "serious_reaction_analysis",
            "metric": "top_serious_reaction",
            "value": deepcopy(reaction),
            "description": (
                f"Top serious reaction ranking "
                f"item #{index}."
            ),
        }

    # =========================================================
    # OUTCOMES
    # =========================================================

    registry["EV-OUTCOME-001"] = {
        "evidence_id": "EV-OUTCOME-001",
        "category": "outcomes",
        "metric": "outcome_distribution",
        "value": deepcopy(
            evidence["outcomes"]
        ),
        "description": (
            "Distribution of reported reaction outcomes."
        ),
    }

    # =========================================================
    # TEMPORAL ANALYSIS
    # =========================================================

    temporal = evidence[
        "temporal_analysis"
    ]

    monthly = temporal[
        "monthly_trends"
    ]

    registry["EV-TREND-001"] = {
        "evidence_id": "EV-TREND-001",
        "category": "temporal_analysis",
        "metric": "highest_volume_month",
        "value": deepcopy(
            monthly["highest_volume_month"]
        ),
        "description": (
            "Highest-volume reporting month."
        ),
    }

    registry["EV-TREND-002"] = {
        "evidence_id": "EV-TREND-002",
        "category": "temporal_analysis",
        "metric": "lowest_volume_month",
        "value": deepcopy(
            monthly["lowest_volume_month"]
        ),
        "description": (
            "Lowest-volume reporting month."
        ),
    }

    registry["EV-TREND-003"] = {
        "evidence_id": "EV-TREND-003",
        "category": "temporal_analysis",
        "metric": "average_monthly_cases",
        "value": monthly[
            "average_monthly_cases"
        ],
        "description": (
            "Average monthly canonical case volume."
        ),
    }

    registry["EV-TREND-004"] = {
        "evidence_id": "EV-TREND-004",
        "category": "temporal_analysis",
        "metric": "monthly_trends",
        "value": deepcopy(
            monthly["months"]
        ),
        "description": (
            "Complete monthly case-volume trend."
        ),
    }

    for index, window in enumerate(
        temporal[
            "high_volume_15_day_windows"
        ],
        start=1,
    ):
        evidence_id = (
            f"EV-TREND-15D-{index:03d}"
        )

        registry[evidence_id] = {
            "evidence_id": evidence_id,
            "category": "temporal_analysis",
            "metric": "high_volume_15_day_window",
            "value": deepcopy(window),
            "description": (
                f"High-volume complete 15-day "
                f"window ranking item #{index}."
            ),
        }

    # =========================================================
    # LIMITATIONS
    # =========================================================

    registry["EV-LIMIT-001"] = {
        "evidence_id": "EV-LIMIT-001",
        "category": "limitations",
        "metric": "known_limitations",
        "value": deepcopy(
            evidence["limitations"]
        ),
        "description": (
            "Known limitations of the supplied dataset "
            "and analysis scope."
        ),
    }

    return registry