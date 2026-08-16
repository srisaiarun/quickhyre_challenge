from __future__ import annotations

import re
from typing import Any


# =============================================================
# PATTERNS
# =============================================================

NUMBER_PATTERN = re.compile(
    r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?%?"
)

DATE_PATTERN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"
)

# Age-group expressions commonly used by the deterministic
# demographic analysis.
#
# Examples:
#   <18
#   18-44
#   45-64
#   65-74
#   75+
#
# We treat these as categorical labels rather than numeric
# claims.
AGE_GROUP_PATTERN = re.compile(
    r"""
    (?:
        <\s*\d+
        |
        \d+\s*-\s*\d+
        |
        \d+\s*\+
    )
    """,
    re.VERBOSE,
)


# =============================================================
# NUMBER EXTRACTION
# =============================================================

def _mask_non_metric_numbers(
    text: str,
) -> str:
    """
    Remove textual constructs whose numbers should NOT be
    interpreted as quantitative claims.

    Currently masks:

    1. ISO dates:
           2025-12-26

    2. Age-group labels:
           <18
           18-44
           45-64
           65-74
           75+

    This prevents the numeric validator from incorrectly
    treating demographic category labels as unsupported
    statistics.
    """

    text = DATE_PATTERN.sub(
        " ",
        text,
    )

    text = AGE_GROUP_PATTERN.sub(
        " ",
        text,
    )

    return text


def _extract_numbers(
    text: str,
) -> list[str]:
    """
    Extract numeric tokens from generated text.

    Examples:

        1024
        1,024
        99.9%
        7.8
    """

    return NUMBER_PATTERN.findall(
        text
    )


def _extract_claim_numbers(
    text: str,
) -> list[str]:
    """
    Extract actual quantitative values from a claim.

    Non-metric constructs such as dates and age-group labels
    are removed first.
    """

    cleaned_text = _mask_non_metric_numbers(
        text
    )

    return _extract_numbers(
        cleaned_text
    )


def _normalize_number(
    value: str,
) -> float:
    """
    Normalize numeric strings for comparison.

    Examples:

        "1,024" -> 1024.0
        "99.9%" -> 99.9
        "7.8"   -> 7.8
    """

    cleaned = (
        value
        .replace(",", "")
        .replace("%", "")
        .strip()
    )

    return float(
        cleaned
    )


# =============================================================
# EVIDENCE NUMERIC EXTRACTION
# =============================================================

def _numbers_from_evidence(
    evidence_item: dict[str, Any],
) -> list[float]:
    """
    Recursively extract numeric values from an evidence item.

    Numeric values inside the evidence's `value` field are
    considered deterministic quantitative evidence.

    Strings are deliberately ignored because categorical
    labels such as:

        18-44
        65-74
        75+

    are not quantitative metrics.
    """

    numbers: list[float] = []

    def walk(
        value: Any,
    ) -> None:

        # -----------------------------------------------------
        # Boolean values are not numeric evidence.
        # -----------------------------------------------------

        if isinstance(
            value,
            bool,
        ):
            return

        # -----------------------------------------------------
        # Numeric values
        # -----------------------------------------------------

        if isinstance(
            value,
            (int, float),
        ):

            numbers.append(
                float(value)
            )

            return

        # -----------------------------------------------------
        # Dictionaries
        # -----------------------------------------------------

        if isinstance(
            value,
            dict,
        ):

            for nested in value.values():

                walk(
                    nested
                )

            return

        # -----------------------------------------------------
        # Lists
        # -----------------------------------------------------

        if isinstance(
            value,
            list,
        ):

            for nested in value:

                walk(
                    nested
                )

            return

        # -----------------------------------------------------
        # Strings deliberately ignored.
        # -----------------------------------------------------

    walk(
        evidence_item.get(
            "value"
        )
    )

    return numbers


# =============================================================
# EVIDENCE ID VALIDATION
# =============================================================

def validate_evidence_ids(
    claims: list[dict],
    registry: dict[str, dict],
) -> list[dict]:
    """
    Validate that every evidence ID referenced by a claim
    actually exists in the evidence registry.
    """

    results = []

    for index, claim in enumerate(
        claims,
        start=1,
    ):

        claim_text = claim.get(
            "text",
            "",
        )

        evidence_ids = claim.get(
            "evidence_ids",
            [],
        )

        errors = []

        # -----------------------------------------------------
        # Validate claim text.
        # -----------------------------------------------------

        if not isinstance(
            claim_text,
            str,
        ) or not claim_text.strip():

            errors.append(
                "Claim text is empty."
            )

        # -----------------------------------------------------
        # Validate evidence ID container.
        # -----------------------------------------------------

        if not evidence_ids:

            errors.append(
                "Claim has no evidence IDs."
            )

        if not isinstance(
            evidence_ids,
            list,
        ):

            errors.append(
                "Evidence IDs must be a list."
            )

            evidence_ids = []

        # -----------------------------------------------------
        # Validate every evidence ID.
        # -----------------------------------------------------

        for evidence_id in evidence_ids:

            if evidence_id not in registry:

                errors.append(
                    "Unknown evidence ID: "
                    f"{evidence_id}"
                )

        results.append(
            {
                "claim_index": index,
                "claim": claim_text,
                "evidence_ids": evidence_ids,
                "valid": (
                    len(errors) == 0
                ),
                "errors": errors,
            }
        )

    return results


# =============================================================
# NUMERIC CLAIM VALIDATION
# =============================================================

def validate_claim_numbers(
    claims: list[dict],
    registry: dict[str, dict],
) -> list[dict]:
    """
    Check numeric values appearing in generated claims
    against numeric values contained in their cited evidence.

    This is a conservative consistency check.

    It does not attempt to prove semantic correctness.

    Dates and categorical age-group labels are excluded from
    numeric validation.
    """

    results = []

    for index, claim in enumerate(
        claims,
        start=1,
    ):

        claim_text = claim.get(
            "text",
            "",
        )

        evidence_ids = claim.get(
            "evidence_ids",
            [],
        )

        if not isinstance(
            evidence_ids,
            list,
        ):

            evidence_ids = []

        # -----------------------------------------------------
        # Extract quantitative numbers from claim.
        # -----------------------------------------------------

        claim_numbers_raw = (
            _extract_claim_numbers(
                claim_text
            )
        )

        claim_numbers: list[float] = []

        for number in claim_numbers_raw:

            try:

                claim_numbers.append(
                    _normalize_number(
                        number
                    )
                )

            except ValueError:

                continue

        # -----------------------------------------------------
        # Collect deterministic numeric evidence.
        # -----------------------------------------------------

        supported_numbers: set[float] = set()

        for evidence_id in evidence_ids:

            evidence_item = registry.get(
                evidence_id
            )

            if not evidence_item:

                continue

            for number in (
                _numbers_from_evidence(
                    evidence_item
                )
            ):

                supported_numbers.add(
                    number
                )

        # -----------------------------------------------------
        # Identify unsupported quantitative values.
        # -----------------------------------------------------

        unsupported_numbers = []

        for number in claim_numbers:

            if number not in supported_numbers:

                unsupported_numbers.append(
                    number
                )

        # -----------------------------------------------------
        # Result
        # -----------------------------------------------------

        results.append(
            {
                "claim_index": index,
                "claim": claim_text,
                "claim_numbers": claim_numbers,
                "supported_numbers": sorted(
                    supported_numbers
                ),
                "unsupported_numbers": (
                    unsupported_numbers
                ),
                "valid": (
                    len(
                        unsupported_numbers
                    ) == 0
                ),
            }
        )

    return results


# =============================================================
# COMPLETE SECTION VALIDATION
# =============================================================

def validate_generated_section(
    generated: dict,
    registry: dict[str, dict],
    allowed_evidence_ids: set[str] | None = None,
) -> dict:
    """
    Run all deterministic validation checks against a
    generated report section.

    Validation includes:

    1. Required section field.
    2. Claims must be a list.
    3. Claim text must exist.
    4. Every claim must contain evidence IDs.
    5. Every evidence ID must exist in the registry.
    6. Evidence IDs must be approved for the section.
    7. Numeric values in claims must exist in cited evidence.

    Dates and categorical age-group labels are excluded from
    numeric validation.
    """

    errors: list[str] = []

    # =========================================================
    # BASIC STRUCTURE
    # =========================================================

    section = generated.get(
        "section"
    )

    claims = generated.get(
        "claims"
    )

    if not section:

        errors.append(
            "Generated section is missing."
        )

    if not isinstance(
        claims,
        list,
    ):

        errors.append(
            "Generated claims must be a list."
        )

        return {
            "valid": False,
            "section": section,
            "errors": errors,
            "evidence_validation": [],
            "number_validation": [],
        }

    # =========================================================
    # ALLOWED EVIDENCE VALIDATION
    # =========================================================

    if allowed_evidence_ids is not None:

        for claim in claims:

            if not isinstance(
                claim,
                dict,
            ):

                continue

            evidence_ids = claim.get(
                "evidence_ids",
                [],
            )

            if not isinstance(
                evidence_ids,
                list,
            ):

                continue

            for evidence_id in evidence_ids:

                if (
                    evidence_id
                    not in allowed_evidence_ids
                ):

                    errors.append(
                        "Claim references evidence "
                        f"{evidence_id}, which is not "
                        "approved for this section."
                    )

    # =========================================================
    # EVIDENCE ID VALIDATION
    # =========================================================

    evidence_validation = (
        validate_evidence_ids(
            claims,
            registry,
        )
    )

    # =========================================================
    # NUMERIC VALIDATION
    # =========================================================

    number_validation = (
        validate_claim_numbers(
            claims,
            registry,
        )
    )

    # =========================================================
    # INVALID EVIDENCE CLAIMS
    # =========================================================

    invalid_evidence_claims = [
        item
        for item in evidence_validation
        if not item["valid"]
    ]

    # =========================================================
    # INVALID NUMERIC CLAIMS
    # =========================================================

    invalid_number_claims = [
        item
        for item in number_validation
        if not item["valid"]
    ]

    # =========================================================
    # ADD ERRORS
    # =========================================================

    if invalid_evidence_claims:

        errors.append(
            "One or more claims contain "
            "invalid evidence references."
        )

    if invalid_number_claims:

        errors.append(
            "One or more claims contain "
            "unsupported numeric values."
        )

    # =========================================================
    # FINAL RESULT
    # =========================================================

    return {
        "valid": len(errors) == 0,
        "section": section,
        "errors": errors,
        "evidence_validation": (
            evidence_validation
        ),
        "number_validation": (
            number_validation
        ),
    }