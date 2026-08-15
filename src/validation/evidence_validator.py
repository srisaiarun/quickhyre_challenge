from __future__ import annotations

import re
from typing import Any


NUMBER_PATTERN = re.compile(
    r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?%?"
)


def _extract_numbers(text: str) -> list[str]:
    """
    Extract numeric tokens from generated text.

    Examples:
        1024
        1,024
        99.9%
        7.8
    """

    return NUMBER_PATTERN.findall(text)


def _normalize_number(value: str) -> float:
    """
    Normalize numeric strings for comparison.
    """

    cleaned = (
        value
        .replace(",", "")
        .replace("%", "")
        .strip()
    )

    return float(cleaned)


def _numbers_from_evidence(
    evidence_item: dict[str, Any],
) -> list[float]:
    """
    Recursively extract numeric values from an evidence item.
    """

    numbers = []

    def walk(value: Any):

        if isinstance(value, bool):
            return

        if isinstance(value, (int, float)):
            numbers.append(float(value))
            return

        if isinstance(value, dict):
            for nested in value.values():
                walk(nested)
            return

        if isinstance(value, list):
            for nested in value:
                walk(nested)
            return

    walk(evidence_item.get("value"))

    return numbers


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

        if not claim_text:
            errors.append(
                "Claim text is empty."
            )

        if not evidence_ids:
            errors.append(
                "Claim has no evidence IDs."
            )

        for evidence_id in evidence_ids:

            if evidence_id not in registry:
                errors.append(
                    f"Unknown evidence ID: "
                    f"{evidence_id}"
                )

        results.append(
            {
                "claim_index": index,
                "claim": claim_text,
                "evidence_ids": evidence_ids,
                "valid": len(errors) == 0,
                "errors": errors,
            }
        )

    return results


def validate_claim_numbers(
    claims: list[dict],
    registry: dict[str, dict],
) -> list[dict]:
    """
    Check numeric values appearing in generated claims
    against numeric values contained in their cited evidence.

    This is a conservative consistency check.

    It does not attempt to prove semantic correctness.
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

        claim_numbers_raw = _extract_numbers(
            claim_text
        )

        claim_numbers = []

        for number in claim_numbers_raw:

            try:
                claim_numbers.append(
                    _normalize_number(number)
                )
            except ValueError:
                continue

        supported_numbers = set()

        for evidence_id in evidence_ids:

            evidence_item = registry.get(
                evidence_id
            )

            if not evidence_item:
                continue

            for number in _numbers_from_evidence(
                evidence_item
            ):
                supported_numbers.add(
                    number
                )

        unsupported_numbers = []

        for number in claim_numbers:

            if number not in supported_numbers:
                unsupported_numbers.append(
                    number
                )

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
                    len(unsupported_numbers) == 0
                ),
            }
        )

    return results


def validate_generated_section(
    generated: dict,
    registry: dict[str, dict],
) -> dict:
    """
    Run all deterministic validation checks against
    a generated section.
    """

    errors = []

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

    evidence_validation = (
        validate_evidence_ids(
            claims,
            registry,
        )
    )

    number_validation = (
        validate_claim_numbers(
            claims,
            registry,
        )
    )

    invalid_evidence_claims = [
        item
        for item in evidence_validation
        if not item["valid"]
    ]

    invalid_number_claims = [
        item
        for item in number_validation
        if not item["valid"]
    ]

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

    return {
        "valid": len(errors) == 0,
        "section": section,
        "errors": errors,
        "evidence_validation": evidence_validation,
        "number_validation": number_validation,
    }