from __future__ import annotations

import re
from typing import Any


# =============================================================
# NUMBER EXTRACTION
# =============================================================

NUMBER_PATTERN = re.compile(
    r"(?<![\w.])"
    r"-?\d+(?:,\d{3})*"
    r"(?:\.\d+)?"
    r"%?"
    r"(?![\w.-])"
)


# =============================================================
# DATE PATTERNS
# =============================================================

# ISO month:
#   2025-07
#   2024-12
#
# ISO full date:
#   2025-07-15
#   2025-02-13
#
# We mask these before extracting numbers.

ISO_DATE_PATTERN = re.compile(
    r"\b\d{4}-\d{2}(?:-\d{2})?\b"
)


# Written dates:
#
#   February 13, 2025
#   October 23 to November 6, 2025
#   July 1 to July 15, 2025
#
# These are masked before numeric extraction so that
# day/month/year components are not treated as statistics.

WRITTEN_DATE_PATTERN = re.compile(
    r"""
    \b
    (?:
        January|February|March|April|May|June|
        July|August|September|October|November|December
    )
    \s+
    \d{1,2}
    (?:
        \s+
        (?:to|through|-)
        \s+
        (?:
            January|February|March|April|May|June|
            July|August|September|October|November|December
        )
        \s+
        \d{1,2}
    )?
    (?:
        \s*,?
        \s*
        \d{4}
    )?
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


# =============================================================
# DATE MASKING
# =============================================================

def _mask_dates(text: str) -> str:
    """
    Replace date expressions with spaces.

    This preserves character positions while ensuring that
    date components are not interpreted as analytical numbers.
    """

    masked_text = ISO_DATE_PATTERN.sub(
        lambda match: " " * len(match.group(0)),
        text,
    )

    masked_text = WRITTEN_DATE_PATTERN.sub(
        lambda match: " " * len(match.group(0)),
        masked_text,
    )

    return masked_text


# =============================================================
# RAW NUMBER EXTRACTION
# =============================================================

def _extract_numbers(text: str) -> list[str]:
    """
    Extract numeric values from generated claims while
    ignoring numbers that are part of dates.

    Extracted examples:

        1024
        1,024
        99.9%
        7.8
        -41.3

    Ignored examples:

        2025-07
        2024-12
        2025-02-13

        February 13, 2025
        October 23 to November 6, 2025
        July 1 to July 15, 2025
    """

    masked_text = _mask_dates(text)

    return NUMBER_PATTERN.findall(
        masked_text
    )


# =============================================================
# NUMBER NORMALIZATION
# =============================================================

def _normalize_number(value: str) -> float:
    """
    Normalize numeric strings for comparison.

    Examples:

        "1,024"  -> 1024.0
        "99.9%"  -> 99.9
        "-41.3%" -> -41.3
    """

    cleaned = (
        value
        .replace(",", "")
        .replace("%", "")
        .strip()
    )

    return float(cleaned)


# =============================================================
# YEAR DETECTION
# =============================================================

def _is_year(
    number_text: str,
    full_text: str,
    match_start: int,
) -> bool:
    """
    Determine whether a numeric token is a calendar year.

    Years such as 2024 and 2025 should not be treated as
    analytical statistics.
    """

    try:
        value = int(
            number_text
            .replace(",", "")
            .replace("%", "")
        )
    except ValueError:
        return False

    return 1900 <= value <= 2100
# =============================================================
# DEMOGRAPHIC AGE CATEGORY DETECTION
# =============================================================

def _is_demographic_category_number(
    number_text: str,
    full_text: str,
    match_start: int,
    match_end: int,
) -> bool:
    """
    Return True when a numeric token is part of a demographic
    age-category label rather than an analytical statistic.

    Supported age-category forms include:

        75+
        65-74
        65‑74
        45–64
        18-44
        <18
        >75
        75 years and older
        age group 65-74

    The function deliberately supports several Unicode dash
    characters because LLM output may use a non-ASCII hyphen
    such as U+2011 (non-breaking hyphen) or U+2013 (en dash).

    Analytical values such as:

        409 cases
        43.5%
        78.8 cases
        -41.3%

    are not treated as demographic category boundaries.
    """

    before = full_text[
        max(0, match_start - 40):
        match_start
    ]

    after = full_text[
        match_end:
        min(len(full_text), match_end + 40)
    ]

    context = (
        before + number_text + after
    ).lower()

    # ---------------------------------------------------------
    # Explicit age wording.
    # ---------------------------------------------------------

    age_context = bool(
        re.search(
            r"\bage(?:\s+group|\s+category)?\b",
            context,
        )
        or re.search(
            r"\byears?\s+(?:and\s+)?(?:older|younger)\b",
            context,
        )
        or re.search(
            r"\byears?\s*(?:old)?\b",
            context,
        )
    )

    # ---------------------------------------------------------
    # Explicit inequality age categories.
    #
    #     <18
    #     >75
    # ---------------------------------------------------------

    immediate_before = before.rstrip()

    if immediate_before.endswith("<") or immediate_before.endswith(">"):
        return True

    # ---------------------------------------------------------
    # Plus age category.
    #
    #     75+
    # ---------------------------------------------------------

    immediate_after = after.lstrip()

    if immediate_after.startswith("+"):
        return True

    # ---------------------------------------------------------
    # Unicode-aware age-range detection.
    #
    # Supported range separators:
    #
    #     -   ASCII hyphen-minus
    #     ‐   U+2010 hyphen
    #     ‑   U+2011 non-breaking hyphen
    #     ‒   U+2012 figure dash
    #     –   U+2013 en dash
    #     —   U+2014 em dash
    #
    # This is important because generated text may contain:
    #
    #     18-44
    #     18‑44
    #     18–44
    #
    # Both bounds are ignored because they are category labels,
    # not analytical statistics.
    # ---------------------------------------------------------

    RANGE_DASHES = r"\-‐‑‒–—"

    # Number is the LOWER bound:
    #     18-44
    #     18‑44
    #     18–44
    next_match = re.match(
        rf"^[{RANGE_DASHES}]\s*(\d{{1,3}})\b",
        immediate_after,
    )

    if next_match:
        try:
            lower = int(
                number_text.replace(",", "")
            )
            upper = int(
                next_match.group(1)
            )
        except ValueError:
            return False

        plausible_age_range = (
            0 <= lower <= 120
            and 0 <= upper <= 120
            and lower < upper
        )

        if age_context or plausible_age_range:
            return True

    # Number is the UPPER bound:
    #     18-44
    #     18‑44
    #     18–44
    previous_match = re.search(
        rf"(\d{{1,3}})\s*[{RANGE_DASHES}]\s*$",
        immediate_before,
    )

    if previous_match:
        try:
            lower = int(
                previous_match.group(1)
            )
            upper = int(
                number_text.replace(",", "")
            )
        except ValueError:
            return False

        plausible_age_range = (
            0 <= lower <= 120
            and 0 <= upper <= 120
            and lower < upper
        )

        if age_context or plausible_age_range:
            return True

    return False


# =============================================================
# MEANINGFUL NUMBER EXTRACTION
# =============================================================

def _extract_meaningful_numbers(
    text: str,
) -> list[float]:
    """
    Extract numbers that should be checked against evidence.

    Excludes:

    - ISO dates such as 2025-02
    - ISO dates such as 2025-02-13
    - written dates such as February 13, 2025
    - calendar years
    - demographic age-category boundaries such as 18-44,
      18‑44, 45-64, 65-74, 75+, <18, and >75
    - the 15 in "15-day window"

    Keeps actual analytical values such as:

    - 1024
    - 78.8
    - 41.3%
    - 61
    """

    meaningful: list[float] = []

    # ---------------------------------------------------------
    # MASK ALL DATE EXPRESSIONS
    # ---------------------------------------------------------

    masked_text = _mask_dates(text)

    # ---------------------------------------------------------
    # EXTRACT REMAINING NUMERIC TOKENS
    # ---------------------------------------------------------

    matches = list(
        NUMBER_PATTERN.finditer(
            masked_text
        )
    )

    for match in matches:

        raw = match.group(0)
                # -----------------------------------------------------
        # Ignore demographic age-category boundaries.
        #
        # Examples:
        #     75+
        #     65-74
        #     45-64
        #     18-44
        #     <18
        #     >75
        #
        # These are category labels, not analytical values.
        # -----------------------------------------------------

        if _is_demographic_category_number(
            raw,
            masked_text,
            match.start(),
            match.end(),
        ):
            continue

        # -----------------------------------------------------
        # Ignore calendar years.
        # -----------------------------------------------------

        if _is_year(
            raw,
            masked_text,
            match.start(),
        ):
            continue

        # -----------------------------------------------------
        # Ignore methodology phrase "15-day".
        # -----------------------------------------------------

        end = match.end()

        following_text = masked_text[end:]

        if re.match(
            r"[- ]day\b",
            following_text,
            re.IGNORECASE,
        ):
            if raw == "15":
                continue

        # -----------------------------------------------------
        # Normalize numeric value.
        # -----------------------------------------------------

        try:
            meaningful.append(
                _normalize_number(raw)
            )
        except ValueError:
            continue

    return meaningful


# =============================================================
# EVIDENCE NUMBER EXTRACTION
# =============================================================

def _numbers_from_evidence(
    evidence_item: dict[str, Any],
) -> list[float]:
    """
    Recursively extract numeric values from an evidence item.
    """

    numbers: list[float] = []

    def walk(value: Any) -> None:

        if isinstance(value, bool):
            return

        if isinstance(
            value,
            (int, float),
        ):
            numbers.append(
                float(value)
            )
            return

        if isinstance(
            value,
            dict,
        ):
            for nested in value.values():
                walk(nested)

            return

        if isinstance(
            value,
            list,
        ):
            for nested in value:
                walk(nested)

            return

    walk(
        evidence_item.get("value")
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

        if not isinstance(
            claim,
            dict,
        ):
            results.append(
                {
                    "claim_index": index,
                    "claim": "",
                    "evidence_ids": [],
                    "valid": False,
                    "errors": [
                        "Claim must be an object."
                    ],
                }
            )

            continue

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
        # Claim text
        # -----------------------------------------------------

        if not isinstance(
            claim_text,
            str,
        ) or not claim_text.strip():

            errors.append(
                "Claim text is empty."
            )

        # -----------------------------------------------------
        # Evidence IDs
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
        # Registry lookup
        # -----------------------------------------------------

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


# =============================================================
# NUMERIC VALIDATION
# =============================================================

def validate_claim_numbers(
    claims: list[dict],
    registry: dict[str, dict],
) -> list[dict]:
    """
    Check numeric values appearing in generated claims
    against numeric values contained in their cited evidence.

    This is a conservative consistency check.

    Special handling:

    - Calendar years are ignored.
    - ISO dates are ignored.
    - Written dates are ignored.
    - "15-day" methodology wording is ignored.
    - Percentage direction is handled conservatively so that
      -41.3 in evidence can support "41.3% decrease".
    """

    results = []

    for index, claim in enumerate(
        claims,
        start=1,
    ):

        if not isinstance(
            claim,
            dict,
        ):
            results.append(
                {
                    "claim_index": index,
                    "claim": "",
                    "claim_numbers": [],
                    "supported_numbers": [],
                    "unsupported_numbers": [],
                    "valid": False,
                }
            )

            continue

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
        # Extract meaningful claim numbers.
        # -----------------------------------------------------

        claim_numbers = (
            _extract_meaningful_numbers(
                claim_text
            )
        )

        # -----------------------------------------------------
        # Extract supported evidence numbers.
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Compare numbers.
        # -----------------------------------------------------

        unsupported_numbers = []

        for number in claim_numbers:

            if number in supported_numbers:
                continue

            # -------------------------------------------------
            # Percentage direction handling.
            #
            # Evidence:
            #     -41.3
            #
            # Claim:
            #     41.3% decrease
            #
            # Both represent the same observed change.
            # -------------------------------------------------

            if (
                abs(number)
                in {
                    abs(value)
                    for value in supported_numbers
                }
            ):
                continue

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


# =============================================================
# SECTION VALIDATION
# =============================================================

def validate_generated_section(
    generated: dict,
    registry: dict[str, dict],
    allowed_evidence_ids: set[str] | None = None,
) -> dict:
    """
    Run all deterministic validation checks against
    a generated section.

    Validation includes:

    1. Required section field.
    2. Claims must be a list.
    3. Every claim must contain text.
    4. Every claim must contain evidence IDs.
    5. Every evidence ID must exist in the registry.
    6. Evidence IDs must be approved for the section
       when allowed_evidence_ids is supplied.
    7. Numeric values in claims must exist in cited evidence.
    """

    errors = []

    # =========================================================
    # BASIC STRUCTURE
    # =========================================================

    if not isinstance(
        generated,
        dict,
    ):

        return {
            "valid": False,
            "section": None,
            "errors": [
                "Generated response must be an object."
            ],
            "evidence_validation": [],
            "number_validation": [],
        }

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
    # ALLOWED EVIDENCE
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
    # INVALID CLAIMS
    # =========================================================

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

    # =========================================================
    # ERROR SUMMARY
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
        "evidence_validation": evidence_validation,
        "number_validation": number_validation,
    }