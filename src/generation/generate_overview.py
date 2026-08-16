from __future__ import annotations

import json
import sys
from pathlib import Path


# =============================================================
# PROJECT PATH SETUP
# =============================================================

# generate_overview.py is located at:
#
# project/
#   src/
#       generation/
#           generate_overview.py
#
# Therefore:
#   parents[0] = generation
#   parents[1] = src
#   parents[2] = project root

BASE_DIR = Path(
    __file__
).resolve().parents[2]

SRC_DIR = BASE_DIR / "src"


# Add src/ to Python's import path.
#
# This allows imports such as:
#
#     from data.loader import load_dataset
#     from evidence.evidence_builder import ...
#
# to work when this file is executed directly.
if str(SRC_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(SRC_DIR),
    )


# =============================================================
# APPLICATION IMPORTS
# =============================================================

from data.loader import load_dataset
from data.case_normalizer import normalize_cases

from evidence.evidence_builder import (
    build_evidence_pack,
)

from evidence.evidence_registry import (
    create_evidence_registry,
)

from context.context_builder import (
    build_overview_context,
)

from llm.client import (
    GeminiClient,
)

from validation.evidence_validator import (
    validate_generated_section,
)


# =============================================================
# PROMPT PATH
# =============================================================

PROMPTS_DIR = (
    BASE_DIR
    / "src"
    / "prompts"
)


# =============================================================
# PROMPT LOADING
# =============================================================

def load_prompt(
    filename: str,
) -> str:
    """
    Load a prompt file from src/prompts.
    """

    path = PROMPTS_DIR / filename

    if not path.exists():

        raise FileNotFoundError(
            f"Prompt file not found: {path}"
        )

    return path.read_text(
        encoding="utf-8"
    ).strip()


# =============================================================
# EVIDENCE SELECTION
# =============================================================

def build_overview_evidence(
    registry: dict,
) -> dict:
    """
    Select ONLY evidence relevant to the Overview section.

    Gemini must not receive the complete evidence registry.
    """

    required_ids = [
        "EV-REPORT-001",
        "EV-CASE-001",
        "EV-CASE-002",
        "EV-CASE-003",
        "EV-CASE-004",
        "EV-CASE-005",
        "EV-CASE-006",
    ]

    selected = {}

    for evidence_id in required_ids:

        if evidence_id not in registry:

            raise KeyError(
                "Required Overview evidence ID "
                f"is missing from registry: "
                f"{evidence_id}"
            )

        selected[evidence_id] = registry[
            evidence_id
        ]

    return selected


# =============================================================
# BUILD GROUNDED LLM PROMPT
# =============================================================

def build_overview_prompt(
    context: dict,
    evidence: dict,
) -> str:
    """
    Build the user prompt for the Overview section.

    The system prompt is supplied separately to Gemini.
    """

    overview_prompt = load_prompt(
        "overview.txt"
    )

    prompt = f"""
{overview_prompt}

============================================================
APPROVED EVIDENCE
============================================================

The following evidence IDs are the ONLY approved sources
for factual claims in this Overview section.

You MUST NOT reference evidence IDs that are not present
below.

{json.dumps(
    evidence,
    indent=2,
    ensure_ascii=False,
)}

============================================================
SECTION CONTEXT
============================================================

Use this section-specific context to understand the scope
of the Overview.

{json.dumps(
    context,
    indent=2,
    ensure_ascii=False,
)}

============================================================
GROUNDING REQUIREMENTS
============================================================

1. Use only the approved evidence supplied above.

2. Every factual claim MUST contain one or more evidence IDs.

3. Do not invent facts, statistics, dates, percentages,
   clinical interpretations, or conclusions.

4. Preserve numerical values from the evidence exactly.

5. Do not calculate new statistics.

6. Do not establish causality between bisoprolol and any
   reported event.

7. Do not determine expectedness because no product label
   or CCDS was supplied.

8. Do not introduce SOC-level conclusions because SOC data
   is not available in the supplied dataset.

9. Keep the wording appropriate for a pharmacovigilance
   safety report.

10. Return ONLY the required structured JSON response.

The required JSON structure is:

{{
    "section": "overview",
    "claims": [
        {{
            "text": "grounded factual statement",
            "evidence_ids": [
                "EV-..."
            ]
        }}
    ]
}}
"""

    return prompt.strip()


# =============================================================
# VALIDATE OVERVIEW
# =============================================================

def validate_overview(
    generated: dict,
    registry: dict,
    overview_evidence: dict,
) -> dict:
    """
    Validate the generated Overview against:

    - complete evidence registry
    - Overview-approved evidence IDs
    - numeric grounding
    """

    allowed_evidence_ids = set(
        overview_evidence.keys()
    )

    return validate_generated_section(
        generated=generated,
        registry=registry,
        allowed_evidence_ids=allowed_evidence_ids,
    )


# =============================================================
# PRINT GENERATED OVERVIEW
# =============================================================

def print_generated_overview(
    generated: dict,
) -> None:
    """
    Print the generated structured Overview.
    """

    print("\n" + "=" * 70)
    print("GENERATED OVERVIEW")
    print("=" * 70)

    print(
        json.dumps(
            generated,
            indent=2,
            ensure_ascii=False,
        )
    )


# =============================================================
# PRINT VALIDATION RESULTS
# =============================================================

def print_validation_results(
    validation: dict,
) -> None:
    """
    Print detailed deterministic validation results.
    """

    print("\n" + "=" * 70)
    print("EVIDENCE VALIDATION")
    print("=" * 70)

    if validation["valid"]:

        print("STATUS: PASS")

    else:

        print("STATUS: FAIL")

        for error in validation[
            "errors"
        ]:

            print(
                f"ERROR: {error}"
            )

    # ---------------------------------------------------------
    # Evidence validation
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLAIM VALIDATION DETAILS")
    print("=" * 70)

    for item in validation[
        "evidence_validation"
    ]:

        status = (
            "PASS"
            if item["valid"]
            else "FAIL"
        )

        print(
            f"\nClaim {item['claim_index']}: "
            f"{status}"
        )

        print(
            f"Text: {item['claim']}"
        )

        print(
            "Evidence IDs: "
            f"{item['evidence_ids']}"
        )

        if item["errors"]:

            for error in item[
                "errors"
            ]:

                print(
                    f"  ERROR: {error}"
                )

    # ---------------------------------------------------------
    # Numeric validation
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("NUMERIC VALIDATION")
    print("=" * 70)

    for item in validation[
        "number_validation"
    ]:

        status = (
            "PASS"
            if item["valid"]
            else "FAIL"
        )

        print(
            f"\nClaim {item['claim_index']}: "
            f"{status}"
        )

        print(
            f"Claim numbers: "
            f"{item['claim_numbers']}"
        )

        print(
            f"Supported numbers: "
            f"{item['supported_numbers']}"
        )

        if item[
            "unsupported_numbers"
        ]:

            print(
                "Unsupported numbers: "
                f"{item['unsupported_numbers']}"
            )


# =============================================================
# MAIN GENERATION PIPELINE
# =============================================================

def generate_overview() -> dict:
    """
    Execute the complete grounded Overview pipeline.

    Pipeline:

        Dataset
            ↓
        Case normalization
            ↓
        Deterministic evidence
            ↓
        Evidence registry
            ↓
        Overview context
            ↓
        Overview evidence selection
            ↓
        Gemini structured generation
            ↓
        Evidence validation
            ↓
        Numeric validation
    """

    print("=" * 70)
    print(
        "GENAR — GROUNDED OVERVIEW GENERATION"
    )
    print("=" * 70)

    # =========================================================
    # 1. LOAD DATASET
    # =========================================================

    print(
        "\n[1/6] Loading dataset..."
    )

    raw_df = load_dataset()

    print(
        f"      Raw rows: {len(raw_df):,}"
    )

    # =========================================================
    # 2. NORMALIZE CASES
    # =========================================================

    print(
        "[2/6] Normalizing cases..."
    )

    cases_df = normalize_cases(
        raw_df
    )

    print(
        f"      Canonical cases: "
        f"{len(cases_df):,}"
    )

    # =========================================================
    # 3. BUILD DETERMINISTIC EVIDENCE
    # =========================================================

    print(
        "[3/6] Building evidence pack..."
    )

    evidence = build_evidence_pack(
        cases_df
    )

    print(
        "      Deterministic evidence built."
    )

    # =========================================================
    # 4. BUILD EVIDENCE REGISTRY
    # =========================================================

    print(
        "[4/6] Building evidence registry..."
    )

    registry = create_evidence_registry(
        evidence
    )

    print(
        f"      Registry items: "
        f"{len(registry)}"
    )

    # =========================================================
    # 5. BUILD OVERVIEW CONTEXT
    # =========================================================

    print(
        "[5/6] Building Overview-specific context..."
    )

    context = build_overview_context(
        evidence
    )

    overview_evidence = (
        build_overview_evidence(
            registry
        )
    )

    print(
        "      Approved Overview evidence:"
    )

    for evidence_id in (
        overview_evidence.keys()
    ):

        print(
            f"        {evidence_id}"
        )

    prompt = build_overview_prompt(
        context=context,
        evidence=overview_evidence,
    )

    # =========================================================
    # 6. GENERATE WITH GEMINI
    # =========================================================

    print(
        "[6/6] Generating grounded Overview..."
    )

    client = GeminiClient()

    generated = client.generate_json(
        prompt=prompt,
        system_instruction=load_prompt(
            "system.txt"
        ),
    )

    # =========================================================
    # DISPLAY GENERATED RESULT
    # =========================================================

    print_generated_overview(
        generated
    )

    # =========================================================
    # VALIDATE
    # =========================================================

    validation = validate_overview(
        generated=generated,
        registry=registry,
        overview_evidence=overview_evidence,
    )

    print_validation_results(
        validation
    )

    # =========================================================
    # FINAL STATUS
    # =========================================================

    print("\n" + "=" * 70)

    if validation["valid"]:

        print(
            "FINAL STATUS: "
            "VALIDATED OVERVIEW"
        )

    else:

        print(
            "FINAL STATUS: "
            "VALIDATION FAILED"
        )

    print("=" * 70)

    return {
        "generated": generated,
        "validation": validation,
        "context": context,
        "overview_evidence": overview_evidence,
    }


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    generate_overview()