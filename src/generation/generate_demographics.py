from __future__ import annotations

import json
import sys
from pathlib import Path


# =============================================================
# PROJECT PATH SETUP
# =============================================================

# File location:
#
# project/
#   src/
#       generation/
#           generate_demographics.py
#
# Therefore:
# parents[0] = generation
# parents[1] = src
# parents[2] = project root

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

SRC_DIR = BASE_DIR / "src"


# Make src/ importable when this script is
# executed directly.
if str(SRC_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(SRC_DIR),
    )


# =============================================================
# APPLICATION IMPORTS
# =============================================================

from data.loader import load_dataset

from data.case_normalizer import (
    normalize_cases,
)

from evidence.evidence_builder import (
    build_evidence_pack,
)

from evidence.evidence_registry import (
    create_evidence_registry,
)

from context.context_builder import (
    build_demographics_context,
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
# SECTION OUTPUT PATH
# =============================================================

SECTION_OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "sections"
)

SECTION_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def save_section_output(
    result: dict,
    filename: str,
) -> Path:
    """
    Save a validated section result to outputs/sections.
    """

    output_path = (
        SECTION_OUTPUT_DIR
        / filename
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


# =============================================================
# PROMPT LOADING
# =============================================================

def load_prompt(
    filename: str,
) -> str:
    """
    Load a prompt from src/prompts.
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
# DEMOGRAPHICS EVIDENCE SELECTION
# =============================================================

def build_demographics_evidence(
    registry: dict,
) -> dict:
    """
    Select ONLY evidence relevant to the
    Demographics section.

    The complete 63-item registry is deliberately
    NOT sent to Gemini.

    Approved evidence:

        EV-DEMO-001
        EV-DEMO-002
        EV-DEMO-003
    """

    required_ids = [
        "EV-DEMO-001",
        "EV-DEMO-002",
        "EV-DEMO-003",
    ]

    selected = {}

    for evidence_id in required_ids:

        if evidence_id not in registry:

            raise KeyError(
                "Required Demographics evidence "
                f"ID is missing from registry: "
                f"{evidence_id}"
            )

        selected[evidence_id] = (
            registry[evidence_id]
        )

    return selected


# =============================================================
# BUILD GROUNDED DEMOGRAPHICS PROMPT
# =============================================================

def build_demographics_prompt(
    context: dict,
    evidence: dict,
) -> str:
    """
    Build the grounded prompt supplied to Gemini.

    Gemini receives:

        1. Demographics task instructions
        2. Approved evidence only
        3. Section-specific context
        4. Strict grounding requirements
    """

    demographics_prompt = load_prompt(
        "demographics.txt"
    )

    prompt = f"""
{demographics_prompt}

============================================================
APPROVED EVIDENCE
============================================================

The following evidence IDs are the ONLY approved sources
for factual claims in this Demographics section.

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

Use this section-specific context to understand the
demographic evidence available for this section.

{json.dumps(
    context,
    indent=2,
    ensure_ascii=False,
)}

============================================================
STRICT GROUNDING REQUIREMENTS
============================================================

1. Use ONLY the approved evidence supplied above.

2. Every factual claim MUST contain one or more evidence
   IDs.

3. Do NOT invent demographic statistics.

4. Do NOT calculate new percentages or statistics.

5. Preserve numerical values exactly as supplied by the
   deterministic analysis.

6. Describe observed age, sex, and country distributions
   conservatively.

7. Do NOT infer clinical risk from demographic distributions.

8. Do NOT infer causality.

9. Do NOT make safety-signal determinations.

10. Do NOT introduce demographic information that is not
    present in the evidence.

11. If evidence is insufficient to support a statement,
    do not make that statement.

12. Evidence IDs must be copied exactly.

13. Return ONLY the required JSON structure.

14. Do not combine multiple demographic evidence items into
    one claim unless every factual component of the combined
    claim is directly supported by the cited evidence IDs.
    
NUMERIC GROUNDING RULES
============================================================

- Every numeric value appearing in a claim MUST appear
  explicitly in the supplied evidence for that claim.

- Do NOT calculate, derive, infer, estimate, round, aggregate,
  subtract, compare, or transform numeric values.

- Do NOT introduce any new numeric threshold or boundary.

- Do NOT use comparative numeric expressions such as:
  "less than 6%",
  "more than 50%",
  "approximately 40%",
  "over 300 cases",
  "under 5%",
  "nearly 60%",
  or similar expressions unless the exact numeric value
  appearing in that expression is explicitly present in the
  supplied evidence.

- Do NOT convert evidence into a newly calculated percentage,
  count, ratio, average, range, difference, or threshold.

- Preserve all numerical values exactly as supplied by the
  deterministic evidence.

- Age-range labels such as <18, 18-44, 45-64, 65-74, and 75+
  are category labels, NOT statistics. Preserve them exactly
  when they are part of the evidence.

- If describing additional demographic categories for which
  detailed statistics are available, only state the values
  explicitly supported by the evidence.

- If a qualitative summary would require introducing an
  unsupported number, omit the number and use a purely
  qualitative statement instead.

- When in doubt, omit the statement rather than introduce
  an unsupported numeric value.

The required JSON structure is:

{{
    "section": "demographics",
    "claims": [
        {{
            "text": "grounded factual statement",
            "evidence_ids": [
                "EV-DEMO-..."
            ]
        }}
    ]
}}
"""

    return prompt.strip()


# =============================================================
# VALIDATE DEMOGRAPHICS
# =============================================================

def validate_demographics(
    generated: dict,
    registry: dict,
    demographics_evidence: dict,
) -> dict:
    """
    Validate generated Demographics against:

        - complete evidence registry
        - approved Demographics evidence
        - numeric grounding
    """

    allowed_evidence_ids = set(
        demographics_evidence.keys()
    )

    return validate_generated_section(
        generated=generated,
        registry=registry,
        allowed_evidence_ids=(
            allowed_evidence_ids
        ),
    )


# =============================================================
# PRINT GENERATED RESULT
# =============================================================

def print_generated_demographics(
    generated: dict,
) -> None:
    """
    Print structured Demographics output.
    """

    print("\n" + "=" * 70)
    print("GENERATED DEMOGRAPHICS")
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
    Print deterministic validation results.
    """

    # ---------------------------------------------------------
    # Overall validation
    # ---------------------------------------------------------

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
    # Evidence ID validation
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
# MAIN DEMOGRAPHICS PIPELINE
# =============================================================

def generate_demographics() -> dict:
    """
    Execute the complete grounded Demographics pipeline.

    Pipeline:

        Dataset
            ↓
        Case normalization
            ↓
        Deterministic evidence
            ↓
        Evidence registry
            ↓
        Demographics context
            ↓
        Demographics evidence selection
            ↓
        Gemini structured generation
            ↓
        Evidence ID validation
            ↓
        Numeric validation
    """

    print("=" * 70)
    print(
        "GENAR — GROUNDED DEMOGRAPHICS GENERATION"
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
        f"      Raw rows: "
        f"{len(raw_df):,}"
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
    # 5. BUILD DEMOGRAPHICS CONTEXT
    # =========================================================

    print(
        "[5/6] Building "
        "Demographics-specific context..."
    )

    context = build_demographics_context(
        evidence
    )

    demographics_evidence = (
        build_demographics_evidence(
            registry
        )
    )

    print(
        "      Approved Demographics evidence:"
    )

    for evidence_id in (
        demographics_evidence.keys()
    ):

        print(
            f"        {evidence_id}"
        )

    prompt = build_demographics_prompt(
        context=context,
        evidence=demographics_evidence,
    )

    # =========================================================
    # 6. GENERATE WITH GEMINI
    # =========================================================

    print(
        "[6/6] Generating grounded "
        "Demographics..."
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

    print_generated_demographics(
        generated
    )

    # =========================================================
    # VALIDATE
    # =========================================================

    validation = validate_demographics(
        generated=generated,
        registry=registry,
        demographics_evidence=(
            demographics_evidence
        ),
    )

    print_validation_results(
        validation
    )

    # =========================================================
    # BUILD FINAL RESULT
    # =========================================================

    result = {
        "section": "demographics",
        "claims": generated.get(
            "claims",
            [],
        ),
        "validation": validation,
    }

    # =========================================================
    # SAVE VALIDATED OUTPUT
    # =========================================================

    if validation["valid"]:

        output_path = save_section_output(
            result=result,
            filename="demographics.json",
        )

        print(
            f"\n      Saved validated section:"
            f"\n      {output_path}"
        )

    # =========================================================
    # FINAL STATUS
    # =========================================================

    print("\n" + "=" * 70)

    if validation["valid"]:

        print(
            "FINAL STATUS: "
            "VALIDATED DEMOGRAPHICS"
        )

    else:

        print(
            "FINAL STATUS: "
            "DEMOGRAPHICS VALIDATION FAILED"
        )

    print("=" * 70)

    return result


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    generate_demographics()