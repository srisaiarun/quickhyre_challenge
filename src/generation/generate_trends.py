from __future__ import annotations

import sys
from pathlib import Path

# =============================================================
# IMPORT PATH
# =============================================================

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import json

from data.loader import load_dataset
from data.case_normalizer import normalize_cases

from evidence.evidence_builder import (
    build_evidence_pack,
)

from evidence.evidence_registry import (
    create_evidence_registry,
)

from context.context_builder import (
    build_trends_context,
)

from llm.client import (
    GeminiClient,
)

from validation.evidence_validator import (
    validate_generated_section,
)


# =============================================================
# PATHS
# =============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

PROMPTS_DIR = BASE_DIR / "src" / "prompts"


# =============================================================
# PROMPT LOADING
# =============================================================

def load_prompt(filename: str) -> str:
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
    )


# =============================================================
# EVIDENCE SELECTION
# =============================================================

def build_trends_evidence(
    registry: dict,
) -> dict:
    """
    Select only the most relevant evidence for the
    Temporal Trends section.

    The LLM receives:
        - highest-volume month
        - lowest-volume month
        - monthly trend summary evidence
        - top observed 15-day windows

    The complete evidence registry remains available for
    deterministic validation, but only approved trend
    evidence is supplied to Gemini.
    """

    required_ids = [
        # -----------------------------------------------------
        # Monthly trend evidence
        # -----------------------------------------------------

        "EV-TREND-001",
        "EV-TREND-002",
        "EV-TREND-003",
        "EV-TREND-004",

        # -----------------------------------------------------
        # High-volume 15-day observations
        # -----------------------------------------------------

        "EV-TREND-15D-001",
        "EV-TREND-15D-002",
        "EV-TREND-15D-003",
        "EV-TREND-15D-004",
        "EV-TREND-15D-005",
    ]

    selected = {}

    for evidence_id in required_ids:

        if evidence_id not in registry:
            raise KeyError(
                f"Required trend evidence ID missing: "
                f"{evidence_id}"
            )

        selected[evidence_id] = registry[
            evidence_id
        ]

    return selected


# =============================================================
# BUILD LLM PROMPT
# =============================================================

def build_trends_prompt(
    context: dict,
    evidence: dict,
) -> str:
    """
    Build the grounded Trends prompt.

    Gemini receives:

        1. System instructions
        2. Section-specific prompt
        3. Approved evidence
        4. Section context
        5. Explicit grounding constraints
    """

    trends_prompt = load_prompt(
        "trends.txt"
    )

    system_prompt = load_prompt(
        "system.txt"
    )

    prompt = f"""
{trends_prompt}

IMPORTANT GROUNDING REQUIREMENTS:

The following evidence IDs are the ONLY approved
sources for factual claims in this response.

APPROVED TEMPORAL EVIDENCE:

{json.dumps(
    evidence,
    indent=2,
    ensure_ascii=False,
)}

SECTION CONTEXT:

{json.dumps(
    context,
    indent=2,
    ensure_ascii=False,
)}

STRICT REQUIREMENTS:

- Use only the approved evidence above.
- Every factual claim must cite one or more
  approved evidence IDs.
- Preserve numerical values exactly.
- Do not calculate new statistics.
- Do not invent dates or case counts.
- Do not call any temporal pattern a safety signal.
- Do not establish causality.
- Do not infer clinical risk.
- Treat high-volume 15-day observations as
  descriptive observations only.
- State that high-volume observations require
  human review and contextual assessment.
- Keep the response concise.
- Prefer 3 to 6 claims.
- Return only the required JSON structure.
"""

    return (
        system_prompt
        + "\n\n"
        + prompt
    )


# =============================================================
# GENERATE TRENDS
# =============================================================

def generate_trends():
    """
    Execute the complete grounded Temporal Trends
    generation pipeline.
    """

    print("=" * 70)
    print(
        "GENAR — GROUNDED TEMPORAL TRENDS GENERATION"
    )
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------

    print("\n[1/6] Loading dataset...")

    raw_df = load_dataset()

    print(
        f"      Raw rows: {len(raw_df):,}"
    )

    # ---------------------------------------------------------
    # 2. NORMALIZE CASES
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 3. BUILD DETERMINISTIC EVIDENCE
    # ---------------------------------------------------------

    print(
        "[3/6] Building evidence pack..."
    )

    evidence = build_evidence_pack(
        cases_df
    )

    print(
        "      Deterministic evidence built."
    )

    # ---------------------------------------------------------
    # 4. BUILD EVIDENCE REGISTRY
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 5. BUILD TRENDS CONTEXT
    # ---------------------------------------------------------

    print(
        "[5/6] Building Temporal Trends-specific "
        "context..."
    )

    context = build_trends_context(
        evidence
    )

    trends_evidence = (
        build_trends_evidence(
            registry
        )
    )

    print(
        "      Approved Temporal Trends evidence:"
    )

    for evidence_id in trends_evidence:
        print(
            f"        {evidence_id}"
        )

    prompt = build_trends_prompt(
        context=context,
        evidence=trends_evidence,
    )

    # ---------------------------------------------------------
    # 6. GENERATE WITH GEMINI
    # ---------------------------------------------------------

    print(
        "[6/6] Generating grounded Temporal Trends..."
    )

    client = GeminiClient()

    generated = client.generate_json(
        prompt=prompt,
        system_instruction=load_prompt(
            "system.txt"
        ),
    )

    # ---------------------------------------------------------
    # DISPLAY GENERATED RESULT
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATED TEMPORAL TRENDS")
    print("=" * 70)

    print(
        json.dumps(
            generated,
            indent=2,
            ensure_ascii=False,
        )
    )

    # ---------------------------------------------------------
    # VALIDATE
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVIDENCE VALIDATION")
    print("=" * 70)

    validation = validate_generated_section(
        generated,
        registry,
        allowed_evidence_ids=set(
            trends_evidence.keys()
        ),
    )

    if validation["valid"]:

        print("STATUS: PASS")

    else:

        print("STATUS: FAIL")

        for error in validation["errors"]:

            print(
                f"ERROR: {error}"
            )

    # ---------------------------------------------------------
    # CLAIM DETAILS
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

            for error in item["errors"]:

                print(
                    f"  ERROR: {error}"
                )

    # ---------------------------------------------------------
    # NUMERIC VALIDATION
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

    # ---------------------------------------------------------
    # FINAL STATUS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)

    if validation["valid"]:

        print(
            "FINAL STATUS: "
            "VALIDATED TEMPORAL TRENDS"
        )

    else:

        print(
            "FINAL STATUS: "
            "TEMPORAL TRENDS VALIDATION FAILED"
        )

    print("=" * 70)

    return {
        "generated": generated,
        "validation": validation,
    }


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    generate_trends()