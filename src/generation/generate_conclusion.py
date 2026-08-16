from __future__ import annotations

import sys
import json
from pathlib import Path

# =============================================================
# MAKE SRC IMPORTABLE WHEN RUN DIRECTLY
# =============================================================

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from data.loader import load_dataset
from data.case_normalizer import normalize_cases

from evidence.evidence_builder import (
    build_evidence_pack,
)

from evidence.evidence_registry import (
    create_evidence_registry,
)

from context.context_builder import (
    build_conclusion_context,
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

def build_conclusion_evidence(
    registry: dict,
) -> dict:
    """
    Select ONLY evidence relevant to the Conclusion section.

    The conclusion receives a compact, controlled subset of
    the complete evidence registry.

    Approved evidence covers:

    - overall case volume
    - seriousness
    - leading reactions
    - serious reactions
    - important temporal observations
    - analytical limitations
    """

    required_ids = [
        # -----------------------------------------------------
        # Overall case volume / seriousness
        # -----------------------------------------------------
        "EV-CASE-001",
        "EV-CASE-002",

        # -----------------------------------------------------
        # Leading reactions
        # -----------------------------------------------------
        "EV-REACTION-TOP-001",
        "EV-REACTION-TOP-002",
        "EV-REACTION-TOP-003",
        "EV-REACTION-TOP-004",
        "EV-REACTION-TOP-005",

        # -----------------------------------------------------
        # Leading serious reactions
        # -----------------------------------------------------
        "EV-REACTION-SERIOUS-001",
        "EV-REACTION-SERIOUS-002",
        "EV-REACTION-SERIOUS-003",
        "EV-REACTION-SERIOUS-004",
        "EV-REACTION-SERIOUS-005",

        # -----------------------------------------------------
        # Important temporal observations
        # -----------------------------------------------------
        "EV-TREND-001",
        "EV-TREND-002",
        "EV-TREND-003",
        "EV-TREND-004",

        "EV-TREND-15D-001",
        "EV-TREND-15D-002",
        "EV-TREND-15D-003",
        "EV-TREND-15D-004",
        "EV-TREND-15D-005",

        # -----------------------------------------------------
        # Analytical limitations
        # -----------------------------------------------------
        "EV-LIMIT-001",
    ]

    selected = {}

    for evidence_id in required_ids:

        if evidence_id not in registry:
            raise KeyError(
                "Required Conclusion evidence ID missing: "
                f"{evidence_id}"
            )

        selected[evidence_id] = registry[
            evidence_id
        ]

    return selected


# =============================================================
# BUILD LLM PROMPT
# =============================================================

def build_conclusion_prompt(
    context: dict,
    evidence: dict,
) -> str:
    """
    Build the complete grounded Conclusion prompt.

    Gemini receives:

    1. Conclusion instructions.
    2. Approved evidence only.
    3. Section-specific context.
    4. Strict grounding requirements.
    """

    conclusion_prompt = load_prompt(
        "conclusion.txt"
    )

    system_prompt = load_prompt(
        "system.txt"
    )

    prompt = f"""
{conclusion_prompt}

IMPORTANT GROUNDING REQUIREMENTS:

The following evidence IDs are the ONLY approved
sources for factual claims in this response.

APPROVED EVIDENCE:

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

CONCLUSION-SPECIFIC RULES:

- Summarize the observed evidence only.
- Do not introduce new calculations.
- Do not calculate new percentages.
- Do not calculate new statistics.
- Do not introduce demographic findings.
- Do not introduce reactions that are not present in the
  approved evidence.
- Do not introduce dates that are not present in the
  approved evidence.
- Do not establish causality.
- Do not declare a safety signal.
- Do not make an expectedness determination.
- Temporal observations must remain descriptive.
- State that high-volume temporal observations require
  human review and contextual assessment where appropriate.
- Preserve numerical values exactly as supplied.
- Every factual claim must contain one or more evidence IDs.
- Evidence IDs must be copied exactly.
- Use only the approved evidence IDs above.
- Keep the conclusion concise and suitable for a
  pharmacovigilance report.
- Return ONLY the required JSON structure.
"""

    return (
        system_prompt
        + "\n\n"
        + prompt
    )


# =============================================================
# GENERATE CONCLUSION
# =============================================================

def generate_conclusion():
    """
    Execute the complete grounded Conclusion generation
    pipeline.
    """

    print("=" * 70)
    print("GENAR — GROUNDED CONCLUSION GENERATION")
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

    print("[2/6] Normalizing cases...")

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

    print("[3/6] Building evidence pack...")

    evidence = build_evidence_pack(
        cases_df
    )

    print(
        "      Deterministic evidence built."
    )

    # ---------------------------------------------------------
    # 4. BUILD EVIDENCE REGISTRY
    # ---------------------------------------------------------

    print("[4/6] Building evidence registry...")

    registry = create_evidence_registry(
        evidence
    )

    print(
        f"      Registry items: "
        f"{len(registry)}"
    )

    # ---------------------------------------------------------
    # 5. BUILD CONCLUSION CONTEXT
    # ---------------------------------------------------------

    print(
        "[5/6] Building Conclusion-specific context..."
    )

    context = build_conclusion_context(
        evidence
    )

    conclusion_evidence = (
        build_conclusion_evidence(
            registry
        )
    )

    print(
        "      Approved Conclusion evidence:"
    )

    for evidence_id in conclusion_evidence:
        print(
            f"        {evidence_id}"
        )

    prompt = build_conclusion_prompt(
        context=context,
        evidence=conclusion_evidence,
    )

    # ---------------------------------------------------------
    # 6. GENERATE WITH GEMINI
    # ---------------------------------------------------------

    print(
        "[6/6] Generating grounded Conclusion..."
    )

    client = GeminiClient()

    generated = client.generate_json(
        prompt=prompt,
        system_instruction=load_prompt(
            "system.txt"
        ),
    )

    # ---------------------------------------------------------
    # DISPLAY GENERATED CONCLUSION
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATED CONCLUSION")
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
            conclusion_evidence.keys()
        ),
    )

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
            "VALIDATED CONCLUSION"
        )

    else:

        print(
            "FINAL STATUS: "
            "CONCLUSION VALIDATION FAILED"
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
    generate_conclusion()