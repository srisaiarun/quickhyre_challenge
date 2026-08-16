from __future__ import annotations

import json
import sys
from pathlib import Path


# =============================================================
# PROJECT PATH
# =============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# =============================================================
# PROJECT IMPORTS
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
    build_safety_findings_context,
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

def build_safety_findings_evidence(
    registry: dict,
) -> dict:
    """
    Select a compact, high-value evidence set for the
    Safety Findings section.

    The LLM receives the leading observed reactions,
    leading serious reactions, and outcome distribution.
    """

    required_ids = [
        # Reaction metadata
        "EV-REACTION-001",
        "EV-REACTION-002",

        # Top 5 reactions
        "EV-REACTION-TOP-001",
        "EV-REACTION-TOP-002",
        "EV-REACTION-TOP-003",
        "EV-REACTION-TOP-004",
        "EV-REACTION-TOP-005",

        # Top 5 serious reactions
        "EV-REACTION-SERIOUS-001",
        "EV-REACTION-SERIOUS-002",
        "EV-REACTION-SERIOUS-003",
        "EV-REACTION-SERIOUS-004",
        "EV-REACTION-SERIOUS-005",

        # Outcomes
        "EV-OUTCOME-001",
    ]

    selected = {}

    for evidence_id in required_ids:

        if evidence_id not in registry:
            raise KeyError(
                f"Required evidence ID missing: "
                f"{evidence_id}"
            )

        selected[evidence_id] = registry[
            evidence_id
        ]

    return selected


# =============================================================
# BUILD LLM PROMPT
# =============================================================

def build_safety_findings_prompt(
    context: dict,
    evidence: dict,
) -> str:
    """
    Build the grounded Safety Findings prompt.

    Gemini receives:

    1. System instructions
    2. Section instructions
    3. Approved evidence
    4. Section-specific context
    """

    safety_prompt = load_prompt(
        "safety_findings.txt"
    )

    system_prompt = load_prompt(
        "system.txt"
    )

    prompt = f"""
{safety_prompt}

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

STRICT REQUIREMENTS:

- Use only the approved evidence above.
- Every factual claim MUST cite one or more evidence IDs.
- Preserve numerical values exactly as supplied.
- Do not calculate new statistics.
- Do not invent information.
- Do not claim bisoprolol caused any reaction.
- Do not establish causality.
- Do not call any reaction a confirmed safety signal.
- Do not infer expectedness.
- Describe reaction frequencies as observed reporting data.
- Describe outcomes as observed reporting data.
- If discussing high-frequency reactions, describe them as
  observed frequencies/rankings only.
- If discussing serious reactions, describe them as
  serious reaction findings in the supplied dataset.
- Do not make clinical risk conclusions.
- Return ONLY the required JSON structure.
"""

    return (
        system_prompt
        + "\n\n"
        + prompt
    )


# =============================================================
# GENERATE SAFETY FINDINGS
# =============================================================

def generate_safety_findings():
    """
    Execute the complete grounded Safety Findings
    generation pipeline.
    """

    print("=" * 70)
    print(
        "GENAR — GROUNDED SAFETY FINDINGS GENERATION"
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
    # 5. BUILD SECTION CONTEXT
    # ---------------------------------------------------------

    print(
        "[5/6] Building Safety Findings-specific context..."
    )

    context = build_safety_findings_context(
        evidence
    )

    safety_evidence = (
        build_safety_findings_evidence(
            registry
        )
    )

    print(
        "      Approved Safety Findings evidence:"
    )

    for evidence_id in safety_evidence:
        print(
            f"        {evidence_id}"
        )

    prompt = build_safety_findings_prompt(
        context=context,
        evidence=safety_evidence,
    )

    # ---------------------------------------------------------
    # 6. GENERATE WITH GEMINI
    # ---------------------------------------------------------

    print(
        "[6/6] Generating grounded Safety Findings..."
    )

    client = GeminiClient()

    generated = client.generate_json(
        prompt=prompt,
        system_instruction=load_prompt(
            "system.txt"
        ),
    )

    # ---------------------------------------------------------
    # DISPLAY RAW GENERATION
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATED SAFETY FINDINGS")
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
            safety_evidence.keys()
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
            "VALIDATED SAFETY FINDINGS"
        )

    else:

        print(
            "FINAL STATUS: "
            "SAFETY FINDINGS VALIDATION FAILED"
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
    generate_safety_findings()