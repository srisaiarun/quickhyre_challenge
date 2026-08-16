from __future__ import annotations

import json
import sys
from pathlib import Path


# =============================================================
# IMPORT PATH
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
    build_limitations_context,
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
    )


# =============================================================
# EVIDENCE SELECTION
# =============================================================

def build_limitations_evidence(
    registry: dict,
) -> dict:
    """
    Select only the evidence approved for the
    Limitations section.
    """

    required_ids = [
        "EV-LIMIT-001",
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
# PROMPT BUILDING
# =============================================================

def build_limitations_prompt(
    context: dict,
    evidence: dict,
) -> str:
    """
    Build a grounded Limitations prompt.

    Gemini receives:

    1. System instructions.
    2. Limitations-specific task instructions.
    3. Approved evidence only.
    4. Section context.
    """

    limitations_prompt = load_prompt(
        "limitations.txt"
    )

    system_prompt = load_prompt(
        "system.txt"
    )

    prompt = f"""
{limitations_prompt}

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

REMEMBER:

- Use ONLY the approved evidence.
- Every factual claim MUST cite one or more evidence IDs.
- Do not introduce additional limitations.
- Do not invent missing information.
- Do not infer clinical risk.
- Do not establish causality.
- Do not make expectedness determinations.
- Preserve the supplied limitation wording and meaning.
- Return ONLY the required JSON structure.
"""

    return (
        system_prompt
        + "\n\n"
        + prompt
    )


# =============================================================
# GENERATE LIMITATIONS
# =============================================================

def generate_limitations():
    """
    Execute the complete grounded Limitations
    generation pipeline.
    """

    print("=" * 70)
    print(
        "GENAR — GROUNDED LIMITATIONS GENERATION"
    )
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------

    print(
        "\n[1/6] Loading dataset..."
    )

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
    # 3. BUILD EVIDENCE
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
    # 4. BUILD REGISTRY
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
    # 5. BUILD CONTEXT
    # ---------------------------------------------------------

    print(
        "[5/6] Building Limitations-specific context..."
    )

    context = build_limitations_context(
        evidence
    )

    limitations_evidence = (
        build_limitations_evidence(
            registry
        )
    )

    print(
        "      Approved Limitations evidence:"
    )

    for evidence_id in (
        limitations_evidence.keys()
    ):
        print(
            f"        {evidence_id}"
        )

    prompt = build_limitations_prompt(
        context=context,
        evidence=limitations_evidence,
    )

    # ---------------------------------------------------------
    # 6. GENERATE
    # ---------------------------------------------------------

    print(
        "[6/6] Generating grounded Limitations..."
    )

    client = GeminiClient()

    generated = client.generate_json(
        prompt=prompt,
        system_instruction=load_prompt(
            "system.txt"
        ),
        max_output_tokens=3000,
    )

    # =========================================================
    # GENERATED OUTPUT
    # =========================================================

    print("\n" + "=" * 70)
    print("GENERATED LIMITATIONS")
    print("=" * 70)

    print(
        json.dumps(
            generated,
            indent=2,
            ensure_ascii=False,
        )
    )

    # =========================================================
    # VALIDATION
    # =========================================================

    print("\n" + "=" * 70)
    print("EVIDENCE VALIDATION")
    print("=" * 70)

    validation = validate_generated_section(
        generated,
        registry,
        allowed_evidence_ids=set(
            limitations_evidence.keys()
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

    # =========================================================
    # CLAIM VALIDATION DETAILS
    # =========================================================

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

    # =========================================================
    # NUMERIC VALIDATION
    # =========================================================

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

    # =========================================================
    # FINAL STATUS
    # =========================================================

    print("\n" + "=" * 70)

    if validation["valid"]:

        print(
            "FINAL STATUS: "
            "VALIDATED LIMITATIONS"
        )

    else:

        print(
            "FINAL STATUS: "
            "LIMITATIONS VALIDATION FAILED"
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
    generate_limitations()