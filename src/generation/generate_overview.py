from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# =============================================================
# PROJECT PATH SETUP
# =============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

SRC_DIR = BASE_DIR / "src"

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
    build_overview_context,
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

PROMPTS_DIR = (
    BASE_DIR
    / "src"
    / "prompts"
)

SECTION_OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "sections"
)

SECTION_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    SECTION_OUTPUT_DIR
    / "overview.json"
)


# =============================================================
# CONFIGURATION
# =============================================================

SECTION_NAME = "overview"

TEMPERATURE = 0.0

MAX_OUTPUT_TOKENS = 700


# =============================================================
# PROMPT LOADING
# =============================================================

def load_prompt(
    filename: str,
) -> str:
    """
    Load a prompt file from src/prompts.
    """

    path = (
        PROMPTS_DIR
        / filename
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}"
        )

    return path.read_text(
        encoding="utf-8"
    ).strip()


# =============================================================
# COMPACT JSON
# =============================================================

def compact_json(
    value: Any,
) -> str:
    """
    Serialize JSON with minimal whitespace.

    This reduces prompt size while preserving
    all factual content.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
    )


# =============================================================
# EVIDENCE SELECTION
# =============================================================

def build_overview_evidence(
    registry: dict,
) -> dict:
    """
    Select ONLY evidence relevant to Overview.

    The complete registry is never passed to the LLM.
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

        selected[evidence_id] = (
            registry[evidence_id]
        )

    return selected


# =============================================================
# BUILD GROUNDED LLM PROMPT
# =============================================================

def build_overview_prompt(
    context: dict,
    evidence: dict,
) -> str:
    """
    Build the grounded Overview user prompt.

    system.txt is intentionally NOT included here.

    GeminiClient receives system.txt separately.
    """

    overview_prompt = load_prompt(
        "overview.txt"
    )

    # ---------------------------------------------------------
    # Compact evidence
    # ---------------------------------------------------------

    evidence_text = compact_json(
        evidence
    )

    # ---------------------------------------------------------
    # Compact context
    #
    # Keep context because Overview-specific context may contain
    # useful scope information, but serialize it compactly.
    # ---------------------------------------------------------

    context_text = compact_json(
        context
    )

    # ---------------------------------------------------------
    # Grounding rules
    # ---------------------------------------------------------

    grounding_rules = """
STRICT OVERVIEW GROUNDING RULES:

1. Use ONLY the approved evidence supplied below.
2. Every factual claim MUST contain at least one evidence ID.
3. Do not invent facts or statistics.
4. Do not invent dates, percentages, ratios, rates, averages,
   thresholds, or derived values.
5. Every numeric value in claim text MUST appear explicitly in
   the evidence supporting that claim.
6. Do NOT write calendar dates in generated claims. Describe the
   reporting period qualitatively, such as "during the reporting period".
7. Do NOT calculate or derive percentages from case counts.
8. Do NOT introduce 0.1% for the non-serious case unless 0.1 is
   explicitly present in the supporting evidence. Prefer:
   "One case was non-serious."
9. Preserve numerical values exactly as supplied.
10. Do not establish causality.
11. Do not make a safety-signal determination.
12. Do not make an expectedness determination.
13. Do not introduce demographic or clinical interpretation.
14. Evidence IDs must be copied exactly.
15. Numbers inside evidence IDs are identifiers and must NOT be
    copied into claim text as factual numbers.
16. Return ONLY valid JSON. No markdown, code fences, explanations,
    or commentary.

REQUIRED JSON STRUCTURE:

{
  "section": "overview",
  "claims": [
    {
      "text": "grounded factual statement",
      "evidence_ids": ["EV-..."]
    }
  ]
}
"""

    # ---------------------------------------------------------
    # Build final prompt
    # ---------------------------------------------------------

    prompt = (
        overview_prompt
        + "\n\n"
        + grounding_rules
        + "\n\n"
        + "============================================================\n"
        + "APPROVED EVIDENCE\n"
        + "============================================================\n"
        + evidence_text
        + "\n\n"
        + "============================================================\n"
        + "SECTION CONTEXT\n"
        + "============================================================\n"
        + context_text
        + "\n\n"
        + "FINAL INSTRUCTION:\n"
        + "Return ONLY the required JSON object."
    )

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
    Validate generated Overview against:

    - complete evidence registry
    - approved Overview evidence IDs
    - numeric grounding
    """

    allowed_evidence_ids = set(
        overview_evidence.keys()
    )

    return validate_generated_section(
        generated=generated,
        registry=registry,
        allowed_evidence_ids=(
            allowed_evidence_ids
        ),
    )


# =============================================================
# PRINT GENERATED OVERVIEW
# =============================================================

def print_generated_overview(
    generated: dict,
) -> None:
    """
    Print generated Overview JSON.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "GENERATED OVERVIEW"
    )

    print(
        "=" * 70
    )

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

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EVIDENCE VALIDATION"
    )

    print(
        "=" * 70
    )

    if validation.get(
        "valid",
        False,
    ):

        print(
            "STATUS: PASS"
        )

    else:

        print(
            "STATUS: FAIL"
        )

        for error in validation.get(
            "errors",
            [],
        ):

            print(
                f"ERROR: {error}"
            )

    # ---------------------------------------------------------
    # Claim validation
    # ---------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CLAIM VALIDATION DETAILS"
    )

    print(
        "=" * 70
    )

    for item in validation.get(
        "evidence_validation",
        [],
    ):

        status = (
            "PASS"
            if item.get(
                "valid",
                False,
            )
            else "FAIL"
        )

        print(
            f"\nClaim {item.get('claim_index')}: "
            f"{status}"
        )

        print(
            f"Text: {item.get('claim')}"
        )

        print(
            "Evidence IDs: "
            f"{item.get('evidence_ids')}"
        )

        for error in item.get(
            "errors",
            [],
        ):

            print(
                f"  ERROR: {error}"
            )

    # ---------------------------------------------------------
    # Numeric validation
    # ---------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "NUMERIC VALIDATION"
    )

    print(
        "=" * 70
    )

    for item in validation.get(
        "number_validation",
        [],
    ):

        status = (
            "PASS"
            if item.get(
                "valid",
                False,
            )
            else "FAIL"
        )

        print(
            f"\nClaim {item.get('claim_index')}: "
            f"{status}"
        )

        print(
            "Claim numbers: "
            f"{item.get('claim_numbers')}"
        )

        print(
            "Supported numbers: "
            f"{item.get('supported_numbers')}"
        )

        unsupported = item.get(
            "unsupported_numbers",
            [],
        )

        if unsupported:

            print(
                "Unsupported numbers: "
                f"{unsupported}"
            )


# =============================================================
# SAVE VALIDATED OVERVIEW
# =============================================================

def save_validated_overview(
    generated: dict,
    validation: dict,
) -> Path:
    """
    Save ONLY the generated section and validation result.

    Internal evidence/context is deliberately not included.
    """

    result = dict(generated)
    result["validation"] = validation

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return OUTPUT_PATH


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
        Approved evidence selection
            ↓
        Gemini generation
            ↓
        Automatic Groq fallback if Gemini fails
            ↓
        Evidence validation
            ↓
        Numeric validation
            ↓
        Save validated output
    """

    print(
        "=" * 70
    )

    print(
        "GENAR — GROUNDED OVERVIEW GENERATION"
    )

    print(
        "=" * 70
    )

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
        "      Canonical cases: "
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
        "      Registry items: "
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

    # ---------------------------------------------------------
    # Build prompt
    # ---------------------------------------------------------

    prompt = build_overview_prompt(
        context=context,
        evidence=overview_evidence,
    )

    # ---------------------------------------------------------
    # Prompt diagnostics
    # ---------------------------------------------------------

    prompt_characters = len(
        prompt
    )

    estimated_tokens = (
        prompt_characters
        // 4
    )

    print(
        "\n      Prompt diagnostics:"
    )

    print(
        f"        Characters: "
        f"{prompt_characters:,}"
    )

    print(
        f"        Estimated tokens: "
        f"{estimated_tokens:,}"
    )

    # =========================================================
    # 6. GENERATE
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
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
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
        overview_evidence=(
            overview_evidence
        ),
    )

    print_validation_results(
        validation
    )

    # =========================================================
    # SAVE ONLY IF VALID
    # =========================================================

    if validation.get(
        "valid",
        False,
    ):

        output_path = (
            save_validated_overview(
                generated=generated,
                validation=validation,
            )
        )

        print(
            "\n      Saved validated section:"
        )

        print(
            f"      {output_path}"
        )

    else:

        print(
            "\n      Overview was NOT saved "
            "because validation failed."
        )

    # =========================================================
    # FINAL STATUS
    # =========================================================

    print(
        "\n"
        + "=" * 70
    )

    if validation.get(
        "valid",
        False,
    ):

        print(
            "FINAL STATUS: "
            "VALIDATED OVERVIEW"
        )

    else:

        print(
            "FINAL STATUS: "
            "OVERVIEW VALIDATION FAILED"
        )

    print(
        "=" * 70
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # Return the same compact structure that is saved.
    # This keeps direct execution and build_report.py consistent.
    # ---------------------------------------------------------

    result = dict(generated)
    result["validation"] = validation

    return result


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    generate_overview()