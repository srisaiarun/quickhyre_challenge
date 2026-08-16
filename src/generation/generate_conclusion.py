from __future__ import annotations

import sys
import json
import re
from pathlib import Path
from typing import Any


# =============================================================
# MAKE SRC IMPORTABLE WHEN RUN DIRECTLY
# =============================================================

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# =============================================================
# IMPORT PROJECT MODULES
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


# =============================================================
# CONSTANTS
# =============================================================

SECTION_NAME = "conclusion"

OUTPUT_FILENAME = "conclusion.json"

MAX_OUTPUT_TOKENS = 500

TEMPERATURE = 0.0


# =============================================================
# SAVE SECTION OUTPUT
# =============================================================

def save_section_output(
    result: dict,
    filename: str,
) -> Path:
    """
    Save a validated section result.
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
# EVIDENCE SELECTION
# =============================================================

def build_conclusion_evidence(
    registry: dict,
) -> dict:
    """
    Select only evidence required by the Conclusion section.

    The evidence remains deterministic and controlled.
    """

    required_ids = [

        # -----------------------------------------------------
        # Overall case volume
        # -----------------------------------------------------

        "EV-CASE-001",

        # -----------------------------------------------------
        # Seriousness
        # -----------------------------------------------------

        "EV-CASE-002",

        # -----------------------------------------------------
        # Leading reported reactions
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
        # Temporal observations
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

        selected[evidence_id] = (
            registry[evidence_id]
        )

    return selected


# =============================================================
# COMPACT OBJECT
# =============================================================

def compact_object(
    value: Any,
) -> Any:
    """
    Recursively compact an object while preserving its factual
    information.

    This removes only presentation/metadata fields that are not
    required for grounding.

    The important evidence IDs and factual values are preserved.
    """

    # ---------------------------------------------------------
    # Dictionaries
    # ---------------------------------------------------------

    if isinstance(
        value,
        dict,
    ):

        result = {}

        # Metadata that can safely be omitted from the LLM prompt.
        removable_keys = {
            "source",
            "source_file",
            "source_path",
            "created_at",
            "updated_at",
            "metadata",
            "provenance",
            "debug",
            "debug_info",
            "internal",
            "notes_internal",
        }

        for key, item in value.items():

            if key in removable_keys:
                continue

            result[key] = compact_object(
                item
            )

        return result

    # ---------------------------------------------------------
    # Lists
    # ---------------------------------------------------------

    if isinstance(
        value,
        list,
    ):

        return [
            compact_object(item)
            for item in value
        ]

    # ---------------------------------------------------------
    # Everything else
    # ---------------------------------------------------------

    return value


# =============================================================
# COMPACT JSON
# =============================================================

def compact_json(
    value: Any,
) -> str:
    """
    Serialize JSON with minimum whitespace.
    """

    return json.dumps(
        compact_object(value),
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
    )


# =============================================================
# BUILD CONCLUSION PROMPT
# =============================================================

def build_conclusion_prompt(
    context: dict,
    evidence: dict,
) -> str:
    """
    Build a compact grounded Conclusion prompt.

    IMPORTANT:

    The previous implementation inserted system.txt into this
    prompt AND passed system.txt separately to GeminiClient.

    That duplicated a large amount of text.

    This implementation intentionally does NOT include
    system.txt here.

    GeminiClient receives system.txt separately.
    """

    conclusion_prompt = load_prompt(
        "conclusion.txt"
    )

    # ---------------------------------------------------------
    # COMPACT EVIDENCE
    # ---------------------------------------------------------

    evidence_text = compact_json(
        evidence
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT send the complete context object.
    #
    # build_conclusion_context() contains information that is
    # already represented by the deterministic evidence registry.
    #
    # Sending both caused unnecessary token duplication.
    #
    # We keep only a very small context marker.
    # ---------------------------------------------------------

    context_marker = {
        "section": "conclusion",
        "grounding_source": (
            "approved_evidence_registry"
        ),
        "demographic_evidence": (
            "not used"
        ),
    }

    context_text = compact_json(
        context_marker
    )

    # ---------------------------------------------------------
    # STRICT RULES
    # ---------------------------------------------------------

    strict_rules = """
STRICT CONCLUSION RULES:

1. Use ONLY the approved evidence supplied below.

2. Every factual claim MUST contain at least one evidence ID.

3. Do not invent facts.

4. Do not introduce new analysis.

5. Do not calculate percentages.

6. Do not calculate counts.

7. Do not calculate ratios.

8. Do not calculate averages.

9. Do not calculate rates.

10. Do not derive numbers from other numbers.

11. Every numeric value appearing in a claim MUST appear
    explicitly in the evidence supporting that claim.

12. Preserve supplied numerical values exactly.

13. Do not introduce calendar dates.

14. Do not introduce date ranges.

15. Do not introduce numeric temporal durations.

16. Do not establish causality.

17. Do not declare a safety signal.

18. Do not make an expectedness determination.

19. Do not introduce demographic findings.

20. Do not introduce reactions not present in the evidence.

21. Temporal observations must remain descriptive.

22. If temporal evidence indicates elevated observations,
    describe them conservatively and state that they require
    human review/contextual assessment only when supported.

23. Evidence IDs must be copied exactly.

24. Evidence IDs may contain numbers.

25. Numbers contained inside evidence IDs do NOT count as
    claim numbers.

26. Never copy "15D" from an evidence ID into claim text.

27. Never write "15-day", "15 day", "15-day window",
    "15 day window", "15-day windows", or "15 day windows".

28. For temporal findings use phrases such as:
    "observed temporal windows",
    "analyzed temporal windows",
    "observed windows",
    "analyzed windows".

29. Do not introduce the number 15 as a temporal duration.

30. Return ONLY valid JSON.

31. Do not return markdown fences.

32. Do not return explanations.

33. Do not return commentary.

34. Use exactly this JSON structure:

{
  "section": "conclusion",
  "claims": [
    {
      "text": "grounded factual statement",
      "evidence_ids": ["EV-..."]
    }
  ]
}
"""

    # ---------------------------------------------------------
    # FINAL PROMPT
    # ---------------------------------------------------------

    prompt = (
        conclusion_prompt
        + "\n\n"
        + strict_rules
        + "\n\n"
        + "APPROVED EVIDENCE:\n"
        + evidence_text
        + "\n\n"
        + "SECTION CONTEXT:\n"
        + context_text
        + "\n\n"
        + "FINAL INSTRUCTION:\n"
        + "Return ONLY the required JSON object."
    )

    return prompt.strip()


# =============================================================
# PRINT VALIDATION
# =============================================================

def print_validation(
    validation: dict,
) -> None:
    """
    Print detailed validation information.
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
# GENERATE CONCLUSION
# =============================================================

def generate_conclusion():
    """
    Execute the complete grounded Conclusion pipeline.
    """

    print(
        "=" * 70
    )

    print(
        "GENAR — GROUNDED CONCLUSION GENERATION"
    )

    print(
        "=" * 70
    )

    # =========================================================
    # 1. LOAD DATA
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
    # 5. BUILD CONCLUSION CONTEXT
    # =========================================================

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

    # ---------------------------------------------------------
    # Build compact prompt
    # ---------------------------------------------------------

    prompt = build_conclusion_prompt(
        context=context,
        evidence=conclusion_evidence,
    )

    # ---------------------------------------------------------
    # Prompt diagnostics
    # ---------------------------------------------------------

    prompt_chars = len(
        prompt
    )

    estimated_tokens = (
        prompt_chars
        // 4
    )

    print(
        "\n      Conclusion prompt size:"
    )

    print(
        f"        Characters: "
        f"{prompt_chars:,}"
    )

    print(
        f"        Estimated tokens: "
        f"{estimated_tokens:,}"
    )

    if estimated_tokens > 6500:

        print(
            "\n      WARNING:"
        )

        print(
            "      Conclusion prompt is still large."
        )

        print(
            "      The client may need to compact it further."
        )

    # =========================================================
    # 6. GENERATE
    # =========================================================

    print(
        "\n[6/6] Generating grounded Conclusion..."
    )

    client = GeminiClient()

    try:

        generated = client.generate_json(
            prompt=prompt,
            system_instruction=load_prompt(
                "system.txt"
            ),
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

    except Exception as exc:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "CONCLUSION GENERATION FAILED"
        )

        print(
            "=" * 70
        )

        print(
            f"\nError: {exc}"
        )

        raise

    # =========================================================
    # DISPLAY GENERATED CONCLUSION
    # =========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "GENERATED CONCLUSION"
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

    # =========================================================
    # VALIDATE
    # =========================================================

    validation = validate_generated_section(
        generated,
        registry,
        allowed_evidence_ids=set(
            conclusion_evidence.keys()
        ),
    )

    print_validation(
        validation
    )

    # =========================================================
    # ADDITIONAL CONCLUSION-SPECIFIC SAFETY CHECK
    # =========================================================
    #
    # The normal validator checks evidence and numeric grounding.
    # We additionally reject forbidden temporal wording in the
    # generated Conclusion.
    #

    forbidden_temporal_patterns = [
        r"\b15[- ]day\b",
        r"\b15[- ]day\s+window",
        r"\b15[- ]day\s+windows",
        r"\b15D\b",
        r"\b15\s+D\b",
    ]

    temporal_violations = []

    claims = generated.get(
        "claims",
        [],
    )

    for index, claim in enumerate(
        claims,
        start=1,
    ):

        text = str(
            claim.get(
                "text",
                ""
            )
        )

        for pattern in forbidden_temporal_patterns:

            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):

                temporal_violations.append(
                    (
                        index,
                        text,
                        pattern,
                    )
                )

    if temporal_violations:

        validation["valid"] = False

        validation.setdefault(
            "errors",
            [],
        ).append(
            "Conclusion contains forbidden numeric "
            "temporal duration wording."
        )

        print(
            "\nERROR: Forbidden temporal wording detected."
        )

        for (
            index,
            text,
            pattern,
        ) in temporal_violations:

            print(
                f"  Claim {index}: "
                f"matched {pattern!r}"
            )

    # =========================================================
    # BUILD FINAL RESULT
    # =========================================================

    result = {
        "section": "conclusion",
        "claims": generated.get(
            "claims",
            [],
        ),
        "validation": validation,
    }

    # =========================================================
    # SAVE ONLY VALIDATED OUTPUT
    # =========================================================

    if validation.get(
        "valid",
        False,
    ):

        output_path = save_section_output(
            result=result,
            filename=OUTPUT_FILENAME,
        )

        print(
            "\n      Saved validated section:"
        )

        print(
            f"      {output_path}"
        )

    else:

        print(
            "\n      Conclusion was NOT saved "
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
            "VALIDATED CONCLUSION"
        )

    else:

        print(
            "FINAL STATUS: "
            "CONCLUSION VALIDATION FAILED"
        )

    print(
        "=" * 70
    )

    return result


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    generate_conclusion()