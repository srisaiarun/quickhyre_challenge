from __future__ import annotations

import json
import re
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

    The LLM receives:
    - reaction metadata
    - top five overall reactions
    - top five serious reactions
    - outcome distribution
    """

    required_ids = [
        # -----------------------------------------------------
        # Reaction metadata
        # -----------------------------------------------------

        "EV-REACTION-001",
        "EV-REACTION-002",

        # -----------------------------------------------------
        # Top 5 reactions
        # -----------------------------------------------------

        "EV-REACTION-TOP-001",
        "EV-REACTION-TOP-002",
        "EV-REACTION-TOP-003",
        "EV-REACTION-TOP-004",
        "EV-REACTION-TOP-005",

        # -----------------------------------------------------
        # Top 5 serious reactions
        # -----------------------------------------------------

        "EV-REACTION-SERIOUS-001",
        "EV-REACTION-SERIOUS-002",
        "EV-REACTION-SERIOUS-003",
        "EV-REACTION-SERIOUS-004",
        "EV-REACTION-SERIOUS-005",

        # -----------------------------------------------------
        # Outcomes
        # -----------------------------------------------------

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

    The prompt strictly separates:
        - factual claim text
        - evidence IDs

    Evidence IDs MUST NOT appear inside claim text because
    their numeric suffixes can otherwise be interpreted as
    factual numbers by the numeric grounding validator.
    """

    safety_prompt = load_prompt(
        "safety_findings.txt"
    )

    system_prompt = load_prompt(
        "system.txt"
    )

    prompt = f"""
{safety_prompt}

============================================================
IMPORTANT GROUNDING REQUIREMENTS
============================================================

The following evidence IDs are the ONLY approved sources
for factual claims in this response.

============================================================
APPROVED EVIDENCE
============================================================

{json.dumps(
    evidence,
    indent=2,
    ensure_ascii=False,
)}

============================================================
SECTION CONTEXT
============================================================

{json.dumps(
    context,
    indent=2,
    ensure_ascii=False,
)}

============================================================
STRICT REQUIREMENTS
============================================================

1. Use ONLY the approved evidence above.

2. Every factual claim MUST contain one or more evidence IDs
   in the "evidence_ids" array.

3. Preserve numerical values EXACTLY as supplied by the
   deterministic evidence.

4. Do NOT calculate new statistics.

5. Do NOT invent information.

6. Do NOT claim bisoprolol caused any reaction.

7. Do NOT establish causality.

8. Do NOT call any reaction a confirmed safety signal.

9. Do NOT infer expectedness.

10. Describe reaction frequencies as observed reporting data.

11. Describe outcomes as observed reporting data.

12. If discussing high-frequency reactions, describe them as
    observed frequencies/rankings only.

13. If discussing serious reactions, describe them as serious
    reaction findings in the supplied dataset.

14. Do NOT make clinical risk conclusions.

15. If evidence is insufficient to support a statement,
    DO NOT make that statement.

============================================================
CRITICAL EVIDENCE-ID RULE
============================================================

16. NEVER place an evidence ID inside the claim "text".

17. Evidence IDs MUST appear ONLY inside the "evidence_ids"
    array.

18. NEVER write evidence IDs in parentheses inside claim text.

19. NEVER write evidence IDs after a sentence inside claim text.

20. NEVER write evidence IDs as citations inside claim text.

21. NEVER write evidence IDs such as:
    EV-REACTION-001
    EV-REACTION-TOP-001
    EV-REACTION-TOP-002
    EV-REACTION-SERIOUS-001
    EV-OUTCOME-001

    inside the "text" field.

22. Evidence IDs are identifiers, not factual statistics.
    Their numeric suffixes must NOT appear in claim text.

23. The "text" field must contain ONLY the grounded factual
    statement.

24. The "evidence_ids" field is the ONLY place where evidence
    IDs should be provided.

============================================================
NUMERIC GROUNDING RULES
============================================================

25. Every number in a claim must appear explicitly in the
    supplied evidence.

26. Do NOT calculate, infer, round, estimate, or derive new
    numeric values.

27. Do NOT introduce unsupported thresholds.

28. Do NOT introduce unsupported percentages.

29. Do NOT introduce unsupported counts.

30. Do NOT convert textual rankings into numeric rankings.

31. Preserve all supplied numerical values exactly.

32. Do NOT introduce numeric values merely to describe the
    number of evidence items.

33. Words such as "top five" are acceptable only when the
    supplied evidence itself establishes that ranking.

34. Never write:
    "(EV-REACTION-001)"
    "(EV-REACTION-TOP-001, EV-REACTION-TOP-002)"
    or any similar evidence-ID notation inside claim text.

============================================================
OUTPUT REQUIREMENTS
============================================================

35. Return ONLY valid JSON.

36. Do NOT use Markdown code fences.

37. Do NOT add commentary before or after the JSON.

38. Use exactly this structure:

{{
    "section": "safety_findings",
    "claims": [
        {{
            "text": "grounded factual statement",
            "evidence_ids": [
                "EV-REACTION-..."
            ]
        }}
    ]
}}
"""

    return (
        system_prompt
        + "\n\n"
        + prompt
    )


# =============================================================
# DETERMINISTIC CLAIM CLEANUP
# =============================================================

def strip_evidence_ids_from_claim_text(
    generated: dict,
    allowed_evidence_ids: set[str],
) -> dict:
    """
    Remove evidence IDs accidentally inserted into claim text.

    This is a deterministic safety layer.

    Example:

        "The dataset contained 3,429 records
         (EV-REACTION-001)."

    becomes:

        "The dataset contained 3,429 records."

    Evidence IDs remain untouched inside the
    "evidence_ids" arrays.

    This prevents the numeric validator from interpreting
    identifiers such as "-001" as factual numbers.
    """

    if not isinstance(generated, dict):
        return generated

    claims = generated.get("claims")

    if not isinstance(claims, list):
        return generated

    # ---------------------------------------------------------
    # Build one safe regex from approved evidence IDs.
    # ---------------------------------------------------------

    escaped_ids = [
        re.escape(evidence_id)
        for evidence_id in allowed_evidence_ids
    ]

    if not escaped_ids:
        return generated

    evidence_pattern = re.compile(
        r"\b(?:"
        + "|".join(escaped_ids)
        + r")\b",
        flags=re.IGNORECASE,
    )

    for claim in claims:

        if not isinstance(claim, dict):
            continue

        text = claim.get("text")

        if not isinstance(text, str):
            continue

        # -----------------------------------------------------
        # Remove evidence IDs from claim text.
        # -----------------------------------------------------

        cleaned = evidence_pattern.sub(
            "",
            text,
        )

        # -----------------------------------------------------
        # Clean common formatting left behind after removal.
        # -----------------------------------------------------

        cleaned = re.sub(
            r"\(\s*,\s*",
            "(",
            cleaned,
        )

        cleaned = re.sub(
            r",\s*\)",
            ")",
            cleaned,
        )

        cleaned = re.sub(
            r"\(\s*\)",
            "",
            cleaned,
        )

        cleaned = re.sub(
            r"\s{2,}",
            " ",
            cleaned,
        )

        cleaned = re.sub(
            r"\s+([,.;:])",
            r"\1",
            cleaned,
        )

        cleaned = re.sub(
            r"([(\[])\s+",
            r"\1",
            cleaned,
        )

        cleaned = re.sub(
            r"\s+([)\]])",
            r"\1",
            cleaned,
        )

        claim["text"] = cleaned.strip()

    return generated


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
        "\n[2/6] Normalizing cases..."
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
        "\n[3/6] Building evidence pack..."
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
        "\n[4/6] Building evidence registry..."
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
        "\n[5/6] Building Safety Findings-specific context..."
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
    # 6. GENERATE WITH GEMINI / GROQ FALLBACK
    # ---------------------------------------------------------

    print(
        "\n[6/6] Generating grounded Safety Findings..."
    )

    client = GeminiClient()

    generated = client.generate_json(
        prompt=prompt,
        system_instruction=load_prompt(
            "system.txt"
        ),
    )

    # ---------------------------------------------------------
    # DETERMINISTIC POST-PROCESSING
    # ---------------------------------------------------------

    print(
        "\n      Applying deterministic evidence-ID cleanup..."
    )

    generated = strip_evidence_ids_from_claim_text(
        generated=generated,
        allowed_evidence_ids=set(
            safety_evidence.keys()
        ),
    )

    # ---------------------------------------------------------
    # DISPLAY RAW / CLEANED GENERATION
    # ---------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "GENERATED SAFETY FINDINGS"
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

    # ---------------------------------------------------------
    # VALIDATE
    # ---------------------------------------------------------

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

    validation = validate_generated_section(
        generated,
        registry,
        allowed_evidence_ids=set(
            safety_evidence.keys()
        ),
    )

    if validation["valid"]:

        print(
            "STATUS: PASS"
        )

    else:

        print(
            "STATUS: FAIL"
        )

        for error in validation[
            "errors"
        ]:

            print(
                f"ERROR: {error}"
            )

    # ---------------------------------------------------------
    # CLAIM DETAILS
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
    # SAVE VALIDATED SECTION
    # ---------------------------------------------------------

    if validation["valid"]:

        output_dir = BASE_DIR / "outputs" / "sections"
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_dir / "safety_findings.json"
        )

        section_output = {
            "section": "safety_findings",
            "claims": generated.get(
                "claims",
                [],
            ),
            "validation": validation,
        }

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                section_output,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"\n      Saved validated section: "
            f"{output_path}"
        )

    # ---------------------------------------------------------
    # FINAL STATUS
    # ---------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

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

    print(
        "=" * 70
    )

    return {
        "section": "safety_findings",
        "claims": generated.get(
            "claims",
            [],
        ),
        "validation": validation,
    }


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    generate_safety_findings()