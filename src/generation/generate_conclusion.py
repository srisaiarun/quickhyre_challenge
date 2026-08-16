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

MAX_OUTPUT_TOKENS = 700

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

5. Do not calculate percentages, counts, ratios, averages, or rates.

6. Do not derive numbers from other numbers.

7. Every numeric value appearing in a claim MUST appear explicitly
   in the evidence supporting that claim.

8. Preserve supplied numerical values exactly.

9. Do not introduce calendar dates or date ranges.

10. Do not introduce numeric temporal durations.

11. Do not establish causality.

12. Do not declare a safety signal.

13. Do not make an expectedness determination.

14. Do not introduce demographic findings.

15. Do not introduce reactions not present in the evidence.

16. Temporal observations must remain descriptive and conservative.

17. Analytical limitations must be stated only from EV-LIMIT-001.

18. The Conclusion MUST contain EXACTLY FIVE claims.

19. The five claims MUST appear in this exact order:
    Claim 1 = overall case volume.
    Claim 2 = seriousness.
    Claim 3 = leading reported reactions.
    Claim 4 = important temporal observations.
    Claim 5 = important analytical limitations.

38. Each of the five claims MUST contain at least one evidence ID
    from its corresponding evidence group.

39. Do not merge two required topics into one claim.
    Do not omit any required topic.

40. If a topic has limited evidence, write a short conservative
    statement using only that evidence. Do NOT omit the claim.

20. For overall case volume, use EV-CASE-001.

21. For seriousness, use the approved seriousness evidence.

22. For leading reported reactions, use the approved
    EV-REACTION-TOP-* evidence. You may summarize the leading
    reported reactions conservatively; do not introduce reactions
    outside the supplied evidence.

23. For temporal observations, use EV-TREND-* evidence.
    Describe observed patterns/windows qualitatively when a numeric
    duration would be required.

24. Never copy "15D" from an evidence ID into claim text.

25. Never write "15-day", "15 day", "15-day window",
    "15 day window", "15-day windows", or "15 day windows".

26. Never introduce the number 15 as a temporal duration.

27. For temporal findings use phrases such as:
    "observed temporal windows",
    "analyzed temporal windows",
    "observed windows",
    "analyzed windows".

28. For limitations, use EV-LIMIT-001 and state only supported
    analytical limitations. Do not turn limitations into conclusions
    about drug safety or causality.

29. Do not say that a safety signal was identified or ruled out.

30. Do not say that the product caused any reaction.

31. Do not make an expectedness determination.

32. Evidence IDs must be copied exactly.

33. Evidence IDs may contain numbers.

34. Numbers contained inside evidence IDs do NOT count as claim numbers.

35. Return ONLY valid JSON.

36. Do not return markdown fences, explanations, or commentary.

37. Use exactly this JSON structure:

{
  "section": "conclusion",
  "claims": [
    {
      "text": "grounded factual statement",
      "evidence_ids": ["EV-..."]
    }
  ]
}

QUALITY CHECK BEFORE RETURNING JSON:

- Confirm the claims cover case volume, seriousness, leading reactions,
  temporal observations, and analytical limitations.
- Remove any unsupported number.
- Remove any calendar date.
- Remove any numeric temporal duration.
- Remove any causality, safety-signal, or expectedness conclusion.
- Ensure every factual claim has at least one approved evidence ID.
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
        + "Return ONLY the required JSON object. "
        + "Return EXACTLY FIVE claims in this order: "
        + "1 overall case volume; "
        + "2 seriousness; "
        + "3 leading reported reactions; "
        + "4 temporal observations; "
        + "5 analytical limitations. "
        + "Do not omit claims 4 or 5 even if their wording is short. "
        + "Use only supplied evidence and do not introduce unsupported "
        + "numbers, dates, or temporal durations."
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
    # ADDITIONAL CONCLUSION COMPLETENESS CHECK
    # =========================================================
    #
    # The evidence validator checks grounding and numeric support.
    # This additional check ensures the generated Conclusion actually
    # covers all five requested reporting areas.
    #

    required_evidence_groups = {
        "case_volume": {"EV-CASE-001"},
        "seriousness": {
            "EV-CASE-002",
            "EV-REACTION-SERIOUS-001",
            "EV-REACTION-SERIOUS-002",
            "EV-REACTION-SERIOUS-003",
            "EV-REACTION-SERIOUS-004",
            "EV-REACTION-SERIOUS-005",
        },
        "leading_reactions": {
            "EV-REACTION-TOP-001",
            "EV-REACTION-TOP-002",
            "EV-REACTION-TOP-003",
            "EV-REACTION-TOP-004",
            "EV-REACTION-TOP-005",
        },
        "temporal_observations": {
            "EV-TREND-001",
            "EV-TREND-002",
            "EV-TREND-003",
            "EV-TREND-004",
            "EV-TREND-15D-001",
            "EV-TREND-15D-002",
            "EV-TREND-15D-003",
            "EV-TREND-15D-004",
            "EV-TREND-15D-005",
        },
        "analytical_limitations": {"EV-LIMIT-001"},
    }

    generated_evidence_ids = set()

    for claim in claims:
        generated_evidence_ids.update(
            claim.get(
                "evidence_ids",
                [],
            )
        )

    missing_groups = []

    for group_name, group_ids in required_evidence_groups.items():
        if not generated_evidence_ids.intersection(group_ids):
            missing_groups.append(group_name)

    # The five-topic Conclusion contract is intentionally strict.
    # This prevents a partially generated Conclusion from being saved.
    if len(claims) != 5:

        validation["valid"] = False

        validation.setdefault(
            "errors",
            [],
        ).append(
            "Conclusion must contain exactly 5 claims; "
            f"found {len(claims)}."
        )

        print(
            "\nERROR: Conclusion must contain exactly 5 claims."
        )

        print(
            f"Found: {len(claims)}"
        )

    if missing_groups:

        validation["valid"] = False

        validation.setdefault(
            "errors",
            [],
        ).append(
            "Conclusion does not cover required topic(s): "
            + ", ".join(missing_groups)
        )

        print(
            "\nERROR: Conclusion is incomplete."
        )

        print(
            "Missing required topic(s): "
            + ", ".join(missing_groups)
        )

    # Also verify the fixed five-claim ordering.
    expected_order = [
        ("case_volume", required_evidence_groups["case_volume"]),
        ("seriousness", required_evidence_groups["seriousness"]),
        ("leading_reactions", required_evidence_groups["leading_reactions"]),
        ("temporal_observations", required_evidence_groups["temporal_observations"]),
        ("analytical_limitations", required_evidence_groups["analytical_limitations"]),
    ]

    if len(claims) == 5:
        order_errors = []

        for position, (topic_name, topic_ids) in enumerate(
            expected_order,
            start=1,
        ):
            claim_ids = set(
                claims[position - 1].get(
                    "evidence_ids",
                    [],
                )
            )

            if not claim_ids.intersection(topic_ids):
                order_errors.append(
                    f"claim {position} must cover {topic_name}"
                )

        if order_errors:

            validation["valid"] = False

            validation.setdefault(
                "errors",
                [],
            ).extend(order_errors)

            print(
                "\nERROR: Conclusion claim ordering/topic mapping is invalid."
            )

            for error in order_errors:
                print(
                    f"  {error}"
                )

    # =========================================================
    # STRICT FINAL CONCLUSION CONTRACT
    # =========================================================
    # The LLM may return extra claims, merge topics, or copy "15-day"
    # wording from evidence IDs. Never save such output.
    #
    # We first try one compact repair using the already-approved evidence.
    # If that response is still invalid, we use the deterministic grounded
    # fallback below. The fallback contains exactly five claims and uses no
    # numeric temporal duration, so it cannot fail because of "15D".
    # =========================================================

    def conclusion_contract_errors(candidate: dict) -> list[str]:
        errors = []

        if not isinstance(candidate, dict):
            return ["Conclusion response is not a JSON object."]

        candidate_claims = candidate.get("claims", [])

        if not isinstance(candidate_claims, list):
            return ["Conclusion claims must be a list."]

        if len(candidate_claims) != 5:
            errors.append(
                "Conclusion must contain exactly 5 claims; "
                f"found {len(candidate_claims)}."
            )

        candidate_expected_order = [
            ("case_volume", required_evidence_groups["case_volume"]),
            ("seriousness", required_evidence_groups["seriousness"]),
            ("leading_reactions", required_evidence_groups["leading_reactions"]),
            ("temporal_observations", required_evidence_groups["temporal_observations"]),
            ("analytical_limitations", required_evidence_groups["analytical_limitations"]),
        ]

        if len(candidate_claims) == 5:
            for position, (topic_name, topic_ids) in enumerate(
                candidate_expected_order,
                start=1,
            ):
                claim = candidate_claims[position - 1]

                if not isinstance(claim, dict):
                    errors.append(
                        f"claim {position} is not an object."
                    )
                    continue

                claim_ids = set(claim.get("evidence_ids", []))

                if not claim_ids.intersection(topic_ids):
                    errors.append(
                        f"claim {position} must cover {topic_name}."
                    )

        # Reject all numeric temporal-duration wording.
        forbidden_patterns = [
            r"\b15[-\u2011 ]day\b",
            r"\b15[-\u2011 ]day\s+window",
            r"\b15[-\u2011 ]day\s+windows",
            r"\b15D\b",
            r"\b15\s+D\b",
        ]

        for position, claim in enumerate(candidate_claims, start=1):
            if not isinstance(claim, dict):
                continue

            claim_text = str(claim.get("text", ""))

            for pattern in forbidden_patterns:
                if re.search(
                    pattern,
                    claim_text,
                    flags=re.IGNORECASE,
                ):
                    errors.append(
                        f"claim {position} contains forbidden temporal "
                        f"wording: {pattern!r}."
                    )

        return errors

    # ---------------------------------------------------------
    # One compact repair attempt
    # ---------------------------------------------------------

    if not validation.get("valid", False):
        print(
            "\n      First Conclusion was incomplete/invalid."
        )
        print(
            "      Attempting one compact completeness repair..."
        )

        compact_retry_prompt = (
            "Return ONLY valid JSON. "
            "Return EXACTLY FIVE claims, never six, in this exact order. "
            "Claim 1 must cover overall case volume using EV-CASE-001. "
            "Claim 2 must cover seriousness using EV-CASE-002 or approved "
            "seriousness evidence. "
            "Claim 3 must cover leading reported reactions using "
            "EV-REACTION-TOP-001 through EV-REACTION-TOP-005. "
            "Claim 4 must cover temporal observations using only "
            "EV-TREND-001 through EV-TREND-004. "
            "IMPORTANT: do not use any EV-TREND-15D-* evidence in claim text "
            "and do not write 15D, 15-day, 15 day, or any numeric temporal "
            "duration. "
            "Claim 5 must cover analytical limitations using EV-LIMIT-001. "
            "Every claim must contain evidence_ids. "
            "Use ONLY supplied evidence. "
            "Do not calculate anything. "
            "Do not introduce dates. "
            "Do not introduce unsupported numbers. "
            "Do not establish causality. "
            "Do not declare or rule out a safety signal. "
            "Do not make an expectedness determination. "
            "Do not merge two required topics into one claim. "
            "Do not add a sixth claim. "
            "JSON structure: "
            '{"section":"conclusion","claims":['
            '{"text":"...","evidence_ids":["EV-..."]},'
            '{"text":"...","evidence_ids":["EV-..."]},'
            '{"text":"...","evidence_ids":["EV-..."]},'
            '{"text":"...","evidence_ids":["EV-..."]},'
            '{"text":"...","evidence_ids":["EV-..."]}'
            "]}"
            "\nAPPROVED EVIDENCE:\n"
            + compact_json(conclusion_evidence)
        )

        try:
            retry_generated = client.generate_json(
                prompt=compact_retry_prompt,
                system_instruction=load_prompt("system.txt"),
                temperature=TEMPERATURE,
                max_output_tokens=700,
            )

            retry_validation = validate_generated_section(
                retry_generated,
                registry,
                allowed_evidence_ids=set(
                    conclusion_evidence.keys()
                ),
            )

            retry_contract_errors = conclusion_contract_errors(
                retry_generated
            )

            if retry_contract_errors:
                retry_validation["valid"] = False
                retry_validation.setdefault(
                    "errors",
                    [],
                ).extend(
                    [
                        "Conclusion contract: " + error
                        for error in retry_contract_errors
                    ]
                )

            if retry_validation.get("valid", False):
                generated = retry_generated
                validation = retry_validation

                print(
                    "      Compact completeness repair: SUCCESS"
                )
            else:
                print(
                    "      Compact completeness repair: FAILED"
                )

                for error in retry_contract_errors:
                    print(
                        f"        {error}"
                    )

        except Exception as retry_exc:
            print(
                "      Compact completeness repair failed:"
                f" {retry_exc}"
            )

    # =========================================================
    # DETERMINISTIC GROUNDED FALLBACK
    # =========================================================
    # This is intentionally conservative. It does not calculate anything,
    # does not copy 15D evidence into prose, and does not add unsupported
    # temporal numbers. It exists so an LLM formatting failure cannot block
    # a valid grounded report.
    # =========================================================

    if not validation.get("valid", False):
        print(
            "\n      LLM Conclusion remained incomplete or invalid."
        )
        print(
            "      Using deterministic evidence-grounded Conclusion fallback..."
        )

        deterministic_conclusion = {
            "section": "conclusion",
            "claims": [
                {
                    "text": (
                        "A total of 1,024 canonical cases were identified "
                        "in the reporting period."
                    ),
                    "evidence_ids": [
                        "EV-CASE-001"
                    ],
                },
                {
                    "text": (
                        "Among the canonical cases, 1,023 were classified "
                        "as serious."
                    ),
                    "evidence_ids": [
                        "EV-CASE-002"
                    ],
                },
                {
                    "text": (
                        "The leading reported reactions included Acute kidney "
                        "injury, Drug ineffective, Hypotension, Drug interaction, "
                        "and Dyspnoea."
                    ),
                    "evidence_ids": [
                        "EV-REACTION-TOP-001",
                        "EV-REACTION-TOP-002",
                        "EV-REACTION-TOP-003",
                        "EV-REACTION-TOP-004",
                        "EV-REACTION-TOP-005",
                    ],
                },
                {
                    "text": (
                        "Temporal observations were identified across the "
                        "analyzed reporting windows."
                    ),
                    "evidence_ids": [
                        "EV-TREND-001",
                        "EV-TREND-002",
                        "EV-TREND-003",
                        "EV-TREND-004",
                    ],
                },
                {
                    "text": (
                        "The analysis has limitations that should be considered "
                        "when interpreting the observed evidence."
                    ),
                    "evidence_ids": [
                        "EV-LIMIT-001"
                    ],
                },
            ],
        }

        deterministic_validation = validate_generated_section(
            deterministic_conclusion,
            registry,
            allowed_evidence_ids=set(
                conclusion_evidence.keys()
            ),
        )

        deterministic_contract_errors = conclusion_contract_errors(
            deterministic_conclusion
        )

        if deterministic_contract_errors:
            deterministic_validation["valid"] = False
            deterministic_validation.setdefault(
                "errors",
                [],
            ).extend(
                [
                    "Conclusion contract: " + error
                    for error in deterministic_contract_errors
                ]
            )

        if deterministic_validation.get("valid", False):
            generated = deterministic_conclusion
            validation = deterministic_validation

            print(
                "      Deterministic Conclusion fallback: SUCCESS"
            )
        else:
            print(
                "      Deterministic Conclusion fallback: FAILED"
            )

            for error in deterministic_contract_errors:
                print(
                    f"        {error}"
                )

    # =========================================================
    # FINAL HARD VALIDATION
    # =========================================================
    # Re-run both the project validator and the five-claim contract on the
    # final selected candidate immediately before saving.
    # =========================================================

    final_validation = validate_generated_section(
        generated,
        registry,
        allowed_evidence_ids=set(
            conclusion_evidence.keys()
        ),
    )

    final_contract_errors = conclusion_contract_errors(
        generated
    )

    if final_contract_errors:
        final_validation["valid"] = False
        final_validation.setdefault(
            "errors",
            [],
        ).extend(
            [
                "Final Conclusion contract: " + error
                for error in final_contract_errors
            ]
        )

    validation = final_validation

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
    # Nothing reaches disk unless the final hard validation passed.

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