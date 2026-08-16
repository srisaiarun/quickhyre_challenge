import json
from pathlib import Path
from datetime import datetime


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
SECTION_DIR = OUTPUT_DIR / "sections"

SECTION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REQUIRED SECTIONS
# ============================================================

REQUIRED_SECTIONS = [
    "overview",
    "demographics",
    "safety_findings",
    "trends",
    "limitations",
    "conclusion",
]


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path) -> dict:
    """
    Load a JSON file and ensure it contains an object.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required section file does not exist:\n{path}"
        )

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON in:\n{path}\n"
            f"Error: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Expected JSON object in:\n{path}"
        )

    return data


def save_json(path: Path, data: dict):
    """
    Save JSON deterministically and readably.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def check_section_structure(
    section_name: str,
    result: dict,
):
    """
    Validate the basic structure of a generated section.

    This does NOT regenerate anything and does NOT call Gemini.
    """

    if not isinstance(result, dict):
        raise RuntimeError(
            f"{section_name}: section must be a JSON object."
        )

    expected_section = section_name

    actual_section = result.get("section")

    if actual_section != expected_section:
        raise RuntimeError(
            f"{section_name}: unexpected section name.\n"
            f"Expected: {expected_section}\n"
            f"Found: {actual_section}"
        )

    claims = result.get("claims")

    if not isinstance(claims, list):
        raise RuntimeError(
            f"{section_name}: 'claims' must be a list."
        )

    for index, claim in enumerate(
        claims,
        start=1,
    ):

        if not isinstance(claim, dict):
            raise RuntimeError(
                f"{section_name}: claim {index} "
                f"must be an object."
            )

        if not isinstance(
            claim.get("text"),
            str,
        ):
            raise RuntimeError(
                f"{section_name}: claim {index} "
                f"must contain string 'text'."
            )

        evidence_ids = claim.get(
            "evidence_ids"
        )

        if not isinstance(
            evidence_ids,
            list,
        ):
            raise RuntimeError(
                f"{section_name}: claim {index} "
                f"must contain list 'evidence_ids'."
            )

        if not evidence_ids:
            raise RuntimeError(
                f"{section_name}: claim {index} "
                f"has no evidence IDs."
            )

        for evidence_id in evidence_ids:

            if not isinstance(
                evidence_id,
                str,
            ):
                raise RuntimeError(
                    f"{section_name}: claim {index} "
                    f"contains a non-string evidence ID."
                )


def check_validation(
    section_name: str,
    result: dict,
):
    """
    Check validation information if it exists.

    Previously generated section files contain validation
    information. A section is accepted only when validation
    explicitly reports valid=True.

    For compatibility, this function also accepts sections
    without a validation field only if they passed the structural
    checks above. However, the final report clearly marks whether
    validation metadata was present.
    """

    validation = result.get(
        "validation"
    )

    if validation is None:
        return False

    if not isinstance(
        validation,
        dict,
    ):
        raise RuntimeError(
            f"{section_name}: invalid validation object."
        )

    if not validation.get(
        "valid",
        False,
    ):
        raise RuntimeError(
            f"{section_name}: VALIDATION FAILED.\n"
            f"{json.dumps(validation, indent=2, ensure_ascii=False)}"
        )

    return True


def load_validated_section(
    section_name: str,
) -> tuple[dict, bool]:

    path = (
        SECTION_DIR
        / f"{section_name}.json"
    )

    print(
        f"\nLoading {section_name.replace('_', ' ').title()}..."
    )

    result = load_json(path)

    check_section_structure(
        section_name,
        result,
    )

    validation_present = check_validation(
        section_name,
        result,
    )

    print(
        f"      ✓ {section_name.replace('_', ' ').title()} loaded."
    )

    if validation_present:
        print(
            "      ✓ Validation metadata: PASS."
        )
    else:
        print(
            "      ⚠ Validation metadata not present."
        )

    return (
        result,
        validation_present,
    )


# ============================================================
# FINAL REPORT VALIDATION
# ============================================================

def validate_combined_report(
    combined: dict,
):
    """
    Final deterministic sanity check over the assembled report.
    """

    if not isinstance(
        combined,
        dict,
    ):
        raise RuntimeError(
            "Combined report must be a JSON object."
        )

    metadata = combined.get(
        "metadata"
    )

    if not isinstance(
        metadata,
        dict,
    ):
        raise RuntimeError(
            "Combined report metadata is missing."
        )

    sections = combined.get(
        "sections"
    )

    if not isinstance(
        sections,
        dict,
    ):
        raise RuntimeError(
            "Combined report sections are missing."
        )

    missing = [
        section
        for section in REQUIRED_SECTIONS
        if section not in sections
    ]

    if missing:
        raise RuntimeError(
            "Combined report is missing sections: "
            + ", ".join(missing)
        )

    for section_name in REQUIRED_SECTIONS:

        section = sections[
            section_name
        ]

        check_section_structure(
            section_name,
            section,
        )

    return True


# ============================================================
# MAIN PIPELINE
# ============================================================

def build_report():

    print("=" * 70)
    print(
        "GENAR — COMPLETE GROUNDED REPORT PIPELINE"
    )
    print("=" * 70)

    print(
        "\nIMPORTANT:"
        "\nThis stage does NOT call Gemini."
        "\nIt assembles previously generated and validated "
        "section outputs."
    )

    results = {}

    validation_status = {}

    # ========================================================
    # LOAD ALL SIX VALIDATED SECTIONS
    # ========================================================

    for index, section_name in enumerate(
        REQUIRED_SECTIONS,
        start=1,
    ):

        print(
            f"\n[{index}/6] Loading "
            f"{section_name.replace('_', ' ').title()}..."
        )

        result, validation_present = (
            load_validated_section(
                section_name
            )
        )

        results[
            section_name
        ] = result

        validation_status[
            section_name
        ] = validation_present

    # ========================================================
    # BUILD COMBINED OUTPUT
    # ========================================================

    combined = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "pipeline": (
                "deterministic_report_assembly"
            ),
            "gemini_called": False,
            "sections": REQUIRED_SECTIONS,
            "validation_metadata_present": (
                all(
                    validation_status.values()
                )
            ),
        },
        "sections": results,
    }

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL REPORT STRUCTURE VALIDATION"
    )

    print(
        "=" * 70
    )

    validate_combined_report(
        combined
    )

    for section_name in REQUIRED_SECTIONS:

        status = validation_status[
            section_name
        ]

        if status:
            print(
                f"{section_name.replace('_', ' ').title():20} : PASS"
            )
        else:
            print(
                f"{section_name.replace('_', ' ').title():20} : "
                "STRUCTURE PASS / VALIDATION METADATA ABSENT"
            )

    # ========================================================
    # SAVE COMBINED REPORT
    # ========================================================

    combined_path = (
        OUTPUT_DIR
        / "grounded_report.json"
    )

    save_json(
        combined_path,
        combined,
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print(
        "\n" + "=" * 70
    )
    print(
        "FINAL STATUS: COMPLETE GROUNDED REPORT ASSEMBLED"
    )
    print(
        "=" * 70
    )

    print(
        "\nGemini calls during assembly: 0"
    )

    print(
        f"\nCombined output:\n"
        f"  {combined_path}"
    )

    print(
        f"\nSection outputs:\n"
        f"  {SECTION_DIR}"
    )

    return combined


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    build_report()