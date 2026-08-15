from data.loader import load_dataset
from data.case_normalizer import normalize_cases

from evidence.evidence_builder import (
    build_evidence_pack,
)

from evidence.evidence_registry import (
    create_evidence_registry,
)

from validation.evidence_validator import (
    validate_generated_section,
)


def main():

    print("=" * 70)
    print("EVIDENCE VALIDATOR TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    raw_df = load_dataset()

    cases_df = normalize_cases(
        raw_df
    )

    # ---------------------------------------------------------
    # BUILD EVIDENCE
    # ---------------------------------------------------------

    evidence = build_evidence_pack(
        cases_df
    )

    registry = create_evidence_registry(
        evidence
    )

    # ---------------------------------------------------------
    # VALID CLAIM
    # ---------------------------------------------------------

    valid_output = {
        "section": "overview",
        "claims": [
            {
                "text": (
                    "A total of 1024 cases were "
                    "reported, of which 1023 "
                    "were classified as serious."
                ),
                "evidence_ids": [
                    "EV-CASE-001",
                    "EV-CASE-002",
                ],
            }
        ],
    }

    result = validate_generated_section(
        valid_output,
        registry,
    )

    print("\nVALID CLAIM TEST")
    print(
        "PASS"
        if result["valid"]
        else "FAIL"
    )

    if result["errors"]:
        print(
            "Errors:",
            result["errors"],
        )

    # ---------------------------------------------------------
    # INVALID NUMBER TEST
    # ---------------------------------------------------------

    invalid_number_output = {
        "section": "overview",
        "claims": [
            {
                "text": (
                    "A total of 1250 cases were "
                    "reported."
                ),
                "evidence_ids": [
                    "EV-CASE-001",
                ],
            }
        ],
    }

    result = validate_generated_section(
        invalid_number_output,
        registry,
    )

    print("\nINVALID NUMBER TEST")

    if result["valid"]:
        print("FAIL")
        print(
            "Validator incorrectly accepted "
            "an unsupported number."
        )
    else:
        print("PASS")
        print(
            "Validator rejected unsupported "
            "number as expected."
        )

    # ---------------------------------------------------------
    # INVALID EVIDENCE ID TEST
    # ---------------------------------------------------------

    invalid_evidence_output = {
        "section": "overview",
        "claims": [
            {
                "text": (
                    "A total of 1024 cases "
                    "were reported."
                ),
                "evidence_ids": [
                    "EV-FAKE-999",
                ],
            }
        ],
    }

    result = validate_generated_section(
        invalid_evidence_output,
        registry,
    )

    print("\nINVALID EVIDENCE ID TEST")

    if result["valid"]:
        print("FAIL")
        print(
            "Validator incorrectly accepted "
            "an unknown evidence ID."
        )
    else:
        print("PASS")
        print(
            "Validator rejected unknown "
            "evidence ID as expected."
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()