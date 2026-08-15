import json

from data.loader import load_dataset
from data.case_normalizer import normalize_cases

from evidence.evidence_builder import (
    build_evidence_pack,
)

from evidence.evidence_registry import (
    create_evidence_registry,
)


def main():

    print("=" * 70)
    print("EVIDENCE REGISTRY TEST")
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

    # ---------------------------------------------------------
    # BUILD REGISTRY
    # ---------------------------------------------------------

    registry = create_evidence_registry(
        evidence
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print(
        f"\nTotal evidence items: "
        f"{len(registry)}"
    )

    print("\nEvidence IDs:")

    for evidence_id in registry:
        print(
            f"  {evidence_id}"
        )

    # ---------------------------------------------------------
    # IMPORTANT EXAMPLES
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("KEY EVIDENCE ITEMS")
    print("=" * 70)

    for evidence_id in [
        "EV-CASE-001",
        "EV-CASE-002",
        "EV-REACTION-TOP-001",
        "EV-REACTION-SERIOUS-001",
        "EV-TREND-001",
        "EV-TREND-15D-001",
        "EV-LIMIT-001",
    ]:

        print(
            json.dumps(
                registry[evidence_id],
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()