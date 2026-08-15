import json

from data.loader import load_dataset
from data.case_normalizer import normalize_cases

from evidence.evidence_builder import (
    build_evidence_pack,
)

from context.context_builder import (
    build_all_contexts,
)


def main():

    print("=" * 70)
    print("CONTEXT ENGINEERING TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # LOAD AND NORMALIZE
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
    # BUILD SECTION CONTEXTS
    # ---------------------------------------------------------
    contexts = build_all_contexts(
        evidence
    )

    # ---------------------------------------------------------
    # PRINT CONTEXT NAMES
    # ---------------------------------------------------------
    print("\nAvailable contexts:")

    for section in contexts:
        print(f"  - {section}")

    # ---------------------------------------------------------
    # PRINT EACH CONTEXT
    # ---------------------------------------------------------
    for section, context in contexts.items():

        print("\n" + "=" * 70)
        print(
            f"{section.upper()} CONTEXT"
        )
        print("=" * 70)

        print(
            json.dumps(
                context,
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()