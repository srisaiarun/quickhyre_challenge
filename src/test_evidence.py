import json

from data.loader import load_dataset
from data.case_normalizer import normalize_cases

from evidence.evidence_builder import (
    build_evidence_pack,
)


def main():

    raw_df = load_dataset()

    cases_df = normalize_cases(raw_df)

    evidence = build_evidence_pack(
        cases_df
    )

    print(
        json.dumps(
            evidence,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()