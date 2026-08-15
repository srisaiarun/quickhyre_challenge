from data.loader import load_dataset
from data.case_normalizer import (
    normalize_cases,
    print_case_summary,
)


def main():
    raw_df = load_dataset()

    cases_df = normalize_cases(raw_df)

    print_case_summary(
        raw_df,
        cases_df,
    )


if __name__ == "__main__":
    main()