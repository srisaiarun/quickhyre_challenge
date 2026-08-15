from data.loader import load_dataset
from data.case_normalizer import normalize_cases

from analysis.case_analysis import (
    analyze_case_volume,
    analyze_demographics,
)

from analysis.reaction_analysis import (
    analyze_reactions,
    analyze_outcomes,
)

from analysis.reaction_analysis import (
    analyze_reactions,
)


def main():

    print("\nLoading dataset...")

    raw_df = load_dataset()

    print("Normalizing cases...")

    cases_df = normalize_cases(raw_df)

    print("Running deterministic analysis...")

    case_volume = analyze_case_volume(
        cases_df
    )

    demographics = analyze_demographics(
        cases_df
    )

    outcomes = analyze_outcomes(
        cases_df
    )

    reactions = analyze_reactions(
        cases_df
    )

    print("\n" + "=" * 70)
    print("DETERMINISTIC ANALYSIS RESULTS")
    print("=" * 70)

    print("\nCASE VOLUME")
    print(case_volume)

    print("\nAGE GROUPS")
    for item in demographics["age_groups"]:
        print(item)

    print("\nSEX")
    for item in demographics["sex"]:
        print(item)

    print("\nTOP COUNTRIES")
    for item in demographics["country"]:
        print(item)

    print("\nOUTCOMES")
    for item in outcomes:
        print(item)

    print("\nTOP REACTIONS")
    for item in reactions["top_reactions"]:
        print(item)

    print("\nTOP SERIOUS REACTIONS")
    for item in reactions["top_serious_reactions"]:
        print(item)

    print(
        f"\nTotal reaction records: "
        f"{reactions['total_reaction_records']:,}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()