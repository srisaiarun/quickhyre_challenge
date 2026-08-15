from data.loader import load_dataset
from data.case_normalizer import normalize_cases

from analysis.time_analysis import (
    analyze_monthly_trends,
    analyze_15_day_windows,
    detect_15_day_spikes,
)


def main():

    raw_df = load_dataset()

    cases_df = normalize_cases(raw_df)

    monthly = analyze_monthly_trends(
        cases_df
    )

    windows = analyze_15_day_windows(
        cases_df
    )

    alerts = detect_15_day_spikes(
        windows
    )

    print("\n" + "=" * 70)
    print("MONTHLY TRENDS")
    print("=" * 70)

    for item in monthly:
        print(item)

    print("\n" + "=" * 70)
    print("15-DAY WINDOWS")
    print("=" * 70)

    for item in windows:
        print(item)

    print("\n" + "=" * 70)
    print("15-DAY ALERTS")
    print("=" * 70)

    if alerts:
        for item in alerts:
            print(item)
    else:
        print("No 15-day volume alerts detected.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()