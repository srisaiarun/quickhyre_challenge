from data.loader import load_dataset
from data.case_normalizer import normalize_cases

from analysis.time_analysis import (
    analyze_monthly_trends,
    analyze_15_day_windows,
    detect_15_day_spikes,
)


def main():
    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------
    raw_df = load_dataset()

    # ---------------------------------------------------------
    # NORMALIZE CASES
    # ---------------------------------------------------------
    cases_df = normalize_cases(raw_df)

    # ---------------------------------------------------------
    # MONTHLY TREND ANALYSIS
    # ---------------------------------------------------------
    monthly = analyze_monthly_trends(cases_df)

    # ---------------------------------------------------------
    # ROLLING 15-DAY ANALYSIS
    # ---------------------------------------------------------
    windows = analyze_15_day_windows(cases_df)

    # ---------------------------------------------------------
    # 15-DAY HIGH-VOLUME OBSERVATIONS
    # ---------------------------------------------------------
    alerts = detect_15_day_spikes(windows)

    # =========================================================
    # PRINT MONTHLY TRENDS
    # =========================================================
    print("\n" + "=" * 70)
    print("MONTHLY TRENDS")
    print("=" * 70)

    for item in monthly["months"]:
        print(item)

    print("\nHighest volume month:")
    print(monthly["highest_volume_month"])

    print("\nLowest volume month:")
    print(monthly["lowest_volume_month"])

    print("\nAverage monthly cases:")
    print(monthly["average_monthly_cases"])

    # =========================================================
    # PRINT 15-DAY HIGH-VOLUME OBSERVATIONS
    # =========================================================
    print("\n" + "=" * 70)
    print("15-DAY HIGH-VOLUME OBSERVATIONS")
    print("=" * 70)

    if alerts:
        for item in alerts:
            print(item)
    else:
        print("No high-volume 15-day observations detected.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()