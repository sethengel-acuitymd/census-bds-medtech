#!/usr/bin/env python3
"""Census BDS Medtech Firm Survival Analysis.

Analyzes long-term survival patterns for Medical Technology Manufacturers
(NAICS 3391) using the Census Bureau's Business Dynamics Statistics API.

Usage:
    python analyze.py                   # Full analysis with charts
    python analyze.py --no-charts       # Tables only, no chart files
    python analyze.py --output-dir out  # Save charts to custom directory
    python analyze.py --compare 3391,3254,5417  # Compare NAICS codes
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tabulate import tabulate

import bds_client
import survival


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_table(df: pd.DataFrame, **kwargs: object) -> None:
    print(tabulate(df, headers="keys", tablefmt="simple", showindex=False, **kwargs))
    print()


def analyze_timeseries(ts: pd.DataFrame) -> pd.DataFrame:
    """Print time series analysis and return enriched DataFrame."""
    print_section("NAICS 3391: Medical Equipment & Supplies — Time Series Overview")

    metrics = survival.compute_annual_metrics(ts)

    # Summary table: show every 5 years + most recent
    years = list(range(1980, 2025, 5)) + [metrics["YEAR"].max()]
    summary = metrics[metrics["YEAR"].isin(years)][
        ["YEAR", "FIRM", "ESTAB", "EMP", "AVG_FIRM_SIZE", "FIRM_DEATH_RATE", "NET_JOB_CREATION_RATE"]
    ].copy()
    summary.columns = ["Year", "Firms", "Estabs", "Employees", "Avg Size", "Death Rate %", "Net Job Rate %"]
    print_table(summary, floatfmt=".1f")

    # Key stats
    first, last = metrics.iloc[0], metrics.iloc[-1]
    print(f"  Period: {int(first['YEAR'])} — {int(last['YEAR'])}")
    print(f"  Firm count:  {int(first['FIRM']):,} → {int(last['FIRM']):,} ({int(last['FIRM'] - first['FIRM']):+,})")
    print(f"  Employment:  {int(first['EMP']):,} → {int(last['EMP']):,} ({int(last['EMP'] - first['EMP']):+,})")
    print(f"  Avg firm size: {first['AVG_FIRM_SIZE']:.0f} → {last['AVG_FIRM_SIZE']:.0f} employees")
    print(f"  Avg annual death rate: {metrics['FIRM_DEATH_RATE'].mean():.1f}%")
    print()

    return metrics


def analyze_firm_age(age_df: pd.DataFrame) -> None:
    """Print firm age distribution analysis."""
    print_section("Firm Age Distribution")

    dist = survival.compute_age_distribution(age_df)

    # Show distribution for recent years
    recent = dist[dist.index >= 2018]
    print("Percentage of firms by age category (recent years):")
    print_table(recent.reset_index(), floatfmt=".1f")


def analyze_survival(age_df: pd.DataFrame) -> None:
    """Print survival proxy analysis."""
    print_section("Cohort Survival Proxy (Cohort Tracking Method)")

    print("Method: Track firm counts across consecutive single-year age buckets.")
    print("E.g., firms aged '0 years' in 2020 vs firms aged '1 year' in 2021.")
    print("BDS excludes M&A and NAICS reclassification from deaths, so the gap")
    print("between cohort counts reflects genuine firm cessation.\n")

    transitions = survival.compute_survival_proxy(age_df)

    # Average survival rates across all years
    avg_survival = (
        transitions.groupby(["AGE_FROM", "AGE_TO"])
        .agg(
            AVG_SURVIVAL=("SURVIVAL_RATE", "mean"),
            MIN_SURVIVAL=("SURVIVAL_RATE", "min"),
            MAX_SURVIVAL=("SURVIVAL_RATE", "max"),
        )
        .reset_index()
    )
    avg_survival.columns = ["From Age", "To Age", "Avg Survival %", "Min %", "Max %"]
    print("Average survival rates by age transition (all years):")
    print_table(avg_survival, floatfmt=".1f")

    # Recent 5-year average
    recent = transitions[transitions["YEAR"] >= 2018]
    if not recent.empty:
        recent_avg = (
            recent.groupby(["AGE_FROM", "AGE_TO"])["SURVIVAL_RATE"]
            .mean()
            .reset_index()
        )
        recent_avg.columns = ["From Age", "To Age", "Survival % (2018-2023)"]
        print("Recent survival rates (2018-2023 average):")
        print_table(recent_avg, floatfmt=".1f")


def analyze_death_rates(age_df: pd.DataFrame) -> None:
    """Print firm death rate analysis by age cohort."""
    print_section("Firm Death Rates by Age (Direct FIRMDEATH Method)")

    print("Method: FIRMDEATH_FIRMS / FIRM count per age bucket per year.")
    print("A firm death requires ALL establishments to cease operations.")
    print("M&A is excluded — acquired firms are not counted as deaths.\n")

    death_rates = survival.compute_death_rate_by_age(age_df)

    # Average death rates by age bucket across all years
    avg_by_age = (
        death_rates.groupby(["FAGE", "FAGE_LABEL"])
        .agg(
            AVG_DEATH_RATE=("DEATH_RATE", "mean"),
            RECENT_DEATH_RATE=("DEATH_RATE", lambda x: x[death_rates.loc[x.index, "YEAR"] >= 2018].mean()),
        )
        .reset_index()
        .sort_values("FAGE")
    )
    avg_by_age.columns = ["Code", "Age Bucket", "Avg Death Rate % (All Years)", "Avg Death Rate % (2018-2023)"]
    display = avg_by_age[["Age Bucket", "Avg Death Rate % (All Years)", "Avg Death Rate % (2018-2023)"]]
    print_table(display, floatfmt=".1f")

    print("Key insight: Young firms die at much higher rates than mature ones.")
    print("The death rate method covers ALL age buckets (including 6-10, 11-15, etc.)")
    print("while the cohort tracking method is limited to single-year buckets (0-5).\n")


def analyze_firm_vs_estab_deaths(ts: pd.DataFrame) -> None:
    """Print comparison of firm deaths vs establishment exits."""
    print_section("Firm Deaths vs. Establishment Exits")

    print("ESTABS_EXIT = ALL establishment closures (partial + full firm death)")
    print("FIRMDEATH_FIRMS = only firms where ALL establishments ceased operations")
    print("FIRMDEATH_EMP = employment at dying firms (prior year, since current = 0)")
    print("Dying firms tend to be much smaller than survivors.\n")

    death_comparison = survival.compute_estab_vs_firm_deaths(ts)

    # Show recent years
    recent = death_comparison[death_comparison["YEAR"] >= 2015].copy()
    display = recent[
        [
            "YEAR",
            "FIRMDEATH_FIRMS",
            "FIRM_DEATH_SHARE_OF_FIRMS",
            "FIRMDEATH_EMP",
            "FIRMDEATH_EMP_SHARE",
            "AVG_DYING_FIRM_SIZE",
            "AVG_SURVIVING_FIRM_SIZE",
        ]
    ].copy()
    display.columns = [
        "Year",
        "Firm Deaths",
        "Death Share %",
        "Emp at Deaths",
        "Emp Share %",
        "Avg Dying Size",
        "Avg Surviving Size",
    ]
    print_table(display, floatfmt=".1f")

    # Summary
    avg_dying = death_comparison["AVG_DYING_FIRM_SIZE"].mean()
    avg_surviving = death_comparison["AVG_SURVIVING_FIRM_SIZE"].mean()
    avg_emp_share = death_comparison["FIRMDEATH_EMP_SHARE"].mean()
    print(f"  Avg dying firm size:    {avg_dying:.0f} employees")
    print(f"  Avg surviving firm size: {avg_surviving:.0f} employees")
    print(f"  Avg employment share lost to firm deaths: {avg_emp_share:.1f}%")
    print(f"  → Firm deaths account for a small share of total employment loss,")
    print(f"    because dying firms are predominantly small/young.")
    print()


def synthesize_survival_profile(age_df: pd.DataFrame, ts: pd.DataFrame) -> None:
    """Print a unified survival profile combining both methods."""
    print_section("Medtech Firm Survival Profile (Synthesis)")

    profile = survival.compute_survival_profile(age_df, ts)
    year_range = profile["recent_year_range"]

    print(f"Based on {year_range} data for NAICS 3391.\n")

    # Cumulative survival curve
    print("Cumulative survival (cohort tracking, of 100 new firms):")
    print("-" * 50)
    for label, pct in profile["cumulative_survival"]:
        bar = "#" * int(pct / 2)
        print(f"  {label:<25s} {pct:5.1f}%  {bar}")
    print()

    # Annual death rates across full lifecycle
    print("Annual death rate by firm age (FIRMDEATH method):")
    print("-" * 50)
    max_rate = max(r for _, r in profile["death_rates_by_age"])
    for label, rate in profile["death_rates_by_age"]:
        bar_len = int((rate / max_rate) * 30) if max_rate > 0 else 0
        bar = "#" * bar_len
        print(f"  {label:<15s} {rate:5.1f}%  {bar}")
    print()

    # Size context
    print("Who dies?")
    print(f"  Avg dying firm:     {profile['avg_dying_firm_size']:.0f} employees")
    print(f"  Avg surviving firm: {profile['avg_surviving_firm_size']:.0f} employees")
    print(f"  Employment lost to firm deaths: {profile['emp_share_lost_to_deaths']:.1f}% of industry total")
    print()

    # Narrative
    surv_5yr = profile["cumulative_survival"][-1][1] if len(profile["cumulative_survival"]) > 1 else 0
    yr1_death = next((r for l, r in profile["death_rates_by_age"] if l == "1 year"), 0)
    mature_death = next((r for l, r in profile["death_rates_by_age"] if l == "26+ years"), 0)

    print("Summary:")
    print(f"  - Of 100 new medtech firms, ~{surv_5yr:.0f} survive to year 5.")
    print(f"  - First-year mortality is highest ({yr1_death:.0f}%), then declines with age.")
    print(f"  - Even mature firms (26+ years) still face {mature_death:.0f}% annual mortality.")
    print(f"  - Dying firms are overwhelmingly small (~{profile['avg_dying_firm_size']:.0f} employees),")
    print(f"    so firm deaths remove only ~{profile['emp_share_lost_to_deaths']:.1f}% of industry employment.")
    print(f"  - The industry is consolidating: fewer firms, but larger and more resilient.")
    print()


def analyze_consolidation(ts: pd.DataFrame) -> None:
    """Print consolidation analysis by decade."""
    print_section("Industry Consolidation by Decade")

    consol = survival.compute_consolidation_metrics(ts)
    display = consol[
        ["DECADE", "START_FIRMS", "END_FIRMS", "FIRM_CHANGE_PCT", "END_EMP", "EMP_CHANGE_PCT", "AVG_FIRM_DEATH_RATE"]
    ].copy()
    display.columns = ["Decade", "Start Firms", "End Firms", "Firm Chg %", "End Emp", "Emp Chg %", "Avg Death Rate %"]
    print_table(display, floatfmt=".1f")


def analyze_comparison(naics_codes: list[str]) -> None:
    """Compare metrics across NAICS codes."""
    print_section("Industry Comparison (Most Recent Year)")

    df = bds_client.get_comparison(naics_codes)
    display = df[["NAICS", "NAICS_LABEL", "FIRM", "ESTAB", "EMP", "NET_JOB_CREATION", "FIRMDEATH_FIRMS"]].copy()
    display.columns = ["NAICS", "Industry", "Firms", "Estabs", "Employees", "Net Jobs", "Firm Deaths"]
    print_table(display)


def generate_charts(metrics: pd.DataFrame, age_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate and save analysis charts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Firm count and employment over time (dual axis)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of Firms", color="tab:blue")
    ax1.plot(metrics["YEAR"], metrics["FIRM"], color="tab:blue", linewidth=2, label="Firms")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Employment", color="tab:red")
    ax2.plot(metrics["YEAR"], metrics["EMP"], color="tab:red", linewidth=2, label="Employment")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.suptitle("NAICS 3391: Firms vs Employment (1978–Present)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "firms_vs_employment.png", dpi=150)
    plt.close(fig)

    # 2. Firm death rate over time
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(metrics["YEAR"], metrics["FIRM_DEATH_RATE"], color="tab:orange", linewidth=2)
    ax.axhline(y=metrics["FIRM_DEATH_RATE"].mean(), color="gray", linestyle="--", alpha=0.7, label="Average")
    ax.set_xlabel("Year")
    ax.set_ylabel("Firm Death Rate (%)")
    ax.set_title("NAICS 3391: Annual Firm Death Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "firm_death_rate.png", dpi=150)
    plt.close(fig)

    # 3. Average firm size over time
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(metrics["YEAR"], metrics["AVG_FIRM_SIZE"], color="tab:green", linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Employees per Firm")
    ax.set_title("NAICS 3391: Average Firm Size (Consolidation Trend)")
    fig.tight_layout()
    fig.savefig(output_dir / "avg_firm_size.png", dpi=150)
    plt.close(fig)

    # 4. Firm age distribution stacked area
    dist = survival.compute_age_distribution(age_df)
    if not dist.empty:
        fig, ax = plt.subplots(figsize=(14, 7))
        dist.plot.area(ax=ax, alpha=0.7)
        ax.set_xlabel("Year")
        ax.set_ylabel("Percentage of Firms")
        ax.set_title("NAICS 3391: Firm Age Distribution Over Time")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "age_distribution.png", dpi=150)
        plt.close(fig)

    # 5. Job creation vs destruction
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(metrics["YEAR"] - 0.2, metrics["JOB_CREATION"], width=0.4, label="Creation", color="tab:green", alpha=0.7)
    ax.bar(metrics["YEAR"] + 0.2, metrics["JOB_DESTRUCTION"], width=0.4, label="Destruction", color="tab:red", alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Jobs")
    ax.set_title("NAICS 3391: Job Creation vs Destruction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "job_dynamics.png", dpi=150)
    plt.close(fig)

    print(f"\n  Charts saved to: {output_dir.resolve()}")
    for f in sorted(output_dir.glob("*.png")):
        print(f"    - {f.name}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Census BDS Medtech Survival Analysis")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    parser.add_argument("--output-dir", type=Path, default=Path("charts"), help="Directory for chart output")
    parser.add_argument("--compare", type=str, help="Comma-separated NAICS codes to compare")
    parser.add_argument("--naics", type=str, default="3391", help="NAICS code to analyze (default: 3391)")
    args = parser.parse_args()

    print(f"\nFetching BDS time series data for NAICS {args.naics}...")
    ts = bds_client.get_timeseries(naics=args.naics)
    print(f"  Retrieved {len(ts)} years of data.")

    metrics = analyze_timeseries(ts)
    analyze_consolidation(ts)

    print(f"\nFetching firm age data for NAICS {args.naics}...")
    age_df = bds_client.get_firm_age_timeseries(naics=args.naics)
    print(f"  Retrieved {len(age_df)} rows (age × year combinations).")

    analyze_firm_age(age_df)
    analyze_survival(age_df)
    analyze_death_rates(age_df)
    analyze_firm_vs_estab_deaths(ts)
    synthesize_survival_profile(age_df, ts)

    if args.compare:
        codes = [c.strip() for c in args.compare.split(",")]
        analyze_comparison(codes)

    if not args.no_charts:
        generate_charts(metrics, age_df, args.output_dir)

    print_section("Analysis Complete")


if __name__ == "__main__":
    main()
