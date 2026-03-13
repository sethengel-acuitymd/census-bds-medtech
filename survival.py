"""Survival and consolidation analysis for BDS data.

BDS Concepts
============

Firm vs. Establishment:
    An *establishment* is a single fixed physical location with economic activity.
    A *firm* is all establishments under common operational control (linked by EIN).
    Single-unit firms have one establishment; multi-unit firms have many.

Firm Death:
    ALL establishments owned by a firm must cease operations for the firm to count
    as dead. A 100-establishment firm where 99 close but 1 continues is NOT a firm
    death. M&A activity is explicitly excluded — if a firm is acquired, its
    establishments transfer to the acquirer and no death is recorded.

Establishment Entry/Exit:
    Entry: zero employment on March 12 of year t-1, positive on March 12 of year t.
    Exit: positive employment on March 12 of year t-1, zero on March 12 of year t.
    Establishment exits include BOTH exits at dying firms AND partial closures at
    surviving multi-unit firms. FIRMDEATH_ESTABS is always a subset of ESTABS_EXIT.

FIRMDEATH variables:
    FIRMDEATH_FIRMS  — count of firms where all establishments ceased operations.
    FIRMDEATH_ESTABS — count of establishments at those dead firms.
    FIRMDEATH_EMP    — employment at those establishments in year t-1 (year t is
                       zero by definition).

NAICS consistency:
    The underlying Longitudinal Business Database assigns a vintage-consistent
    NAICS code (pegged to 2012 NAICS) for each establishment's entire history.
    NAICS reclassification does not create spurious births/deaths.

Retiming:
    Economic Census years (ending in 2 and 7) create artificial spikes in recorded
    births/deaths for multi-unit establishments. The LBD retiming algorithm
    redistributes these to intercensal years, smoothing the time series.
"""

import pandas as pd

# Firm age codes in order (excluding aggregates)
INDIVIDUAL_AGE_CODES = ["010", "020", "030", "040", "050", "060", "070", "080", "090", "100", "110"]
INDIVIDUAL_AGE_LABELS = [
    "0 years",
    "1 year",
    "2 years",
    "3 years",
    "4 years",
    "5 years",
    "6-10 years",
    "11-15 years",
    "16-20 years",
    "21-25 years",
    "26+ years",
]


def compute_annual_metrics(ts: pd.DataFrame) -> pd.DataFrame:
    """Compute derived annual metrics from time series data.

    Adds columns:
        - AVG_FIRM_SIZE: average employees per firm
        - FIRM_DEATH_RATE: firm deaths as % of total firms
        - NET_ENTRY_RATE: establishment entry rate minus exit rate
        - FIRM_YOY_CHANGE: year-over-year change in firm count
        - EMP_YOY_CHANGE: year-over-year change in employment
    """
    df = ts.copy()
    df["AVG_FIRM_SIZE"] = (df["EMP"] / df["FIRM"]).round(1)
    df["FIRM_DEATH_RATE"] = ((df["FIRMDEATH_FIRMS"] / df["FIRM"]) * 100).round(2)
    df["NET_ENTRY_RATE"] = (df["ESTABS_ENTRY_RATE"] - df["ESTABS_EXIT_RATE"]).round(2)
    df["FIRM_YOY_CHANGE"] = df["FIRM"].diff()
    df["EMP_YOY_CHANGE"] = df["EMP"].diff()
    return df


def compute_age_distribution(age_df: pd.DataFrame) -> pd.DataFrame:
    """Compute firm age distribution percentages for each year.

    Returns a pivot table with years as rows and age categories as columns,
    showing the percentage of firms in each age bucket.
    """
    # Filter to individual age codes only (not aggregates like "Total", "1-5 years", "11+ years")
    individual = age_df[age_df["FAGE"].isin(INDIVIDUAL_AGE_CODES)].copy()

    # Get total firms per year from the "001" (Total) code
    totals = age_df[age_df["FAGE"] == "001"][["YEAR", "FIRM"]].rename(columns={"FIRM": "TOTAL_FIRMS"})
    individual = individual.merge(totals, on="YEAR")
    individual["PCT"] = ((individual["FIRM"] / individual["TOTAL_FIRMS"]) * 100).round(2)

    pivot = individual.pivot_table(index="YEAR", columns="FAGE_LABEL", values="PCT")
    # Reorder columns by age
    ordered = [label for label in INDIVIDUAL_AGE_LABELS if label in pivot.columns]
    return pivot[ordered]


def compute_survival_proxy(age_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate cohort survival by tracking firm counts through age buckets over time.

    Compares the number of firms in age bucket N in year Y with the number in
    bucket N+1 in year Y+1. Because BDS excludes M&A and NAICS reclassification
    from firm deaths, the gap between cohort counts genuinely reflects firms that
    ceased all operations (not acquisitions or industry reclassification).

    Limited to single-year age buckets (0→1→2→3→4→5) because multi-year buckets
    (6-10, 11-15, etc.) accumulate multiple cohorts and can't be compared 1:1.

    Returns a DataFrame with columns: YEAR, AGE_FROM, AGE_TO, SURVIVAL_RATE.
    """
    individual = age_df[age_df["FAGE"].isin(INDIVIDUAL_AGE_CODES)].copy()
    individual = individual.sort_values(["YEAR", "FAGE"])

    # Build survival transitions between consecutive single-year age buckets only.
    # Multi-year buckets (6-10, 11-15, etc.) can't be compared 1:1 across years
    # because they accumulate multiple cohorts. We only track 0→1→2→3→4→5.
    single_year_codes = INDIVIDUAL_AGE_CODES[:6]  # 010 through 060
    single_year_labels = INDIVIDUAL_AGE_LABELS[:6]  # "0 years" through "5 years"

    transitions = []
    age_pairs = list(zip(single_year_codes[:-1], single_year_codes[1:]))
    label_pairs = list(zip(single_year_labels[:-1], single_year_labels[1:]))

    for (from_code, to_code), (from_label, to_label) in zip(age_pairs, label_pairs):
        from_data = individual[individual["FAGE"] == from_code][["YEAR", "FIRM"]].rename(
            columns={"FIRM": "FIRMS_FROM"}
        )
        to_data = individual[individual["FAGE"] == to_code][["YEAR", "FIRM"]].rename(
            columns={"FIRM": "FIRMS_TO"}
        )
        # Shift: firms in from_code at year Y should appear in to_code at year Y+1
        to_data = to_data.copy()
        to_data["YEAR"] = to_data["YEAR"] - 1  # align to the "from" year

        merged = from_data.merge(to_data, on="YEAR", how="inner")
        merged["AGE_FROM"] = from_label
        merged["AGE_TO"] = to_label
        merged["SURVIVAL_RATE"] = ((merged["FIRMS_TO"] / merged["FIRMS_FROM"]) * 100).round(1)
        transitions.append(merged)

    return pd.concat(transitions, ignore_index=True).sort_values(["YEAR", "AGE_FROM"])


def compute_cumulative_survival_by_year(age_df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative 5-year survival rates for each cohort year.

    For a cohort born in year Y:
      - Year 0→1 survival uses the "0 years" transition from year Y
      - Year 1→2 survival uses the "1 year" transition from year Y+1
      - Year 2→3 survival uses the "2 years" transition from year Y+2
      - ... and so on through year 4→5

    Also computes conditional survival: given a firm survived year 1,
    what's the probability it reaches year 5?

    Returns a DataFrame with one row per cohort year and columns for
    each year's cumulative survival percentage.
    """
    transitions = compute_survival_proxy(age_df)
    labels = INDIVIDUAL_AGE_LABELS[:5]  # "0 years" through "4 years"

    # Pivot: for each (YEAR, AGE_FROM), get the SURVIVAL_RATE
    # YEAR here is the "from" year — the year in which the firm was at AGE_FROM
    pivot = transitions.pivot_table(index="YEAR", columns="AGE_FROM", values="SURVIVAL_RATE")
    pivot = pivot[[l for l in labels if l in pivot.columns]]

    # Build per-cohort survival by looking up each transition in the right year.
    # For cohort born in year Y: transition i happens in year Y+i.
    all_years = sorted(pivot.index)
    rows = []

    for cohort_year in all_years:
        cumulative = 100.0
        row = {"YEAR": cohort_year}
        complete = True

        for i, label in enumerate(labels):
            lookup_year = cohort_year + i
            if lookup_year not in pivot.index or label not in pivot.columns:
                complete = False
                break
            rate = pivot.loc[lookup_year, label]
            if pd.isna(rate):
                complete = False
                break
            cumulative = cumulative * (rate / 100)
            row[f"SURV_YEAR_{i + 1}"] = round(cumulative, 1)

        if complete:
            rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # Conditional: if survived year 1, probability of reaching year 5
    if "SURV_YEAR_1" in result.columns and "SURV_YEAR_5" in result.columns:
        result["COND_1_TO_5"] = ((result["SURV_YEAR_5"] / result["SURV_YEAR_1"]) * 100).round(1)

    return result


def compute_death_rate_by_age(age_df: pd.DataFrame) -> pd.DataFrame:
    """Compute firm death rates by age cohort using FIRMDEATH_FIRMS.

    This complements compute_survival_proxy by using the explicit firm death count
    rather than inferring survival from cohort size changes. The death rate is:

        death_rate = FIRMDEATH_FIRMS / FIRM * 100

    for each age bucket in each year. This works for ALL age buckets (including
    multi-year ones like 6-10, 11-15, etc.) since it doesn't require tracking
    cohorts across years.

    Advantages over the cohort-tracking proxy:
        - Works for all age buckets, not just single-year ones.
        - Directly measures firm cessation rather than inferring it.
        - Not affected by cohort-size noise from data suppression.

    Limitations:
        - Shows the death rate within an age bucket for a single year, not the
          cumulative probability of a new firm reaching a given age.
        - Cannot distinguish whether a bucket's death rate changed because the
          firms in it changed, or because conditions changed.

    Returns a DataFrame with: YEAR, FAGE_LABEL, FIRM, FIRMDEATH_FIRMS, DEATH_RATE.
    """
    individual = age_df[age_df["FAGE"].isin(INDIVIDUAL_AGE_CODES)].copy()
    individual["DEATH_RATE"] = ((individual["FIRMDEATH_FIRMS"] / individual["FIRM"]) * 100).round(2)
    return individual[["YEAR", "FAGE", "FAGE_LABEL", "FIRM", "FIRMDEATH_FIRMS", "DEATH_RATE"]].sort_values(
        ["YEAR", "FAGE"]
    )


def compute_estab_vs_firm_deaths(ts: pd.DataFrame) -> pd.DataFrame:
    """Compare establishment exits with firm deaths to show partial vs. full closure.

    ESTABS_EXIT counts ALL establishment closures — both at dying firms and at
    surviving multi-unit firms that closed some locations. FIRMDEATH_ESTABS counts
    only closures at firms that fully died. The difference reveals how much
    establishment churn comes from restructuring (partial closure) vs. firm death.

    Returns a DataFrame with per-year breakdown.
    """
    df = ts.copy()
    # FIRMDEATH_ESTABS may not be in the timeseries data; compute what we can
    df["FIRM_DEATH_SHARE_OF_FIRMS"] = ((df["FIRMDEATH_FIRMS"] / df["FIRM"]) * 100).round(2)
    df["FIRMDEATH_EMP_SHARE"] = ((df["FIRMDEATH_EMP"] / df["EMP"]) * 100).round(2)
    df["AVG_DYING_FIRM_SIZE"] = (df["FIRMDEATH_EMP"] / df["FIRMDEATH_FIRMS"]).round(1)
    df["AVG_SURVIVING_FIRM_SIZE"] = (
        (df["EMP"] - df["FIRMDEATH_EMP"]) / (df["FIRM"] - df["FIRMDEATH_FIRMS"])
    ).round(1)
    return df[
        [
            "YEAR",
            "FIRM",
            "FIRMDEATH_FIRMS",
            "FIRM_DEATH_SHARE_OF_FIRMS",
            "EMP",
            "FIRMDEATH_EMP",
            "FIRMDEATH_EMP_SHARE",
            "AVG_DYING_FIRM_SIZE",
            "AVG_SURVIVING_FIRM_SIZE",
        ]
    ]


def compute_survival_profile(
    age_df: pd.DataFrame, ts: pd.DataFrame, recent_years: int = 5
) -> dict:
    """Synthesize cohort tracking and FIRMDEATH data into a unified survival profile.

    Combines:
      1. Cohort tracking (years 0-5): cumulative probability of reaching each age
      2. FIRMDEATH death rates (all ages): annual mortality at each life stage
      3. Firm size context: dying vs. surviving firm size

    Returns a dict with:
      - cumulative_survival: list of (age_label, cumulative_pct) tuples from cohort method
      - death_rates_by_age: list of (age_label, annual_death_rate_pct) from FIRMDEATH
      - avg_dying_firm_size: float
      - avg_surviving_firm_size: float
      - emp_share_lost_to_deaths: float
      - recent_year_range: str
    """
    max_year = age_df["YEAR"].max()
    cutoff = max_year - recent_years + 1

    # --- Cohort tracking: cumulative survival through years 0-5 ---
    transitions = compute_survival_proxy(age_df)
    recent_trans = transitions[transitions["YEAR"] >= cutoff]

    cumulative = 100.0
    cumulative_survival = [("Year 0 (new entrants)", 100.0)]
    for from_label, to_label in zip(INDIVIDUAL_AGE_LABELS[:5], INDIVIDUAL_AGE_LABELS[1:6]):
        bucket = recent_trans[recent_trans["AGE_FROM"] == from_label]
        if bucket.empty:
            break
        avg_rate = bucket["SURVIVAL_RATE"].mean()
        cumulative = cumulative * (avg_rate / 100)
        cumulative_survival.append((f"Year {len(cumulative_survival)}", round(cumulative, 1)))

    # --- FIRMDEATH death rates by age (all buckets) ---
    death_rates = compute_death_rate_by_age(age_df)
    recent_deaths = death_rates[death_rates["YEAR"] >= cutoff]
    avg_death_by_age = (
        recent_deaths.groupby(["FAGE", "FAGE_LABEL"])["DEATH_RATE"]
        .mean()
        .reset_index()
        .sort_values("FAGE")
    )
    death_rates_by_age = list(zip(avg_death_by_age["FAGE_LABEL"], avg_death_by_age["DEATH_RATE"].round(1)))

    # --- Firm size context ---
    estab_deaths = compute_estab_vs_firm_deaths(ts)
    recent_estab = estab_deaths[estab_deaths["YEAR"] >= cutoff]

    return {
        "cumulative_survival": cumulative_survival,
        "death_rates_by_age": death_rates_by_age,
        "avg_dying_firm_size": round(recent_estab["AVG_DYING_FIRM_SIZE"].mean(), 1),
        "avg_surviving_firm_size": round(recent_estab["AVG_SURVIVING_FIRM_SIZE"].mean(), 1),
        "emp_share_lost_to_deaths": round(recent_estab["FIRMDEATH_EMP_SHARE"].mean(), 1),
        "recent_year_range": f"{int(cutoff)}-{int(max_year)}",
    }


def compute_consolidation_metrics(ts: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics showing industry consolidation over time.

    Returns a DataFrame with decade summaries showing firm count changes,
    employment changes, and average firm size growth.
    """
    df = ts.copy()
    df["DECADE"] = (df["YEAR"] // 10) * 10

    summary = (
        df.groupby("DECADE")
        .agg(
            START_YEAR=("YEAR", "min"),
            END_YEAR=("YEAR", "max"),
            START_FIRMS=("FIRM", "first"),
            END_FIRMS=("FIRM", "last"),
            START_EMP=("EMP", "first"),
            END_EMP=("EMP", "last"),
            AVG_FIRM_DEATH_RATE=("FIRMDEATH_FIRMS", lambda x: ((x / df.loc[x.index, "FIRM"]) * 100).mean()),
        )
        .reset_index()
    )
    summary["FIRM_CHANGE_PCT"] = (
        ((summary["END_FIRMS"] - summary["START_FIRMS"]) / summary["START_FIRMS"]) * 100
    ).round(1)
    summary["EMP_CHANGE_PCT"] = (
        ((summary["END_EMP"] - summary["START_EMP"]) / summary["START_EMP"]) * 100
    ).round(1)
    summary["AVG_FIRM_DEATH_RATE"] = summary["AVG_FIRM_DEATH_RATE"].round(2)
    return summary
