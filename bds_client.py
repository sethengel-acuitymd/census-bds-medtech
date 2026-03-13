"""Client for the Census Bureau Business Dynamics Statistics (BDS) API."""

import os
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://api.census.gov/data/timeseries/bds"


def _get_api_key() -> str | None:
    return os.environ.get("CENSUS_API_KEY")


def query(
    variables: list[str],
    naics: str = "3391",
    year: str = "*",
    geo: str = "us:1",
    predicates: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Query the BDS API and return a DataFrame.

    Args:
        variables: List of variable names to fetch (e.g. ["FIRM", "EMP"]).
        naics: NAICS code to filter by. Default "3391" (Medical Equipment & Supplies).
        year: Year filter. Use "*" for all years, or "2020,2021" for specific years.
        geo: Geography in "level:code" format. Default "us:1" (national).
        predicates: Additional predicates (e.g. {"FAGE": "*"} for firm age breakdown).
    """
    params: dict[str, Any] = {
        "get": ",".join(variables),
        "for": geo,
        "NAICS": naics,
        "YEAR": year,
    }
    if predicates:
        params.update(predicates)
    key = _get_api_key()
    if key:
        params["key"] = key

    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    headers = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=headers)

    # The API can return duplicate columns (e.g. FAGE as both a variable and predicate).
    # Drop duplicate columns, keeping the first occurrence.
    df = df.loc[:, ~df.columns.duplicated()]

    # Convert numeric columns (skip labels, codes, and classification fields)
    skip_cols = {"NAICS_LABEL", "FAGE_LABEL", "EAGE_LABEL", "FAGE", "EAGE", "EMPSZES_LABEL", "EMPSZFI_LABEL", "NAME"}
    numeric_cols = [c for c in variables if c not in skip_cols]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "YEAR" in df.columns:
        df["YEAR"] = df["YEAR"].astype(int)
        df = df.sort_values("YEAR").reset_index(drop=True)

    return df


def get_timeseries(naics: str = "3391") -> pd.DataFrame:
    """Get full time series of core metrics for a NAICS code."""
    return query(
        variables=[
            "FIRM",
            "ESTAB",
            "EMP",
            "JOB_CREATION",
            "JOB_CREATION_RATE",
            "JOB_DESTRUCTION",
            "JOB_DESTRUCTION_RATE",
            "NET_JOB_CREATION",
            "NET_JOB_CREATION_RATE",
            "ESTABS_ENTRY",
            "ESTABS_ENTRY_RATE",
            "ESTABS_EXIT",
            "ESTABS_EXIT_RATE",
            "FIRMDEATH_FIRMS",
            "FIRMDEATH_EMP",
            "NAICS_LABEL",
        ],
        naics=naics,
    )


def get_firm_age_timeseries(naics: str = "3391") -> pd.DataFrame:
    """Get firm age breakdown across all years for a NAICS code."""
    return query(
        variables=[
            "FIRM",
            "ESTAB",
            "EMP",
            "JOB_CREATION",
            "JOB_DESTRUCTION",
            "FIRMDEATH_FIRMS",
            "NAICS_LABEL",
            "FAGE",
            "FAGE_LABEL",
        ],
        naics=naics,
        predicates={"FAGE": "*"},
    )


def get_comparison(naics_codes: list[str], year: str = "2023") -> pd.DataFrame:
    """Get metrics for multiple NAICS codes for comparison."""
    frames = []
    for code in naics_codes:
        df = query(
            variables=["FIRM", "ESTAB", "EMP", "NET_JOB_CREATION", "FIRMDEATH_FIRMS", "NAICS_LABEL"],
            naics=code,
            year=year,
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
