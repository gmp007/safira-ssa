"""Skill Advancement Index construction and country-code normalization."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

import numpy as np
import pandas as pd

from .constants import COUNTRY_ALIASES


def normalize_country_key(value: str) -> str:
    """Return a punctuation- and accent-insensitive country key."""
    text = (value or "").strip()
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


_NORMALIZED_COUNTRY_ALIASES = {
    normalize_country_key(name): code for name, code in COUNTRY_ALIASES.items()
}


def get_standard_country_code(
    user_input: str,
    aliases: Dict[str, str] = COUNTRY_ALIASES,
    unknown_code: str = "XXX",
) -> str:
    """Resolve a country name or ISO3 code to the standard SSA ISO3 code."""
    key = normalize_country_key(user_input)
    code = _NORMALIZED_COUNTRY_ALIASES.get(key)
    if code:
        return code

    if aliases is not COUNTRY_ALIASES:
        alt = {normalize_country_key(name): iso3 for name, iso3 in aliases.items()}
        return alt.get(key, unknown_code)

    return unknown_code


def _as_percent(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    max_value = values.max(skipna=True)
    if pd.notna(max_value) and max_value <= 1.5:
        values = values * 100.0
    return values


def _minmax_0_100(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(np.nan, index=values.index)

    mn = float(np.nanmin(values.values))
    mx = float(np.nanmax(values.values))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return pd.Series(np.nan, index=values.index)
    return (values - mn) / (mx - mn) * 100.0


def _log_minmax_0_100(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").clip(lower=0)
    values = np.log1p(values)
    return _minmax_0_100(pd.Series(values, index=series.index))


def _mean_of_available(df: pd.DataFrame, columns: List[str], mode: str) -> pd.Series:
    present = [column for column in columns if column in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index)

    scaled = []
    for column in present:
        if mode == "percent_minmax":
            scaled.append(_minmax_0_100(_as_percent(df[column])))
        elif mode == "log_minmax":
            scaled.append(_log_minmax_0_100(df[column]))
        else:
            scaled.append(_minmax_0_100(df[column]))
    return pd.concat(scaled, axis=1).mean(axis=1, skipna=True)


def build_composite_sai(df: pd.DataFrame) -> pd.DataFrame:
    """Build the four-pillar Skill Advancement Index on a 0 to 100 scale."""
    frame = df.copy()

    foundational_cols = [
        "Literacy_Rate_Adult_Total",
        "Literacy_Rate_Youth_Total",
        "Primary_Completion_Rate_Total",
    ]
    advanced_cols = ["Enrollment_Tertiary", "R_and_D_Expenditure"]
    digital_count_cols = ["Secure_Internet_Servers"]
    digital_percent_cols = ["Fixed_Broadband_Subscriptions"]

    frame["dim_foundational"] = _mean_of_available(
        frame, foundational_cols, mode="percent_minmax"
    )
    frame["dim_advanced"] = _mean_of_available(
        frame, advanced_cols, mode="percent_minmax"
    )

    digital_counts = _mean_of_available(frame, digital_count_cols, mode="log_minmax")
    digital_percent = _mean_of_available(
        frame, digital_percent_cols, mode="percent_minmax"
    )
    frame["dim_digital"] = pd.concat([digital_counts, digital_percent], axis=1).mean(
        axis=1, skipna=True
    )

    if "Unemployment_Total" in frame.columns:
        unemployment = _as_percent(frame["Unemployment_Total"])
        unemployment_norm = _minmax_0_100(unemployment)
        frame["dim_labor"] = (100.0 - unemployment_norm).clip(lower=0, upper=100)
    else:
        frame["dim_labor"] = np.nan

    dimensions = ["dim_foundational", "dim_advanced", "dim_digital", "dim_labor"]
    for dimension in dimensions:
        frame[dimension] = frame[dimension].clip(lower=0, upper=100)

    frame["SAI"] = frame[dimensions].mean(axis=1, skipna=True).clip(lower=0, upper=100)
    return frame


build_composite_SAI = build_composite_sai
