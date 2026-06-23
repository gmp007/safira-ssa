"""Data loading and World Bank indicator collection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from .constants import DEFAULT_SHEET_NAME, SSA_COUNTRIES, WORLD_BANK_INDICATORS
from .sai import build_composite_sai, get_standard_country_code

LOGGER = logging.getLogger(__name__)


def _require_world_bank_client():
    try:
        from pandas_datareader import wb
    except ImportError as exc:
        raise ImportError(
            "World Bank downloads require pandas-datareader. Install SAFIRA-SSA "
            "with the data extra or install pandas-datareader directly."
        ) from exc
    return wb


def fetch_all_indicators_as_one(
    indicators: Dict[str, Dict[str, str]],
    countries: List[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Fetch World Bank indicators and return one wide country-year panel."""
    wb = _require_world_bank_client()
    all_data = []

    for indicator_name, info in indicators.items():
        indicator_code = info["code"]
        print(f"Fetching {indicator_name} ({indicator_code})...")

        try:
            data = wb.download(
                indicator=indicator_code,
                country=countries,
                start=start_year,
                end=end_year,
            )
        except Exception as exc:  # pragma: no cover - network errors vary
            LOGGER.exception("Failed to fetch %s (%s)", indicator_name, indicator_code)
            print(f"Error fetching '{indicator_name}' -> '{indicator_code}': {exc}")
            continue

        if data.empty:
            LOGGER.warning("No data returned for %s (%s)", indicator_name, indicator_code)
            print(f"Warning: No data returned for '{indicator_name}' -> '{indicator_code}'.")
            continue

        data = data.reset_index().rename(columns={indicator_code: "Value"})
        data["Indicator"] = indicator_name
        all_data.append(data)

    if not all_data:
        print("No indicators returned data. Returning empty DataFrame.")
        return pd.DataFrame()

    df_long = pd.concat(all_data, ignore_index=True)
    return df_long.pivot(
        index=["year", "country"], columns="Indicator", values="Value"
    ).reset_index()


def validate_country_codes(countries: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Split country codes into World Bank-valid and invalid groups."""
    wb = _require_world_bank_client()
    available_countries = wb.get_countries()
    available_codes = set(available_countries["iso3c"].tolist())

    countries = list(countries)
    invalid = [code for code in countries if code not in available_codes]
    valid = [code for code in countries if code in available_codes]

    if invalid:
        print(f"Invalid country codes detected: {invalid}")
        LOGGER.warning("Invalid country codes detected: %s", invalid)
    else:
        print("All country codes are valid.")
        LOGGER.info("All country codes are valid.")

    return valid, invalid


def test_single_indicator_country(
    indicator_code: str,
    country_code: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Fetch one indicator-country pair as a quick World Bank connectivity check."""
    wb = _require_world_bank_client()
    data = wb.download(
        indicator=indicator_code,
        country=[country_code],
        start=start_year,
        end=end_year,
    )
    if data.empty:
        print(f"No data returned for indicator '{indicator_code}' and country '{country_code}'.")
    else:
        print(f"Data retrieved for indicator '{indicator_code}' and country '{country_code}':")
        print(data)
    return data


def collect_world_bank_panel(
    output_file: str | Path,
    indicators: Dict[str, Dict[str, str]] | None = None,
    countries: Iterable[str] | str = "SSA",
    start_year: int = 2000,
    end_year: int = 2024,
    sheet_name: str = DEFAULT_SHEET_NAME,
    validate_codes: bool = True,
    connectivity_check: bool = False,
) -> pd.DataFrame:
    """Download the configured World Bank panel and write it to Excel."""
    indicators = indicators or WORLD_BANK_INDICATORS
    if isinstance(countries, str) and countries.upper() == "SSA":
        country_codes = list(SSA_COUNTRIES)
    elif isinstance(countries, str):
        country_codes = [code.strip().upper() for code in countries.split(",") if code.strip()]
    else:
        country_codes = [str(code).strip().upper() for code in countries]

    if validate_codes:
        country_codes, invalid = validate_country_codes(country_codes)
        if invalid:
            print(f"[WARN] Ignoring invalid country codes: {invalid}")
    if not country_codes:
        raise ValueError("No valid World Bank country codes were provided.")

    if connectivity_check:
        test_single_indicator_country("NY.GDP.MKTP.KD.ZG", "NGA", start_year, end_year)

    panel = fetch_all_indicators_as_one(
        indicators=indicators,
        countries=country_codes,
        start_year=start_year,
        end_year=end_year,
    )
    if panel.empty:
        raise RuntimeError("World Bank returned no usable indicator data.")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        panel.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"All indicators saved to '{output_path}', sheet='{sheet_name}'.")
    return panel


def read_panel(path: str | Path, sheet_name: str = DEFAULT_SHEET_NAME) -> pd.DataFrame:
    """Read an Excel or CSV panel."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find data file: {data_path}")

    suffix = data_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(data_path, sheet_name=sheet_name)
    if suffix == ".csv":
        return pd.read_csv(data_path)

    raise ValueError(f"Unsupported data file extension: {data_path.suffix}")


def prepare_panel(path: str | Path, sheet_name: str = DEFAULT_SHEET_NAME) -> pd.DataFrame:
    """Load a raw indicator panel, build SAI, add ISO3 codes, and sort it."""
    df = read_panel(path, sheet_name=sheet_name)
    if "year" not in df.columns or "country" not in df.columns:
        raise ValueError("Expected columns 'year' and 'country' in the dataset.")

    df = build_composite_sai(df).dropna(subset=["SAI"]).copy()
    df["country_code"] = df["country"].apply(lambda value: get_standard_country_code(str(value)))
    df = df[df["country_code"] != "XXX"].copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    return (
        df.drop_duplicates(subset=["country_code", "year"])
        .sort_values(["country_code", "year"])
        .reset_index(drop=True)
    )


def write_sample_panel(path: str | Path) -> Path:
    """Write a compact deterministic panel for tests and smoke examples."""
    rows: list[dict[str, Any]] = []
    countries = [
        ("Nigeria", "NGA", 12.0),
        ("Kenya", "KEN", 18.0),
        ("Rwanda", "RWA", 22.0),
        ("Ghana", "GHA", 16.0),
        ("South Africa", "ZAF", 30.0),
    ]
    for country, _, offset in countries:
        for idx, year in enumerate(range(2000, 2012)):
            rows.append(
                {
                    "year": year,
                    "country": country,
                    "Literacy_Rate_Adult_Total": 45 + offset + idx * 1.4,
                    "Literacy_Rate_Youth_Total": 55 + offset + idx * 1.3,
                    "Primary_Completion_Rate_Total": 50 + offset + idx * 1.2,
                    "Enrollment_Tertiary": 6 + offset * 0.2 + idx * 0.5,
                    "R_and_D_Expenditure": 0.1 + offset * 0.005 + idx * 0.01,
                    "Secure_Internet_Servers": 20 + offset * 4 + idx * 8,
                    "Fixed_Broadband_Subscriptions": 0.5 + offset * 0.04 + idx * 0.2,
                    "Unemployment_Total": max(2.0, 18 - offset * 0.15 - idx * 0.2),
                    "GDP_Per_Capita": 900 + offset * 120 + idx * 65,
                    "Gov_Effectiveness": -1.0 + offset * 0.03 + idx * 0.02,
                    "GDP_Growth": 2.0 + idx * 0.1,
                    "Electric_Power_Consumption": 150 + offset * 10 + idx * 12,
                    "Youth_Unemployment": 25 - offset * 0.1 - idx * 0.2,
                    "High_Tech_Exports_USD": 1_000_000 + offset * 20_000 + idx * 50_000,
                    "Public_Ed_Expenditure": 3.0 + offset * 0.02 + idx * 0.03,
                    "Labor_Force_Participation": 55 + offset * 0.1 + idx * 0.05,
                    "Population_Growth": 2.8 - idx * 0.03,
                    "Life_Expectancy_Birth": 55 + offset * 0.2 + idx * 0.3,
                    "Rural_Urban_Divide_Proxy": 65 - idx * 0.5,
                    "Inflation_Consumer_Prices": 8.0 - idx * 0.1,
                }
            )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if output_path.suffix.lower() == ".csv":
        frame.to_csv(output_path, index=False)
    else:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name=DEFAULT_SHEET_NAME, index=False)
    return output_path
