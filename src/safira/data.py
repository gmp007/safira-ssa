"""Data loading and World Bank indicator collection."""

from __future__ import annotations

import logging
import time
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from .constants import DEFAULT_PACKAGED_DATA_FILE, DEFAULT_SHEET_NAME, SSA_COUNTRIES, WORLD_BANK_INDICATORS
from .sai import build_composite_sai, get_standard_country_code

LOGGER = logging.getLogger(__name__)
PACKAGE_DATA_PREFIX = "package://safira/"
WORLD_BANK_API = "https://api.worldbank.org/v2"


def _require_requests():
    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "World Bank downloads require requests. Install SAFIRA-SSA with its default dependencies."
        ) from exc
    return requests


def _world_bank_payload(path: str, params: dict[str, Any]) -> Any:
    """Request one World Bank API page with a few retries."""
    requests = _require_requests()
    response = None
    for attempt in range(4):
        try:
            response = requests.get(f"{WORLD_BANK_API}/{path}", params=params, timeout=180)
            response.raise_for_status()
            break
        except requests.RequestException:  # pragma: no cover - depends on network behavior
            if attempt == 3:
                raise
            time.sleep(2 + attempt * 3)
    if response is None:
        raise RuntimeError(f"World Bank request failed for {path}.")
    return response.json()


def _world_bank_records(path: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Read all pages from a World Bank API endpoint."""
    query = {"format": "json", "per_page": 20000}
    if params:
        query.update(params)

    records: list[dict[str, Any]] = []
    page = 1
    while True:
        query["page"] = page
        payload = _world_bank_payload(path, query)
        if not isinstance(payload, list) or len(payload) < 2:
            raise RuntimeError(f"Unexpected World Bank response for {path}.")

        metadata = payload[0] or {}
        data = payload[1] or []
        records.extend(data)

        total_pages = int(metadata.get("pages") or 1)
        if page >= total_pages:
            break
        page += 1

    return records


def _world_bank_source_records(path: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Read source-specific records such as the Worldwide Governance Indicators."""
    query = {"format": "json", "per_page": 20000}
    if params:
        query.update(params)

    records: list[dict[str, Any]] = []
    page = 1
    while True:
        query["page"] = page
        payload = _world_bank_payload(path, query)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected World Bank source response for {path}.")

        source = payload.get("source") or {}
        data = source.get("data") or []
        records.extend(data)

        total_pages = int(payload.get("pages") or 1)
        if page >= total_pages:
            break
        page += 1

    return records


def _source_variable(record: dict[str, Any], concept: str) -> dict[str, Any]:
    """Return one concept variable from a source-specific World Bank record."""
    for variable in record.get("variable", []):
        if str(variable.get("concept", "")).lower() == concept.lower():
            return variable
    return {}


def _source_records_to_frame(
    records: list[dict[str, Any]],
    indicator_name: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Convert source-specific records to the standard long panel format."""
    rows = []
    for record in records:
        country = _source_variable(record, "Country")
        time_value = _source_variable(record, "Time")
        year_text = str(time_value.get("value") or time_value.get("id") or "").replace("YR", "")
        year = pd.to_numeric(year_text, errors="coerce")
        if pd.isna(year):
            continue
        year = int(year)
        if year < start_year or year > end_year:
            continue
        rows.append(
            {
                "year": year,
                "country": country.get("value"),
                "Indicator": indicator_name,
                "Value": record.get("value"),
            }
        )

    data = pd.DataFrame(rows)
    if data.empty:
        return data
    data["Value"] = pd.to_numeric(data["Value"], errors="coerce")
    return data.dropna(subset=["year", "country"])


def fetch_all_indicators_as_one(
    indicators: Dict[str, Dict[str, str]],
    countries: List[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Fetch World Bank indicators and return one wide country-year panel."""
    all_data = []
    country_part = ";".join(countries)

    for indicator_name, info in indicators.items():
        indicator_code = info["code"]
        print(f"Fetching {indicator_name} ({indicator_code})...")

        try:
            source_id = str(info.get("source_id", "")).strip()
            if source_id:
                source_code = str(info.get("source_code") or indicator_code)
                records = _world_bank_source_records(
                    f"sources/{source_id}/Country/{country_part}/Series/{source_code}"
                )
                data = _source_records_to_frame(records, indicator_name, start_year, end_year)
            else:
                records = _world_bank_records(
                    f"country/{country_part}/indicator/{indicator_code}",
                    {"date": f"{start_year}:{end_year}"},
                )
                rows = []
                for record in records:
                    country = record.get("country") or {}
                    rows.append(
                        {
                            "year": record.get("date"),
                            "country": country.get("value"),
                            "Indicator": indicator_name,
                            "Value": record.get("value"),
                        }
                    )
                data = pd.DataFrame(rows)
        except Exception as exc:  # pragma: no cover - network errors vary
            LOGGER.exception("Failed to fetch %s (%s)", indicator_name, indicator_code)
            print(f"Error fetching '{indicator_name}' -> '{indicator_code}': {exc}")
            continue

        if not records:
            LOGGER.warning("No data returned for %s (%s)", indicator_name, indicator_code)
            print(f"Warning: No data returned for '{indicator_name}' -> '{indicator_code}'.")
            continue

        if data.empty:
            LOGGER.warning("No usable rows returned for %s (%s)", indicator_name, indicator_code)
            print(f"Warning: No usable rows returned for '{indicator_name}' -> '{indicator_code}'.")
            continue

        data["year"] = pd.to_numeric(data["year"], errors="coerce")
        data["Value"] = pd.to_numeric(data["Value"], errors="coerce")
        data = data.dropna(subset=["year", "country"])
        data["year"] = data["year"].astype(int)
        all_data.append(data)

    if not all_data:
        print("No indicators returned data. Returning empty DataFrame.")
        return pd.DataFrame()

    df_long = pd.concat(all_data, ignore_index=True)
    panel = df_long.pivot_table(
        index=["year", "country"], columns="Indicator", values="Value", aggfunc="first"
    ).reset_index()
    panel.columns.name = None
    return panel.sort_values(["country", "year"]).reset_index(drop=True)


def validate_country_codes(countries: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Split country codes into World Bank-valid and invalid groups."""
    records = _world_bank_records("country", {"per_page": 400})
    available_codes = {str(record.get("id", "")).upper() for record in records}

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
    records = _world_bank_records(
        f"country/{country_code}/indicator/{indicator_code}",
        {"date": f"{start_year}:{end_year}"},
    )
    rows = [
        {
            "year": int(record["date"]),
            "country": (record.get("country") or {}).get("value"),
            indicator_code: record.get("value"),
        }
        for record in records
        if record.get("date") is not None
    ]
    data = pd.DataFrame(rows)
    if data.empty:
        print(f"No data returned for indicator '{indicator_code}' and country '{country_code}'.")
    else:
        print(f"Data retrieved for indicator '{indicator_code}' and country '{country_code}':")
        print(data)
    return data


def _package_resource(path: str | Path):
    """Return a package resource when the path uses SAFIRA's package URI."""
    text = str(path)
    if text == "packaged":
        text = DEFAULT_PACKAGED_DATA_FILE
    if not text.startswith(PACKAGE_DATA_PREFIX):
        return None
    relative_path = text[len(PACKAGE_DATA_PREFIX) :].lstrip("/")
    resource = resources.files("safira").joinpath(relative_path)
    if not resource.is_file():
        raise FileNotFoundError(f"Cannot find packaged data resource: {text}")
    return resource


def collect_world_bank_panel(
    output_file: str | Path,
    indicators: Dict[str, Dict[str, str]] | None = None,
    countries: Iterable[str] | str = "SSA",
    start_year: int = 2000,
    end_year: int = 2026,
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
    resource = _package_resource(path)
    if resource is not None:
        suffix = Path(resource.name).suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            with resource.open("rb") as handle:
                return pd.read_excel(handle, sheet_name=sheet_name)
        if suffix == ".csv":
            with resource.open("r", encoding="utf-8") as handle:
                return pd.read_csv(handle)
        raise ValueError(f"Unsupported packaged data extension: {suffix}")

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
