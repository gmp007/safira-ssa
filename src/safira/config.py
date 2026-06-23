"""Configuration helpers for SAFIRA-SSA `.in` files."""

from __future__ import annotations

import configparser
from pathlib import Path
from typing import Iterable

from .constants import (
    DEFAULT_DATA_FILE,
    DEFAULT_FIGURES_DIR,
    DEFAULT_INPUT_FILE,
    DEFAULT_LEGACY_MODEL,
    DEFAULT_MODEL_ASSETS,
    DEFAULT_MODEL_WEIGHTS,
    DEFAULT_PACKAGED_DATA_FILE,
)


DEFAULT_INPUT_TEXT = f"""# SAFIRA-SSA input file
# Lines beginning with # are comments. Values can be changed without editing Python code.

[workflow]
data_mode = packaged
train_model = true
forecast = true
make_plots = true
zip_figures = true
write_sample_data = false

[paths]
data_file = {DEFAULT_DATA_FILE}
packaged_data_file = {DEFAULT_PACKAGED_DATA_FILE}
sheet_name = All_Indicators
figures_dir = {DEFAULT_FIGURES_DIR}
model_weights = {DEFAULT_MODEL_WEIGHTS}
model_assets = {DEFAULT_MODEL_ASSETS}
legacy_model = {DEFAULT_LEGACY_MODEL}
sample_data_file = examples/sample_sai_panel.xlsx
world_file =

[data]
start_year = 2000
end_year = 2026
countries = SSA
validate_country_codes = true
connectivity_check = false

[model]
lookback = 5
hidden_dim = 32
num_layers = 4
dropout = 0.25
bidirectional = false
epochs = 1000
early_stopping = false
patience = 50
min_delta = 1e-5
batch_size = 16
learning_rate = 1e-3
use_pretrained = false
plot_losses = false
input_features = SAI_scaled

[forecast]
country = Nigeria
year = 2025

[plots]
show = false
selected = choropleth, spaghetti, pillar_heatmap, box, gdp_scatter, gdp_scatter_2020, gdp_scatter_highlight, gdp_scatter_top10, corr, corr_lower, slope, radar, umap, forecast_diag, nigeria_comparison, nigeria_comparison_seaborn, forecast_diag_zaf, spaghetti_selected, spaghetti_selected_alt, future_projection
forecast_diag_countries = NGA, CIV, KEN, CMR, RWA, GHA
selected_country_codes = KEN, RWA, GHA, ZAF, NGA, COD
nigeria_peers = RWA, UGA, KEN, GHA
future_horizon = 10
figures_zip = figures.zip
"""


def read_config(path: str | Path = DEFAULT_INPUT_FILE) -> configparser.ConfigParser:
    """Read a SAFIRA-SSA input file."""
    config = configparser.ConfigParser()
    config.optionxform = str
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    config.read(file_path)
    return config


def write_default_input(path: str | Path = DEFAULT_INPUT_FILE, overwrite: bool = False) -> Path:
    """Write the default SAFIRA-SSA input file."""
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing input file: {output_path}")
    output_path.write_text(DEFAULT_INPUT_TEXT, encoding="utf-8")
    return output_path


def cfg_get(config: configparser.ConfigParser, section: str, option: str, fallback: str = "") -> str:
    """Read a string option and strip whitespace."""
    return config.get(section, option, fallback=fallback).strip()


def cfg_bool(config: configparser.ConfigParser, section: str, option: str, fallback: bool = False) -> bool:
    """Read a boolean option."""
    return config.getboolean(section, option, fallback=fallback)


def cfg_int(config: configparser.ConfigParser, section: str, option: str, fallback: int = 0) -> int:
    """Read an integer option."""
    return config.getint(section, option, fallback=fallback)


def cfg_float(config: configparser.ConfigParser, section: str, option: str, fallback: float = 0.0) -> float:
    """Read a float option."""
    return config.getfloat(section, option, fallback=fallback)


def cfg_list(
    config: configparser.ConfigParser,
    section: str,
    option: str,
    fallback: Iterable[str] | str = (),
) -> list[str]:
    """Read a comma-separated list option."""
    if isinstance(fallback, str):
        fallback_text = fallback
    else:
        fallback_text = ", ".join(fallback)
    raw = config.get(section, option, fallback=fallback_text)
    return [part.strip() for part in raw.split(",") if part.strip()]
