# SAFIRA-SSA

**Skill Advancement Forecasting and Intelligence for Readiness in Africa, Sub-Saharan Africa**

SAFIRA-SSA packages the notebook workflow for *Forecasting the Future of Skills in Sub-Saharan Africa: AI, Automation, and the Skill Advancement Index* into a reusable Python project. It uses a packaged World Bank snapshot by default, can refresh that data from the World Bank API when internet access is available, builds the Skill Advancement Index (SAI), trains or reloads an LSTM forecaster, produces country-year forecasts, and generates the figures used to compare SAI trajectories, pillars, regions, and country profiles.

The code was refactored from the original `PSAIM_WB_Forecast.ipynb` notebook into a modern package layout so it can be installed with `pip install .`, run from a command line, tested, and hosted on GitHub/PyPI.

## Authors

- **Chinedu Ekuma**
- **Kelechi Ekuma**

## What SAFIRA-SSA Does

SAFIRA-SSA supports a complete research-to-analysis workflow:

1. Loads a packaged country-year World Bank panel for Sub-Saharan Africa, or downloads a fresh panel on demand.
2. Builds the **Skill Advancement Index (SAI)** on a 0 to 100 scale.
3. Estimates four SAI dimensions:
   - **Foundational**: literacy and primary completion.
   - **Advanced**: tertiary enrollment and research capacity.
   - **Digital**: secure internet infrastructure and fixed broadband access.
   - **Labor**: labor-market alignment through reversed unemployment.
4. Trains an LSTM sequence model over country SAI histories.
5. Saves model weights and non-tensor assets separately for safer reloads.
6. Forecasts future SAI values for a selected country and year.
7. Generates the major comparison plots from the notebook, including trajectories, pillar heatmaps, regional boxes, GDP-SAI scatter plots, correlation plots, slope plots, radar plots, UMAP plots, forecast diagnostics, Nigeria peer comparisons, and SSA median future projections.

## Repository Layout

```text
safira-ssa/
├── src/safira/
│   ├── cli.py          # Command line entry point
│   ├── config.py       # .in file parsing and defaults
│   ├── constants.py    # SSA countries, aliases, and World Bank indicators
│   ├── data.py         # World Bank download, panel loading, sample data
│   ├── forecast.py     # Training, model loading, and forecasting
│   ├── models.py       # PyTorch datasets and LSTM models
│   ├── plotting.py     # All figure-generation routines
│   ├── resources/      # Packaged offline World Bank snapshot and metadata
│   └── sai.py          # SAI construction and country normalization
├── tests/              # Unit tests for index construction and config parsing
├── examples/           # Sample input for a fast local smoke run
├── data/               # Default location for downloaded World Bank data
├── models/             # Default location for trained model artifacts
├── safira.in           # Main editable workflow input file
├── setup.cfg
├── setup.py
└── MANIFEST.in
```

## Installation

From the repository root:

```bash
pip install .
```

For geospatial maps and UMAP plots, install the optional extras:

```bash
pip install ".[full]"
```

For development tests and packaging checks:

```bash
pip install ".[dev]"
```

## Quick Start

Create or refresh the default input file:

```bash
safira write-input --output safira.in --overwrite
```

Run the workflow described by the input file:

```bash
safira run --input safira.in
```

The default `safira.in` uses the packaged snapshot, so this command can run without internet access after installation. To refresh the World Bank data into `data/ssa_sai_all_indicators.xlsx`, set `data_mode = download` in `safira.in` or run only the download step:

```bash
safira fetch-data --input safira.in
```

Write the deterministic sample panel used by examples and tests:

```bash
safira write-sample-data --output examples/sample_sai_panel.xlsx
```

## The `safira.in` Input File

The input file is the main user-facing control surface. It lets users run the same workflow in different ways without editing Python code.

Important sections:

- `[workflow]`: switches for data source mode, training the model, forecasting, plotting, zipping figures, and writing sample data.
- `[paths]`: locations for the custom/downloaded data file, packaged data URI, model weights, model assets, legacy model file, figures directory, sample data file, and optional Natural Earth world file.
- `[data]`: World Bank years, country selection, country-code validation, and optional connectivity check.
- `[model]`: LSTM lookback, hidden dimension, layers, dropout, epochs, learning rate, early stopping, batch size, and input features.
- `[forecast]`: country and year requested for forecast or historical SAI lookup.
- `[plots]`: selected plot routines and country lists for comparison plots.

The default `safira.in` follows the original notebook defaults as closely as possible: the packaged 2000-2025 World Bank snapshot, univariate `SAI_scaled` LSTM input, Nigeria 2025 forecast, and the full plot set from the notebook's “MAIN - call whichever plots you want” block.

The data source is controlled by one line:

```ini
[workflow]
data_mode = packaged
```

Available modes:

- `packaged`: use the offline snapshot installed inside the Python package.
- `download`: download the requested World Bank panel to `[paths] data_file`, then use that file.
- `custom`: use the user-supplied file at `[paths] data_file`.

## Data Workflow

The package ships with this offline data resource:

```text
package://safira/resources/safira_ssa_worldbank_snapshot.xlsx
```

The companion metadata file is:

```text
package://safira/resources/safira_ssa_worldbank_snapshot_metadata.json
```

The packaged snapshot was generated on **2026-06-23** from the World Bank API. It requests 2000-2026 data, and the resulting file currently contains 2000-2025 observations because 2026 country-year values were not yet present in the returned World Bank series at generation time.

Snapshot contents:

- 46 Sub-Saharan African countries.
- 73 configured indicators.
- 1,194 raw country-year rows.
- 1,089 rows after SAI construction drops rows without enough pillar information.
- World Development Indicators source last updated by World Bank on 2026-04-08.
- Worldwide Governance Indicators source last updated by World Bank on 2026-03-18.

Live refreshes use the official World Bank JSON API directly. Standard WDI indicators are read through the country-indicator endpoint, while the governance indicators are read from World Bank source `3`, Worldwide Governance Indicators, using the correct `GOV_WGI_*` source series IDs. The indicator list lives in `src/safira/constants.py` and was extracted directly from the notebook to avoid transcription drift.

Default output:

```text
data/ssa_sai_all_indicators.xlsx
```

Default sheet:

```text
All_Indicators
```

The generated panel contains one row per country-year and one column per World Bank indicator. SAFIRA-SSA then computes SAI and the four dimension columns during loading:

```text
dim_foundational
dim_advanced
dim_digital
dim_labor
SAI
country_code
```

To work with your own panel, set:

```ini
[workflow]
data_mode = custom

[paths]
data_file = path/to/your_panel.xlsx
```

Custom Excel files should contain the `All_Indicators` sheet unless `[paths] sheet_name` is changed. CSV files are also supported.

## Forecasting Workflow

The forecasting class is `safira.forecast.SkillAdvancementForecaster`. The legacy notebook class name `psai` remains available as an alias.

Model artifacts are split into:

```text
models/time_series_sai_model_weights.pth
models/time_series_sai_assets.pkl
```

The `.pth` file stores PyTorch tensor weights. The `.pkl` file stores non-tensor assets such as input feature names, lookback, hyperparameters, the target scaler, and exogenous feature scalers. This is safer and easier to reload than a single mixed object checkpoint.

Programmatic example:

```python
from safira.forecast import SkillAdvancementForecaster

model = SkillAdvancementForecaster(
    data_path="package://safira/resources/safira_ssa_worldbank_snapshot.xlsx",
    lookback=5,
    epochs=1000,
    input_features=["SAI_scaled"],
)

result = model.run(country="Nigeria", year=2025)
print(result.value)
```

## Plot Selection

The `[plots] selected` line in `safira.in` controls which figures are produced. Available plot names are:

```text
choropleth
spaghetti
pillar_heatmap
box
gdp_scatter
gdp_scatter_2020
gdp_scatter_highlight
gdp_scatter_top10
corr
corr_lower
slope
radar
umap
forecast_diag
nigeria_comparison
nigeria_comparison_seaborn
forecast_diag_zaf
spaghetti_selected
spaghetti_selected_alt
future_projection
```

Plots that require unavailable optional dependencies or missing model files are skipped with an explicit warning. This allows long figure batches to continue when, for example, geospatial dependencies are not installed.

## Sample Workflow

The repository includes a small deterministic sample input:

```bash
safira run --input examples/safira_sample.in
```

That sample writes `examples/sample_sai_panel.xlsx` when run and keeps training/forecasting/plotting disabled by default. It is meant to prove the input-file and data-preparation path without requiring internet access or PyTorch.

To test plotting on the sample panel, edit `examples/safira_sample.in`:

```ini
[workflow]
make_plots = true
```

To test model training on the sample panel, install the forecasting dependencies and set:

```ini
[workflow]
train_model = true
forecast = true
```

## Tests

Run the lightweight test suite:

```bash
python -m unittest discover -s tests
```

The tests focus on deterministic pieces that should work without network access: country-code normalization, SAI construction, config parsing, sample data writing, and panel preparation.

## Packaging

Build a source distribution and wheel:

```bash
python -m build
```

Check the package metadata:

```bash
twine check dist/*
```

Install locally from the working tree:

```bash
pip install .
```

## Notes on Reproducibility

- The packaged snapshot is fixed at build time so offline users can reproduce the bundled run. Use `data_mode = download` or `safira fetch-data` to refresh from the live World Bank API.
- Live World Bank results can change if the World Bank revises historical series, changes indicator coverage, or adds newer years.
- The notebook's original full workflow trains an LSTM, so exact model weights may vary unless the runtime, dependencies, random seeds, and hardware behavior are fixed.
- The sample panel is synthetic and exists only for testing package mechanics. It is not a substitute for the World Bank data used in the actual research workflow.

## License

SAFIRA-SSA is distributed under the GNU General Public License version 3.
