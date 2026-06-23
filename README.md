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

SAFIRA-SSA requires Python 3.10 or newer. Install from the repository root:

```bash
pip install .
```

That basic install is enough for data loading, SAI construction, model training, forecasting, and most plots. Optional install choices are:

| Command | What it adds |
| --- | --- |
| `pip install .` | Core package, CLI, packaged data, training, forecasting, and standard plots. |
| `pip install ".[geo]"` | Geospatial dependencies for choropleth map plotting. |
| `pip install ".[umap]"` | UMAP dependency for latent country-cluster plots. |
| `pip install ".[full]"` | Both geospatial and UMAP optional plotting dependencies. |
| `pip install ".[dev]"` | Developer tools for tests, source/wheel builds, and package checks. |
| `pip install ".[full,dev]"` | Everything needed for full plotting plus development checks. |

The quotes around extras are recommended on macOS shells such as `zsh`.

After installation, the command-line tool is:

```bash
safira
```

To confirm the CLI is visible:

```bash
safira --help
```

## Quick Start

The default workflow uses the packaged World Bank snapshot, so it can run without downloading data from the internet.

Create or refresh the editable input file:

```bash
safira write-input --output safira.in --overwrite
```

Run the workflow described by `safira.in`:

```bash
safira run --input safira.in
```

The run can train the model, produce the requested forecast, make figures, and zip the figures depending on the switches in `[workflow]`.

To download fresh World Bank data before a run:

```bash
safira fetch-data --input safira.in
```

That command writes the downloaded panel to the file named by `[paths] data_file`, usually:

```text
data/ssa_sai_all_indicators.xlsx
```

Write the deterministic sample panel used by examples and tests:

```bash
safira write-sample-data --output examples/sample_sai_panel.xlsx
```

## CLI Commands

The command line has four main commands:

| Command | What it does |
| --- | --- |
| `safira run --input safira.in` | Runs the workflow described in the input file. Depending on `[workflow]`, this can download data, train, forecast, plot, and zip figures. |
| `safira fetch-data --input safira.in` | Downloads only the World Bank panel and writes it to `[paths] data_file`. It does not train, forecast, or plot. |
| `safira write-input --output safira.in --overwrite` | Writes a fresh editable input file. Use this if you want to restore the default configuration. |
| `safira write-sample-data --output examples/sample_sai_panel.xlsx` | Writes a small synthetic data file for quick package checks. This is not research data. |

## The `safira.in` Input File

The input file is the main user-facing control surface. It lets users run the same workflow in different ways without editing Python code.

SAFIRA-SSA input files use the simple `.ini` format:

- Section names appear in square brackets, for example `[workflow]`.
- Each setting is written as `name = value`.
- `true` and `false` are used for on/off choices.
- Lists are comma-separated, for example `NGA, KEN, GHA`.
- Relative paths are interpreted from the directory where you run the command.
- Lines beginning with `#` are comments.

You can create a fresh copy at any time:

```bash
safira write-input --output safira.in --overwrite
```

The default `safira.in` follows the original notebook defaults as closely as possible: the packaged 2000-2025 World Bank snapshot, univariate `SAI_scaled` LSTM input, Nigeria 2025 forecast, and the full plot set from the notebook's “MAIN - call whichever plots you want” block.

### Common Data Setups

Use the packaged offline snapshot:

```ini
[workflow]
data_mode = packaged
```

This is the safest first run because it does not require internet access.

Download fresh World Bank data and run everything in one command:

```ini
[workflow]
data_mode = download
```

```bash
safira run --input safira.in
```

SAFIRA-SSA will first download the requested World Bank panel to `[paths] data_file`, then use that file for training, forecasting, and plotting.

Download fresh World Bank data first, then run later from the downloaded file:

```ini
[workflow]
data_mode = download
```

```bash
safira fetch-data --input safira.in
```

After the download finishes, change only this line:

```ini
[workflow]
data_mode = custom
```

Then run:

```bash
safira run --input safira.in
```

This two-step approach is useful when you want to confirm that the data download succeeded before spending time on training and plotting.

Use your own Excel or CSV panel:

```ini
[workflow]
data_mode = custom

[paths]
data_file = path/to/your_panel.xlsx
```

Custom Excel files should contain the `All_Indicators` sheet unless `[paths] sheet_name` is changed. CSV files are also supported.

### Complete Input Reference

Every option in the default `safira.in` is described below.

#### `[workflow]`

Controls which major steps run.

| Option | Default | Meaning |
| --- | --- | --- |
| `data_mode` | `packaged` | Chooses the data source. Use `packaged`, `download`, or `custom`. |
| `train_model` | `true` | Trains an LSTM model from the selected data. |
| `forecast` | `true` | Produces the requested country-year forecast or historical lookup. |
| `make_plots` | `true` | Generates the plot names listed in `[plots] selected`. |
| `zip_figures` | `true` | Creates a zip archive of generated figures. |
| `write_sample_data` | `false` | Writes a small deterministic sample file to `[paths] sample_data_file`. Useful for smoke tests. |

`data_mode` choices:

| Value | What happens |
| --- | --- |
| `packaged` | Uses the snapshot installed inside the package at `[paths] packaged_data_file`. No internet required. |
| `download` | Downloads World Bank data to `[paths] data_file`, then uses that downloaded file. Internet required. |
| `custom` | Uses the existing file at `[paths] data_file`. No download is attempted. |

#### `[paths]`

Controls where files are read from and written to.

| Option | Default | Meaning |
| --- | --- | --- |
| `data_file` | `data/ssa_sai_all_indicators.xlsx` | Local Excel or CSV file used when `data_mode = custom`; also the output file created when `data_mode = download` or `safira fetch-data` is used. |
| `packaged_data_file` | `package://safira/resources/safira_ssa_worldbank_snapshot.xlsx` | Built-in offline World Bank snapshot used when `data_mode = packaged`. |
| `sheet_name` | `All_Indicators` | Excel sheet containing the wide country-year indicator panel. Ignored for CSV files. |
| `figures_dir` | `figures` | Directory where plots are saved. Created automatically if needed. |
| `model_weights` | `models/time_series_sai_model_weights.pth` | PyTorch model weights file written after training and read when loading a saved model. |
| `model_assets` | `models/time_series_sai_assets.pkl` | Non-tensor model assets such as scalers, feature names, and training settings. |
| `legacy_model` | `models/time_series_sai_model.pth` | Legacy single-file model path kept for compatibility with older notebook-style saves. |
| `sample_data_file` | `examples/sample_sai_panel.xlsx` | Output path used when `write_sample_data = true` or when running `safira write-sample-data`. |
| `world_file` | blank | Optional local Natural Earth world boundary file for choropleth maps. Leave blank unless you have a local map file. |

#### `[data]`

Controls World Bank downloads. These settings matter only when `data_mode = download` or `safira fetch-data` is used.

| Option | Default | Meaning |
| --- | --- | --- |
| `start_year` | `2000` | First year requested from the World Bank API. |
| `end_year` | `2026` | Last year requested from the World Bank API. If the World Bank has no values for the latest year yet, the downloaded file will simply stop at the latest available year. |
| `countries` | `SSA` | Countries to download. Use `SSA` for the built-in Sub-Saharan Africa country list, or provide comma-separated ISO3 codes such as `NGA, KEN, GHA`. |
| `validate_country_codes` | `true` | Checks country codes against the World Bank before downloading. Invalid codes are ignored with a warning. |
| `connectivity_check` | `false` | Runs a quick test download for one Nigeria GDP-growth series before the full download. Useful for diagnosing internet/API access. |

#### `[model]`

Controls the LSTM forecasting model.

| Option | Default | Meaning |
| --- | --- | --- |
| `lookback` | `5` | Number of past yearly observations used to predict the next SAI value. |
| `hidden_dim` | `32` | Width of the LSTM hidden layer. Larger values can model more complexity but may train more slowly. |
| `num_layers` | `4` | Number of stacked LSTM layers. |
| `dropout` | `0.25` | Dropout rate used for regularization. Use `0.0` to disable dropout. |
| `bidirectional` | `false` | Whether to use a bidirectional LSTM. For forecasting, `false` is the conservative default. |
| `epochs` | `1000` | Maximum number of training epochs. Reduce this for quick tests. |
| `early_stopping` | `false` | Stops training early when validation loss stops improving. |
| `patience` | `50` | Number of epochs to wait for improvement when early stopping is enabled. |
| `min_delta` | `1e-5` | Minimum validation-loss improvement counted by early stopping. |
| `batch_size` | `16` | Number of training sequences per optimization batch. |
| `learning_rate` | `1e-3` | Optimizer learning rate. |
| `use_pretrained` | `false` | Loads existing model weights and assets instead of training from scratch. Requires files at `model_weights` and `model_assets`. |
| `plot_losses` | `false` | Shows or saves training-loss diagnostics when supported by the runtime. |
| `input_features` | `SAI_scaled` | Comma-separated model input features. The default univariate model uses scaled SAI only. |

For a quick test run, set:

```ini
[model]
epochs = 1
hidden_dim = 8
num_layers = 1
dropout = 0.0
```

#### `[forecast]`

Controls the printed forecast result.

| Option | Default | Meaning |
| --- | --- | --- |
| `country` | `Nigeria` | Country to forecast or look up. A country name or recognized ISO3-style alias can be used. |
| `year` | `2025` | Target year. If the year is already in the prepared data, SAFIRA-SSA reports the observed SAI; otherwise it forecasts forward. |

#### `[plots]`

Controls figure generation.

| Option | Default | Meaning |
| --- | --- | --- |
| `show` | `false` | Whether plots should open interactively. Keep `false` for batch runs and servers. |
| `selected` | full notebook plot list | Comma-separated plot names to generate. Remove names to make a shorter run. |
| `forecast_diag_countries` | `NGA, CIV, KEN, CMR, RWA, GHA` | Countries used by `forecast_diag`. Requires saved model weights and assets. |
| `selected_country_codes` | `KEN, RWA, GHA, ZAF, NGA, COD` | Countries highlighted by selected spaghetti plots. |
| `nigeria_peers` | `RWA, UGA, KEN, GHA` | Peer countries used by Nigeria comparison plots. |
| `future_horizon` | `10` | Number of future years used by `future_projection`. |
| `figures_zip` | `figures.zip` | Name or path for the zip archive created when `zip_figures = true`. |

Available values for `[plots] selected`:

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

Plots that require unavailable optional dependencies or missing model files are skipped with a warning so the rest of the plot batch can continue.

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

The `[plots] selected` line in `safira.in` controls which figures are produced. The full list of available plot names is documented in the `[plots]` part of the complete input reference above.

For a short plotting run, use only a few names:

```ini
[plots]
selected = spaghetti, box, corr_lower
```

For the full notebook-style plotting run, keep the default list:

```ini
[workflow]
make_plots = true

[plots]
selected = choropleth, spaghetti, pillar_heatmap, box, gdp_scatter, gdp_scatter_2020, gdp_scatter_highlight, gdp_scatter_top10, corr, corr_lower, slope, radar, umap, forecast_diag, nigeria_comparison, nigeria_comparison_seaborn, forecast_diag_zaf, spaghetti_selected, spaghetti_selected_alt, future_projection
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

Install the development extras before running packaging checks:

```bash
pip install ".[dev]"
```

Build a source distribution and wheel from the repository root:

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

Install locally with all optional plotting dependencies:

```bash
pip install ".[full]"
```

## Notes on Reproducibility

- The packaged snapshot is fixed at build time so offline users can reproduce the bundled run. Use `data_mode = download` or `safira fetch-data` to refresh from the live World Bank API.
- Live World Bank results can change if the World Bank revises historical series, changes indicator coverage, or adds newer years.
- The notebook's original full workflow trains an LSTM, so exact model weights may vary unless the runtime, dependencies, random seeds, and hardware behavior are fixed.
- The sample panel is synthetic and exists only for testing package mechanics. It is not a substitute for the World Bank data used in the actual research workflow.

## License

SAFIRA-SSA is distributed under the GNU General Public License version 3.
