# SAFIRA-SSA

**Skill Advancement Forecasting and Intelligence for Readiness in Africa, Sub-Saharan Africa**

SAFIRA-SSA is a policy-facing research package for measuring, forecasting, and comparing skills readiness in Sub-Saharan Africa in the age of artificial intelligence, automation, and workforce transformation. It provides a reproducible decision-support workflow for constructing the Skill Advancement Index (SAI), refreshing or reusing World Bank data, forecasting country trajectories, and generating diagnostics that can inform education, labor-market, digital-readiness, and workforce-development strategy.

At its core, SAFIRA-SSA operationalizes the **Skill Advancement Index (SAI)**: a multidimensional index designed to capture the structural conditions that shape national workforce readiness. The framework combines internationally comparable World Bank indicators across foundational education, advanced and technical skills, digital readiness, and labor-market alignment. It then uses time-series forecasting to project country-level skill-readiness trajectories and generates diagnostic figures for comparing countries, pillars, regional patterns, and future trajectories.

The package is designed for researchers, policymakers, development practitioners, education planners, labor-market analysts, and institutions interested in evidence-based skills strategy. It is not intended to replace local knowledge or policy judgment. Instead, it provides a transparent analytical framework for asking sharper questions about where AI-era skills gaps are emerging, which readiness dimensions appear to constrain progress, and where targeted investments in education, digital infrastructure, and technical training may be most urgent.

The package is organized for practical reuse: it can be installed with `pip install .`, controlled through a plain-text `safira.in` file, run from the command line, tested, used offline with packaged data, and refreshed online from the World Bank API when updated evidence is needed.

## Authors

- **Chinedu Ekuma**
- **Kelechi Ekuma**

## Why This Package Matters

AI and automation are changing the skill profile required for productive participation in the global economy. For Sub-Saharan Africa, this transition is especially consequential. The region combines a large and growing youth population with uneven educational quality, infrastructure gaps, expanding digital connectivity, high unemployment in many countries, and heterogeneous readiness for technology-intensive labor markets.

SAFIRA-SSA turns this policy challenge into a reproducible analytical workflow. It helps users move from broad statements about "future skills" to country-year evidence, pillar-level diagnosis, and forward-looking trajectories. The package can help answer questions such as:

- Which countries show improving, stagnant, or declining skills-readiness trajectories?
- Are readiness gaps driven primarily by foundational education, advanced technical capacity, digital infrastructure, or labor-market alignment?
- Which countries are peers or outliers when SAI trajectories are compared over time?
- How does skills readiness relate to macro-development indicators such as GDP growth?
- Where might progress plateau without stronger investment in digital readiness, education reform, or technical training?
- How can policymakers use common indicators to compare readiness while still adapting interventions to national contexts?

Applied analyses with SAFIRA-SSA can reveal substantial divergence in skills readiness across SSA countries and help identify whether digital readiness, advanced technical skills, foundational education, or labor-market alignment are the strongest constraints on overall skills advancement.

## What SAFIRA-SSA Does

SAFIRA-SSA supports a complete research-to-policy analysis workflow:

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
7. Generates major comparison and diagnostic plots, including trajectories, pillar heatmaps, regional boxes, GDP-SAI scatter plots, correlation plots, slope plots, radar plots, UMAP plots, forecast diagnostics, Nigeria peer comparisons, and SSA median future projections.

## Policy and Development Uses

SAFIRA-SSA is intended to support anticipatory skills planning rather than retrospective reporting alone. The package can be used to produce evidence for:

| Use case | How SAFIRA-SSA helps |
| --- | --- |
| National skills strategy | Tracks SAI and pillar trends to identify whether bottlenecks are educational, technical, digital, or labor-market related. |
| AI-readiness planning | Highlights digital and advanced-skills deficits that may limit participation in AI-enabled labor markets. |
| Education and training reform | Connects foundational and advanced-skills indicators to forecasted readiness trajectories. |
| Regional benchmarking | Compares countries and peer groups using common indicators and reproducible visual diagnostics. |
| Investment prioritization | Helps identify where infrastructure, tertiary training, research capacity, or labor-market alignment may require attention. |
| Scenario preparation | Uses forecast trajectories to inform discussions about future readiness, plateau risks, and policy urgency. |

The output should be interpreted as decision support, not as a deterministic policy prescription. Forecasts are only as strong as the underlying indicators, historical coverage, modeling assumptions, and data revisions. Users should combine SAFIRA-SSA outputs with country-specific institutional knowledge, qualitative evidence, and stakeholder engagement.

## Methodological Overview

The package follows a transparent analytical sequence:

1. **Indicator collection**: gather country-year World Bank indicators for Sub-Saharan Africa.
2. **Domain construction**: organize indicators into the SAI domains of foundational education, advanced and technical skills, digital readiness, and labor-market alignment.
3. **Index construction**: normalize and aggregate available indicators into a 0 to 100 SAI scale.
4. **Forecasting**: train or reload a Long Short-Term Memory (LSTM) time-series model over country SAI histories.
5. **Diagnostics and visualization**: generate figures that compare trajectories, pillar profiles, correlations, country clusters, peer comparisons, forecast diagnostics, and future regional projections.

This structure makes the code useful for recurring skills-readiness monitoring, new country panels, updated World Bank data pulls, or modified skills-readiness frameworks.

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

The default `safira.in` is configured for a complete first run: the packaged 2000-2025 World Bank snapshot, univariate `SAI_scaled` LSTM input, Nigeria 2025 forecast, and the full diagnostic plot set.

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
| `legacy_model` | `models/time_series_sai_model.pth` | Legacy single-file model path kept for compatibility with older saved-model artifacts. |
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
| `selected` | full diagnostic plot list | Comma-separated plot names to generate. Remove names to make a shorter run. |
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

Live refreshes use the official World Bank JSON API directly. Standard WDI indicators are read through the country-indicator endpoint, while the governance indicators are read from World Bank source `3`, Worldwide Governance Indicators, using the correct `GOV_WGI_*` source series IDs. The indicator list lives in `src/safira/constants.py` so data definitions remain explicit and auditable.

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

The forecasting class is `safira.forecast.SkillAdvancementForecaster`. The legacy class name `psai` remains available as an alias.

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

For the full diagnostic plotting run, keep the default list:

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
- The full workflow trains an LSTM, so exact model weights may vary unless the runtime, dependencies, random seeds, and hardware behavior are fixed.
- The sample panel is synthetic and exists only for testing package mechanics. It is not a substitute for the World Bank data used in the actual research workflow.

## Citing the Paper and Code

SAFIRA-SSA accompanies the preprint:

```text
Forecasting the Future of Skills in Sub-Saharan Africa:
AI, Automation, and the Skill Advancement Index
```

If you use SAFIRA-SSA, the packaged World Bank snapshot, the SAI construction workflow, the LSTM forecasting workflow, or any part of the repository resources in academic work, policy analysis, reports, teaching, or derivative software, please cite the accompanying paper and acknowledge the code repository.

Suggested preprint citation:

```text
Ekuma, K., and Ekuma, C. Forecasting the Future of Skills in Sub-Saharan Africa:
AI, Automation, and the Skill Advancement Index. Preprint, 2026.
Code: https://github.com/gmp007/safira-ssa
```

BibTeX placeholder:

```bibtex
@misc{ekuma2026safira,
  title  = {Forecasting the Future of Skills in Sub-Saharan Africa: AI, Automation, and the Skill Advancement Index},
  author = {Ekuma, Kelechi and Ekuma, Chinedu},
  year   = {2026},
  note   = {Preprint. Code available at https://github.com/gmp007/safira-ssa}
}
```

The citation details will be updated once the manuscript is formally published.

## License

SAFIRA-SSA is distributed under the GNU General Public License version 3.
