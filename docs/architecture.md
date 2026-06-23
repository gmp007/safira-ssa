# SAFIRA-SSA Architecture

The original notebook has been split by responsibility:

- `safira.data`: World Bank download, panel reading, sample data writing.
- `safira.sai`: country normalization and SAI construction.
- `safira.models`: PyTorch sequence dataset and LSTM models.
- `safira.forecast`: training, model persistence, reloads, and country-year forecasts.
- `safira.plotting`: figure generation from a prepared panel.
- `safira.config`: `.in` file parsing and default input generation.
- `safira.cli`: command line orchestration.

The default `safira.in` file keeps workflow decisions outside Python code. That is the replacement for editing notebook cells such as “MAIN - call whichever plots you want.”
