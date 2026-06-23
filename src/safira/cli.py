"""Command line interface for SAFIRA-SSA."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    cfg_bool,
    cfg_float,
    cfg_get,
    cfg_int,
    cfg_list,
    read_config,
    write_default_input,
)
from .constants import DEFAULT_INPUT_FILE
from .data import collect_world_bank_panel, prepare_panel, write_sample_panel


def _build_forecaster(config, data_path: str, force_pretrained: bool = False):
    from .forecast import SkillAdvancementForecaster

    return SkillAdvancementForecaster(
        data_path=data_path,
        sheet_name=cfg_get(config, "paths", "sheet_name", "All_Indicators"),
        lookback=cfg_int(config, "model", "lookback", 5),
        hidden_dim=cfg_int(config, "model", "hidden_dim", 32),
        num_layers=cfg_int(config, "model", "num_layers", 4),
        dropout=cfg_float(config, "model", "dropout", 0.25),
        bidirectional=cfg_bool(config, "model", "bidirectional", False),
        epochs=cfg_int(config, "model", "epochs", 1000),
        early_stopping=cfg_bool(config, "model", "early_stopping", False),
        patience=cfg_int(config, "model", "patience", 50),
        min_delta=cfg_float(config, "model", "min_delta", 1e-5),
        batch_size=cfg_int(config, "model", "batch_size", 16),
        learning_rate=cfg_float(config, "model", "learning_rate", 1e-3),
        use_pretrained=force_pretrained or cfg_bool(config, "model", "use_pretrained", False),
        plot_losses=cfg_bool(config, "model", "plot_losses", False),
        input_features=cfg_list(config, "model", "input_features", ["SAI_scaled"]),
        model_weights_file=cfg_get(config, "paths", "model_weights"),
        assets_file=cfg_get(config, "paths", "model_assets"),
        legacy_model_file=cfg_get(config, "paths", "legacy_model"),
    )


def _data_mode(config) -> str:
    """Return the configured data source mode."""
    mode = cfg_get(config, "workflow", "data_mode", "").lower()
    if mode:
        return mode
    return "download" if cfg_bool(config, "workflow", "fetch_data", False) else "custom"


def _workflow_data_file(config) -> str:
    """Resolve the data path for packaged, download, or custom workflows."""
    mode = _data_mode(config)
    if mode == "packaged":
        return cfg_get(config, "paths", "packaged_data_file", "packaged") or "packaged"
    if mode in {"download", "custom"}:
        return cfg_get(config, "paths", "data_file")
    raise ValueError("workflow.data_mode must be one of: packaged, download, custom.")


def run_workflow(input_file: str = DEFAULT_INPUT_FILE) -> None:
    """Run the configured SAFIRA-SSA workflow."""
    config = read_config(input_file)
    data_mode = _data_mode(config)

    if cfg_bool(config, "workflow", "write_sample_data", False):
        sample_path = cfg_get(config, "paths", "sample_data_file", "examples/sample_sai_panel.xlsx")
        write_sample_panel(sample_path)
        print(f"[INFO] Wrote sample panel: {sample_path}")

    data_file = _workflow_data_file(config)
    download_file = cfg_get(config, "paths", "data_file")
    sheet_name = cfg_get(config, "paths", "sheet_name", "All_Indicators")

    if data_mode == "download":
        collect_world_bank_panel(
            output_file=download_file,
            countries=cfg_get(config, "data", "countries", "SSA"),
            start_year=cfg_int(config, "data", "start_year", 2000),
            end_year=cfg_int(config, "data", "end_year", 2026),
            sheet_name=sheet_name,
            validate_codes=cfg_bool(config, "data", "validate_country_codes", True),
            connectivity_check=cfg_bool(config, "data", "connectivity_check", False),
        )
        data_file = download_file
    elif data_mode == "packaged":
        print("[INFO] Using packaged World Bank snapshot.")

    train_model = cfg_bool(config, "workflow", "train_model", True)
    forecast_requested = cfg_bool(config, "workflow", "forecast", True)
    make_plots = cfg_bool(config, "workflow", "make_plots", True)

    prepared_df = None
    forecaster = None
    if train_model or forecast_requested:
        forecaster = _build_forecaster(config, data_path=data_file, force_pretrained=not train_model)
        if train_model:
            prepared_df = forecaster.fit_or_load()
        elif forecast_requested:
            forecaster.load_model()

    if forecast_requested:
        if prepared_df is None:
            prepared_df = prepare_panel(data_file, sheet_name=sheet_name)
        country = cfg_get(config, "forecast", "country", "Nigeria")
        year = cfg_int(config, "forecast", "year", 2025)
        result = forecaster.forecast(country=country, year=year, df=prepared_df)
        label = "FORECAST" if result.is_forecast else "OBSERVED"
        print(f"[{label}] {result.country_code} {result.year}: SAI={result.value:.2f}")
        if result.scaled_value is not None:
            print(f"[{label}] scaled={result.scaled_value:.4f}")

    if make_plots:
        from .plotting import SAIPlotter, run_named_plots

        if prepared_df is None:
            prepared_df = prepare_panel(data_file, sheet_name=sheet_name)
        plotter = SAIPlotter(
            prepared_df,
            figures_dir=cfg_get(config, "paths", "figures_dir", "figures"),
            show=cfg_bool(config, "plots", "show", False),
            world_file=cfg_get(config, "paths", "world_file", "") or None,
        )
        outputs = run_named_plots(
            plotter,
            names=cfg_list(config, "plots", "selected", []),
            weights_file=cfg_get(config, "paths", "model_weights"),
            assets_file=cfg_get(config, "paths", "model_assets"),
            forecast_diag_countries=cfg_list(config, "plots", "forecast_diag_countries", []),
            selected_country_codes=cfg_list(config, "plots", "selected_country_codes", []),
            nigeria_peers=cfg_list(config, "plots", "nigeria_peers", []),
            future_horizon=cfg_int(config, "plots", "future_horizon", 10),
        )
        print(f"[INFO] Generated {len(outputs)} figure output(s).")
        if cfg_bool(config, "workflow", "zip_figures", False):
            plotter.zip_figures(cfg_get(config, "plots", "figures_zip", "figures.zip"))


def fetch_data(input_file: str = DEFAULT_INPUT_FILE) -> None:
    """Run only the World Bank data download step."""
    config = read_config(input_file)
    collect_world_bank_panel(
        output_file=cfg_get(config, "paths", "data_file"),
        countries=cfg_get(config, "data", "countries", "SSA"),
        start_year=cfg_int(config, "data", "start_year", 2000),
        end_year=cfg_int(config, "data", "end_year", 2026),
        sheet_name=cfg_get(config, "paths", "sheet_name", "All_Indicators"),
        validate_codes=cfg_bool(config, "data", "validate_country_codes", True),
        connectivity_check=cfg_bool(config, "data", "connectivity_check", False),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="safira",
        description="SAFIRA-SSA: Skill Advancement Forecasting and Intelligence for Readiness in Africa.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the workflow described by a .in file.")
    run_parser.add_argument("--input", default=DEFAULT_INPUT_FILE, help="Path to the SAFIRA .in file.")

    fetch_parser = subparsers.add_parser("fetch-data", help="Download World Bank indicators only.")
    fetch_parser.add_argument("--input", default=DEFAULT_INPUT_FILE, help="Path to the SAFIRA .in file.")

    sample_parser = subparsers.add_parser("write-sample-data", help="Write the bundled sample panel.")
    sample_parser.add_argument("--output", default="examples/sample_sai_panel.xlsx", help="Output Excel or CSV path.")

    input_parser = subparsers.add_parser("write-input", help="Write a default safira.in file.")
    input_parser.add_argument("--output", default=DEFAULT_INPUT_FILE, help="Output .in path.")
    input_parser.add_argument("--overwrite", action="store_true", help="Overwrite the file if it exists.")

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {None, "run"}:
        run_workflow(args.input if args.command == "run" else DEFAULT_INPUT_FILE)
    elif args.command == "fetch-data":
        fetch_data(args.input)
    elif args.command == "write-sample-data":
        output = write_sample_panel(args.output)
        print(f"[INFO] Wrote sample panel: {output}")
    elif args.command == "write-input":
        output = write_default_input(Path(args.output), overwrite=args.overwrite)
        print(f"[INFO] Wrote input file: {output}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
