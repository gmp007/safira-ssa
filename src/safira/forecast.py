"""Training and inference for the SAFIRA-SSA SAI forecaster."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from .constants import (
    COUNTRY_ALIASES,
    DEFAULT_LEGACY_MODEL,
    DEFAULT_MODEL_ASSETS,
    DEFAULT_MODEL_WEIGHTS,
)
from .data import prepare_panel
from .models import AdvancedLSTMForecastNet, EarlyStopping, TimeSeriesDataset
from .sai import get_standard_country_code


@dataclass
class ForecastResult:
    """Forecast or historical SAI lookup result."""

    country_code: str
    year: int
    value: float
    scaled_value: float | None
    is_forecast: bool
    max_observed_year: int


def _load_trained_model_bundle(model_file: str):
    """Load the legacy combined checkpoint produced by older notebook versions."""
    try:
        payload = torch.load(model_file, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(model_file, map_location="cpu")

    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(
            "Checkpoint must contain 'model_state_dict'. Retrain with the packaged "
            "SAFIRA-SSA workflow if this is a raw state_dict."
        )

    input_features = payload.get("input_features", ["SAI_scaled"])
    lookback = int(payload.get("lookback", 5))
    hparams = payload.get("hparams", {})
    model = AdvancedLSTMForecastNet(
        input_size=len(input_features),
        lookback=lookback,
        hidden_dim=int(hparams.get("hidden_dim", 32)),
        num_layers=int(hparams.get("num_layers", 1)),
        dropout=float(hparams.get("dropout", 0.0)),
        bidirectional=bool(hparams.get("bidirectional", False)),
    ).eval()
    model.load_state_dict(payload["model_state_dict"])

    return (
        model,
        payload.get("target_scaler", None),
        payload.get("exog_scalers", {}),
        input_features,
        lookback,
    )


def build_feature_matrix_for_country(
    cdf: pd.DataFrame,
    input_features: list[str],
    target_scaler,
    exog_scalers: dict,
) -> np.ndarray:
    """Build the scaled feature matrix for one country."""
    if target_scaler is None:
        raise ValueError("No target scaler found. Train or load model assets first.")

    y_scaled = target_scaler.transform(cdf[["SAI"]].values.astype(float)).flatten()
    columns = []
    for feature in input_features:
        if feature == "SAI_scaled":
            columns.append(y_scaled.reshape(-1, 1))
        elif feature not in exog_scalers:
            warnings.warn(f"Exogenous feature '{feature}' missing in assets; filling zeros.")
            columns.append(np.zeros((len(cdf), 1), dtype=np.float32))
        else:
            scaler, median = exog_scalers[feature]
            if feature in cdf.columns:
                raw = (
                    pd.to_numeric(cdf[feature], errors="coerce")
                    .fillna(median)
                    .values.reshape(-1, 1)
                    .astype(float)
                )
            else:
                raw = np.full((len(cdf), 1), median, dtype=float)
            columns.append(scaler.transform(raw).astype(np.float32))

    return np.concatenate(columns, axis=1).astype(np.float32)


def build_feature_matrix_for_ssa_median(
    df_all: pd.DataFrame,
    years: np.ndarray,
    input_features: list[str],
    target_scaler,
    exog_scalers: dict,
) -> np.ndarray:
    """Build the scaled feature matrix for SSA median SAI by year."""
    if target_scaler is None:
        raise ValueError("No target scaler found. Train or load model assets first.")

    y_by_year = df_all.groupby("year")["SAI"].median().reindex(years)
    y_scaled = target_scaler.transform(y_by_year.values.reshape(-1, 1)).flatten()

    columns = []
    for feature in input_features:
        if feature == "SAI_scaled":
            columns.append(y_scaled.reshape(-1, 1))
        elif feature in df_all.columns and feature in exog_scalers:
            scaler, median = exog_scalers[feature]
            series = df_all.groupby("year")[feature].median().reindex(years)
            raw = (
                pd.to_numeric(series, errors="coerce")
                .fillna(median)
                .values.reshape(-1, 1)
                .astype(float)
            )
            columns.append(scaler.transform(raw).astype(np.float32))
        elif feature in exog_scalers:
            scaler, median = exog_scalers[feature]
            raw = np.full((len(years), 1), median, dtype=float)
            columns.append(scaler.transform(raw).astype(np.float32))
        else:
            warnings.warn(f"Exogenous feature '{feature}' not available; filling zeros.")
            columns.append(np.zeros((len(years), 1), dtype=np.float32))

    return np.concatenate(columns, axis=1).astype(np.float32)


def load_plot_model(
    weights_file: str = DEFAULT_MODEL_WEIGHTS,
    assets_file: str = DEFAULT_MODEL_ASSETS,
    device: str = "cpu",
):
    """Load model weights and non-tensor assets for plotting workflows."""
    assets = joblib.load(assets_file)
    input_features = assets["input_features"]
    lookback = int(assets["lookback"])
    hparams = assets.get("hparams", {})

    model = AdvancedLSTMForecastNet(
        input_size=len(input_features),
        lookback=lookback,
        hidden_dim=int(hparams.get("hidden_dim", 32)),
        num_layers=int(hparams.get("num_layers", 3)),
        dropout=float(hparams.get("dropout", 0.25)),
        bidirectional=bool(hparams.get("bidirectional", False)),
    ).to(device)
    model.eval()

    try:
        state = torch.load(weights_file, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_file, map_location=device)
    model.load_state_dict(state)
    return model, assets


class SkillAdvancementForecaster:
    """Train, load, and run forecasts for the Skill Advancement Index."""

    def __init__(
        self,
        data_path: str = "data/ssa_sai_all_indicators.xlsx",
        sheet_name: str = "All_Indicators",
        lookback: int = 3,
        hidden_dim: Union[int, List[int]] = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        epochs: int = 200,
        early_stopping: bool = True,
        patience: int = 20,
        min_delta: float = 1e-5,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        use_pretrained: bool = False,
        plot_losses: bool = False,
        input_features: Optional[List[str]] = None,
        model_weights_file: str = DEFAULT_MODEL_WEIGHTS,
        assets_file: str = DEFAULT_MODEL_ASSETS,
        legacy_model_file: str = DEFAULT_LEGACY_MODEL,
    ):
        self.data_path = data_path
        self.sheet_name = sheet_name
        self.lookback = int(lookback)
        self.hidden_dim = hidden_dim if isinstance(hidden_dim, int) else int(hidden_dim[0])
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)
        self.epochs = int(epochs)
        self.early_stopping = bool(early_stopping)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.use_pretrained = bool(use_pretrained)
        self.plot_losses = bool(plot_losses)
        self.input_features = input_features or ["SAI_scaled"]

        self.model_weights_file = model_weights_file
        self.assets_file = assets_file
        self.model_file = legacy_model_file

        self.model = None
        self.train_mse = None
        self.val_mse = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_scaler: Optional[MinMaxScaler] = None
        self.exog_scalers: Dict[str, Tuple[MinMaxScaler, float]] = {}

        try:
            self.target_feature_idx = self.input_features.index("SAI_scaled")
        except ValueError as exc:
            raise ValueError("input_features must contain 'SAI_scaled'.") from exc

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def load_data(self) -> pd.DataFrame:
        """Load the panel and compute the SAI fields."""
        print(f"[INFO] Loading data from '{self.data_path}' (sheet={self.sheet_name}) ...")
        return prepare_panel(self.data_path, sheet_name=self.sheet_name)

    def prepare_data(self, df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, pd.DataFrame]:
        """Create train and validation data loaders without time leakage."""
        for feature in self.input_features:
            if feature != "SAI_scaled" and feature not in df.columns:
                raise ValueError(f"Input feature '{feature}' not found in dataframe.")

        rng = np.random.RandomState(42)
        codes = np.array(list(df["country_code"].unique()), dtype=object)
        rng.shuffle(codes)
        split = max(1, int(0.8 * len(codes)))
        train_codes = codes[:split]

        train_df = df[df["country_code"].isin(train_codes)].copy()
        val_df = df[~df["country_code"].isin(train_codes)].copy()
        if val_df.empty:
            print("[WARN] Validation split is empty; using training split for validation.")
            val_df = train_df.copy()

        self.target_scaler = MinMaxScaler()
        self.target_scaler.fit(train_df[["SAI"]].values.astype(float))

        self.exog_scalers = {}
        for column in [c for c in self.input_features if c != "SAI_scaled"]:
            scaler = MinMaxScaler()
            train_values = pd.to_numeric(train_df[column], errors="coerce")
            median = float(np.nanmedian(train_values.values)) if not np.isnan(train_values.values).all() else 0.0
            scaled_input = train_values.fillna(median).values.reshape(-1, 1).astype(float)
            scaler.fit(scaled_input)
            self.exog_scalers[column] = (scaler, median)

        def make_sequences(sub_df: pd.DataFrame) -> List[np.ndarray]:
            sequences = []
            for code in sub_df["country_code"].unique():
                cdf = sub_df[sub_df["country_code"] == code].sort_values("year")
                y_scaled = self.target_scaler.transform(cdf[["SAI"]].values.astype(float)).flatten()
                feature_columns = []
                for feature in self.input_features:
                    if feature == "SAI_scaled":
                        feature_columns.append(y_scaled.reshape(-1, 1))
                    else:
                        scaler, median = self.exog_scalers[feature]
                        raw = (
                            pd.to_numeric(cdf[feature], errors="coerce")
                            .fillna(median)
                            .values.reshape(-1, 1)
                            .astype(float)
                        )
                        feature_columns.append(scaler.transform(raw))

                seq = np.concatenate(feature_columns, axis=1).astype(np.float32)
                if seq.shape[0] >= self.lookback + 1:
                    sequences.append(seq)
            return sequences

        train_sequences = make_sequences(train_df)
        val_sequences = make_sequences(val_df)
        if not train_sequences:
            raise RuntimeError("No training sequences were built. Check data coverage and lookback.")
        if not val_sequences:
            print("[WARN] No validation sequences were built; reusing training sequences.")
            val_sequences = train_sequences

        train_ds = TimeSeriesDataset(
            train_sequences, lookback=self.lookback, target_idx=self.target_feature_idx
        )
        val_ds = TimeSeriesDataset(
            val_sequences, lookback=self.lookback, target_idx=self.target_feature_idx
        )

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, df

    def define_model(self) -> None:
        """Instantiate the LSTM model for the configured features."""
        print("[INFO] Defining an AdvancedLSTMForecastNet model ...")
        self.model = AdvancedLSTMForecastNet(
            input_size=len(self.input_features),
            lookback=self.lookback,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        ).to(self.device)

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Train the model and keep the best validation state."""
        print(f"[INFO] Training for up to {self.epochs} epochs ...")
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        stopper = EarlyStopping(self.patience, self.min_delta) if self.early_stopping else None
        best_val_loss = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []
            for batch in train_loader:
                past = batch["past"].to(self.device)
                future = batch["future"].to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(past), future)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            train_mse = float(np.mean(train_losses))
            self.train_losses.append(train_mse)

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    past = batch["past"].to(self.device)
                    future = batch["future"].to(self.device)
                    val_losses.append(criterion(self.model(past), future).item())
            val_mse = float(np.mean(val_losses)) if val_losses else float("inf")
            self.val_losses.append(val_mse)
            scheduler.step()

            if (epoch + 1) % 20 == 0 or epoch == 0 or (epoch + 1) == self.epochs:
                print(f"Epoch {epoch + 1}/{self.epochs}: train_MSE={train_mse:.6f}, val_MSE={val_mse:.6f}")

            if val_mse < best_val_loss - 1e-12:
                best_val_loss = val_mse
                best_state = copy.deepcopy(self.model.state_dict())
            if stopper:
                stopper.step(val_mse)
                if stopper.should_stop:
                    print("[EARLY STOPPING] Triggered.")
                    break

        self.model.load_state_dict(best_state)
        self.train_mse = train_mse
        self.val_mse = best_val_loss
        print(f"\n[INFO] Best Val MSE: {self.val_mse:.6f}")

    def save_model(self) -> None:
        """Save tensor weights and non-tensor assets separately."""
        Path(self.model_weights_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.assets_file).parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), self.model_weights_file)
        assets = {
            "input_features": self.input_features,
            "lookback": self.lookback,
            "hparams": {
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "bidirectional": self.bidirectional,
            },
            "target_scaler": self.target_scaler,
            "exog_scalers": self.exog_scalers,
        }
        joblib.dump(assets, self.assets_file)
        print(f"[INFO] Saved weights -> {self.model_weights_file}")
        print(f"[INFO] Saved assets  -> {self.assets_file}")

    def _migrate_legacy_checkpoint(self, old_file: str, new_weights: str, new_assets: str) -> None:
        """Convert a legacy combined checkpoint into weights plus assets files."""
        try:
            payload = torch.load(old_file, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(old_file, map_location=self.device)

        if isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
            assets = {
                "input_features": payload.get("input_features", self.input_features),
                "lookback": payload.get("lookback", self.lookback),
                "hparams": payload.get(
                    "hparams",
                    {
                        "hidden_dim": self.hidden_dim,
                        "num_layers": self.num_layers,
                        "dropout": self.dropout,
                        "bidirectional": self.bidirectional,
                    },
                ),
                "target_scaler": payload.get("target_scaler", None),
                "exog_scalers": payload.get("exog_scalers", {}),
            }
        else:
            state_dict = payload
            assets = {
                "input_features": self.input_features,
                "lookback": self.lookback,
                "hparams": {
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "bidirectional": self.bidirectional,
                },
                "target_scaler": self.target_scaler,
                "exog_scalers": self.exog_scalers,
            }

        Path(new_weights).parent.mkdir(parents=True, exist_ok=True)
        Path(new_assets).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, new_weights)
        joblib.dump(assets, new_assets)
        print(f"[MIGRATE] Wrote weights -> {new_weights}")
        print(f"[MIGRATE] Wrote assets  -> {new_assets}")

    def load_model(self) -> None:
        """Load architecture, scalers, and model weights."""
        if (
            (not os.path.isfile(self.model_weights_file) or not os.path.isfile(self.assets_file))
            and os.path.isfile(self.model_file)
        ):
            print("[INFO] Found legacy bundle. Migrating to weights+assets format ...")
            self._migrate_legacy_checkpoint(
                self.model_file, self.model_weights_file, self.assets_file
            )

        if not os.path.isfile(self.assets_file) or not os.path.isfile(self.model_weights_file):
            raise FileNotFoundError(
                f"Missing {self.model_weights_file} and/or {self.assets_file}. "
                "Train first or provide both files."
            )

        assets = joblib.load(self.assets_file)
        self.input_features = assets.get("input_features", self.input_features)
        self.lookback = int(assets.get("lookback", self.lookback))
        hparams = assets.get("hparams", {})
        self.hidden_dim = int(hparams.get("hidden_dim", self.hidden_dim))
        self.num_layers = int(hparams.get("num_layers", self.num_layers))
        self.dropout = float(hparams.get("dropout", self.dropout))
        self.bidirectional = bool(hparams.get("bidirectional", self.bidirectional))
        self.target_scaler = assets.get("target_scaler", None)
        self.exog_scalers = assets.get("exog_scalers", {})
        self.target_feature_idx = self.input_features.index("SAI_scaled")

        self.define_model()
        try:
            state = torch.load(self.model_weights_file, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(self.model_weights_file, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"[INFO] Loaded model from weights={self.model_weights_file}, assets={self.assets_file}")

    def fit_or_load(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Load pretrained assets when requested, otherwise train and save."""
        if df is None:
            df = self.load_data()

        if self.use_pretrained and (
            os.path.isfile(self.model_weights_file) or os.path.isfile(self.model_file)
        ):
            self.load_model()
            self.train_mse = float("nan")
            self.val_mse = float("nan")
            return df

        train_loader, val_loader, df = self.prepare_data(df)
        self.define_model()
        self.train_model(train_loader, val_loader)
        self.save_model()
        return df

    def forecast(self, country: str, year: int, df: pd.DataFrame | None = None) -> ForecastResult:
        """Return an observed SAI value or forecast for a requested country-year."""
        if self.model is None or self.target_scaler is None:
            df = self.fit_or_load(df)
        elif df is None:
            df = self.load_data()

        code = get_standard_country_code(country, COUNTRY_ALIASES)
        if code == "XXX":
            raise ValueError(f"No known match for country '{country}'.")

        year_req = int(year)
        cdata = df[df["country_code"] == code].sort_values("year")
        if len(cdata) < self.lookback + 1:
            raise ValueError(f"Not enough data for country '{code}' to forecast.")

        max_year = int(cdata["year"].max())
        self.model.eval()

        if year_req <= max_year:
            row = cdata[cdata["year"] == year_req]
            if row.empty:
                raise ValueError(
                    f"Year {year_req} is before/equal to max year {max_year}, but no exact row exists."
                )
            actual_sai = float(row["SAI"].values[0])
            scaled_sai = float(self.target_scaler.transform([[actual_sai]])[0, 0])
            return ForecastResult(code, year_req, actual_sai, scaled_sai, False, max_year)

        steps_ahead = year_req - max_year
        cur_seq = build_feature_matrix_for_country(
            cdata, self.input_features, self.target_scaler, self.exog_scalers
        )[-self.lookback :, :]
        cur_seq = torch.tensor(cur_seq, dtype=torch.float32, device=self.device).unsqueeze(0)

        exog_last_scaled = {}
        for feature in self.input_features:
            if feature == "SAI_scaled":
                continue
            scaler, median = self.exog_scalers[feature]
            last_raw = cdata[feature].iloc[-1]
            if pd.isna(last_raw):
                last_raw = median
            exog_last_scaled[feature] = float(
                scaler.transform(np.array([[last_raw]], dtype=float))[0, 0]
            )

        forecast_scaled = None
        with torch.no_grad():
            for _ in range(steps_ahead):
                next_sai_scaled = float(self.model(cur_seq).item())
                next_sai_scaled = float(np.clip(next_sai_scaled, 0.0, 1.0))
                next_vector = [
                    next_sai_scaled if feature == "SAI_scaled" else exog_last_scaled[feature]
                    for feature in self.input_features
                ]
                next_tensor = torch.tensor(
                    next_vector, dtype=torch.float32, device=self.device
                ).view(1, 1, -1)
                cur_seq = torch.cat([cur_seq[:, 1:, :], next_tensor], dim=1)
                forecast_scaled = next_sai_scaled

        forecast_value = float(self.target_scaler.inverse_transform([[forecast_scaled]])[0, 0])
        return ForecastResult(code, year_req, forecast_value, forecast_scaled, True, max_year)

    def run(self, country: str = "Nigeria", year: int = 2025) -> ForecastResult:
        """Run the complete train/load plus forecast workflow and print the result."""
        result = self.forecast(country=country, year=year)
        label = "FORECAST" if result.is_forecast else "OBSERVED"
        print(f"[{label}] For country_code={result.country_code} in year {result.year}:")
        print(f"          SAI (0..100) = {result.value:.2f}")
        if result.scaled_value is not None:
            print(f"          SAI scaled    = {result.scaled_value:.4f}")
        if self.train_mse is not None and self.val_mse is not None and not self.use_pretrained:
            print(f"Final Train MSE: {self.train_mse:.4f}, Best Val MSE: {self.val_mse:.4f}")
        return result


psai = SkillAdvancementForecaster
