"""PyTorch models used by SAFIRA-SSA forecasting."""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """Sliding-window dataset for one-step sequence forecasting."""

    def __init__(self, sequences: List[np.ndarray], lookback: int, target_idx: int = 0):
        self.lookback = int(lookback)
        self.target_idx = int(target_idx)
        X, y = [], []

        for seq in sequences:
            seq = np.asarray(seq, dtype=np.float32)
            if seq.ndim == 1:
                seq = seq.reshape(-1, 1)

            if seq.shape[0] < self.lookback + 1:
                continue

            for i in range(seq.shape[0] - self.lookback):
                X.append(seq[i : i + self.lookback, :])
                y.append(seq[i + self.lookback, self.target_idx])

        if not X:
            raise ValueError(
                "No usable sequences were found. Ensure each series length "
                "is at least lookback + 1."
            )

        self.X = torch.tensor(np.stack(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"past": self.X[idx], "future": self.y[idx]}


class LSTMForecastNet(nn.Module):
    """Simple univariate LSTM forecaster retained for notebook compatibility."""

    def __init__(self, lookback: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class AdvancedLSTMForecastNet(nn.Module):
    """Configurable LSTM regressor for the SAI sequence."""

    def __init__(
        self,
        input_size: int = 1,
        lookback: int = 3,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lookback = int(lookback)
        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        self.fc = nn.Linear(hidden_dim * self.num_directions, 1)
        self.post_dropout = nn.Dropout(dropout) if dropout > 0 else None
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.lstm(x)
        if self.bidirectional:
            hidden = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            hidden = hn[-1]

        if self.post_dropout is not None:
            hidden = self.post_dropout(hidden)
        return self.fc(hidden)


class EarlyStopping:
    """Track validation loss and signal when training should stop."""

    def __init__(self, patience: int = 50, min_delta: float = 1e-5):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_val_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> None:
        if val_loss < (self.best_val_loss - self.min_delta):
            self.best_val_loss = float(val_loss)
            self.counter = 0
            return

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
