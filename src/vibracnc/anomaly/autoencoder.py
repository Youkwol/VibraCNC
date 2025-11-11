from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(slots=True)
class AutoencoderConfig:
    input_dim: int
    seq_len: int
    latent_dim: int = 32
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 50
    batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LSTMAutoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.to_latent = nn.Linear(config.hidden_dim, config.latent_dim)
        self.from_latent = nn.Linear(config.latent_dim, config.hidden_dim)
        self.decoder = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(config.hidden_dim, config.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded_seq, _ = self.encoder(x)
        latent = self.to_latent(encoded_seq[:, -1, :])
        repeated = latent.unsqueeze(1).repeat(1, x.size(1), 1)
        decoded_input = self.from_latent(repeated)
        decoded_seq, _ = self.decoder(decoded_input)
        return self.output(decoded_seq)


def build_dataloader(features: np.ndarray, config: AutoencoderConfig, shuffle: bool = True) -> DataLoader:
    tensor = torch.tensor(features, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle)


def train_autoencoder(
    model: LSTMAutoencoder,
    train_loader: DataLoader,
    config: AutoencoderConfig,
    val_loader: Optional[DataLoader] = None,
) -> dict[str, list[float]]:
    device = torch.device(config.device)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(train_loader.dataset)
        history["train_loss"].append(epoch_loss)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(device)
                    recon = model(batch)
                    loss = criterion(recon, batch)
                    val_loss += loss.item() * batch.size(0)
            val_loss /= len(val_loader.dataset)
            history["val_loss"].append(val_loss)
        else:
            history["val_loss"].append(float("nan"))

    return history


def reconstruction_error(model: LSTMAutoencoder, features: np.ndarray, config: AutoencoderConfig) -> np.ndarray:
    device = torch.device(config.device)
    model.eval()
    loader = build_dataloader(features, config, shuffle=False)
    errors: list[np.ndarray] = []
    criterion = nn.MSELoss(reduction="none")
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch).mean(dim=(1, 2))
            errors.append(loss.cpu().numpy())
    return np.concatenate(errors)


def save_model(model: LSTMAutoencoder, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: Path, config: AutoencoderConfig) -> LSTMAutoencoder:
    model = LSTMAutoencoder(config)
    model.load_state_dict(torch.load(path, map_location=config.device))
    model.to(config.device)
    model.eval()
    return model

