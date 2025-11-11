from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from vibracnc.anomaly.autoencoder import (
    AutoencoderConfig,
    LSTMAutoencoder,
    reconstruction_error,
    train_autoencoder,
    build_dataloader,
)
from vibracnc.data.preprocessing import WindowingConfig, dataframe_to_windows, normalize_amplitudes


@dataclass(slots=True)
class AnomalyDetectionArtifacts:
    frequencies: np.ndarray
    train_history: dict[str, list[float]]
    threshold: float
    errors: np.ndarray
    config: AutoencoderConfig


def create_fft_features(
    frames: Sequence[pd.DataFrame],
    sensor_columns: Sequence[str],
    window_config: WindowingConfig,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    freq_list: list[np.ndarray] = []
    amp_list: list[np.ndarray] = []

    for frame in frames:
        freq, amp = dataframe_to_windows(frame, sensor_columns, window_config)
        freq_list.append(freq)
        amp_list.append(amp)

    frequencies = freq_list[0]
    stacks = np.concatenate(amp_list, axis=0)
    if normalize:
        stacks = normalize_amplitudes(stacks, axis=0)
    return frequencies, stacks


def train_normal_model(
    normal_frames: Sequence[pd.DataFrame],
    sensor_columns: Sequence[str],
    window_config: WindowingConfig,
    autoencoder_config: AutoencoderConfig,
    val_ratio: float = 0.1,
    random_state: int = 42,
) -> tuple[LSTMAutoencoder, AnomalyDetectionArtifacts]:
    frequencies, features = create_fft_features(normal_frames, sensor_columns, window_config)
    seq_len = features.shape[1]
    feature_dim = features.shape[2]
    if autoencoder_config.seq_len != seq_len or autoencoder_config.input_dim != feature_dim:
        autoencoder_config = AutoencoderConfig(
            input_dim=feature_dim,
            seq_len=seq_len,
            latent_dim=autoencoder_config.latent_dim,
            hidden_dim=autoencoder_config.hidden_dim,
            num_layers=autoencoder_config.num_layers,
            dropout=autoencoder_config.dropout,
            lr=autoencoder_config.lr,
            weight_decay=autoencoder_config.weight_decay,
            epochs=autoencoder_config.epochs,
            batch_size=autoencoder_config.batch_size,
            device=autoencoder_config.device,
        )

    train_idx, val_idx = train_test_split(
        np.arange(features.shape[0]), test_size=val_ratio, random_state=random_state, shuffle=True
    )
    train_loader = build_dataloader(features[train_idx], autoencoder_config, shuffle=True)
    val_loader = build_dataloader(features[val_idx], autoencoder_config, shuffle=False)

    model = LSTMAutoencoder(autoencoder_config)
    history = train_autoencoder(model, train_loader, autoencoder_config, val_loader)

    train_errors = reconstruction_error(model, features[train_idx], autoencoder_config)
    threshold = np.percentile(train_errors, 95)

    artifacts = AnomalyDetectionArtifacts(
        frequencies=frequencies,
        train_history=history,
        threshold=threshold,
        errors=train_errors,
        config=autoencoder_config,
    )
    return model, artifacts


def score_anomalies(
    model: LSTMAutoencoder,
    frames: Sequence[pd.DataFrame],
    sensor_columns: Sequence[str],
    window_config: WindowingConfig,
    config: AutoencoderConfig,
) -> np.ndarray:
    _, features = create_fft_features(frames, sensor_columns, window_config)
    return reconstruction_error(model, features, config)


def classify_anomalies(errors: np.ndarray, threshold: float) -> np.ndarray:
    return (errors > threshold).astype(int)


def score_with_artifacts(
    model: LSTMAutoencoder,
    frames: Sequence[pd.DataFrame],
    sensor_columns: Sequence[str],
    window_config: WindowingConfig,
    artifacts: AnomalyDetectionArtifacts,
) -> np.ndarray:
    return score_anomalies(model, frames, sensor_columns, window_config, artifacts.config)

