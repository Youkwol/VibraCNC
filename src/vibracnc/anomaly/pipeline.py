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
    threshold: float  # 윈도우 단위 임계값
    frame_threshold: float  # 프레임 단위 평균 오차 임계값
    errors: np.ndarray
    config: AutoencoderConfig
    norm_min: np.ndarray | None = None  # 정규화 파라미터 (학습 데이터 기준)
    norm_max: np.ndarray | None = None  # 정규화 파라미터 (학습 데이터 기준)


def create_fft_features(
    frames: Sequence[pd.DataFrame],
    sensor_columns: Sequence[str],
    window_config: WindowingConfig,
    normalize: bool = True,
    norm_min: np.ndarray | None = None,
    norm_max: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    FFT 특징을 생성합니다.
    
    Returns
    -------
    frequencies : np.ndarray
    stacks : np.ndarray (정규화된 특징)
    norm_min : np.ndarray | None (정규화 파라미터, 학습 데이터 기준)
    norm_max : np.ndarray | None (정규화 파라미터, 학습 데이터 기준)
    """
    freq_list: list[np.ndarray] = []
    amp_list: list[np.ndarray] = []

    for frame in frames:
        freq, amp = dataframe_to_windows(frame, sensor_columns, window_config)
        freq_list.append(freq)
        amp_list.append(amp)

    frequencies = freq_list[0]
    stacks = np.concatenate(amp_list, axis=0)
    
    if normalize:
        if norm_min is not None and norm_max is not None:
            # 테스트 데이터: 학습 데이터의 정규화 파라미터 사용
            denom = np.where((norm_max - norm_min) == 0, 1, norm_max - norm_min)
            stacks = (stacks - norm_min) / denom
            return frequencies, stacks, norm_min, norm_max
        else:
            # 학습 데이터: 정규화 파라미터 계산
            norm_min = stacks.min(axis=0, keepdims=True)
            norm_max = stacks.max(axis=0, keepdims=True)
            stacks = normalize_amplitudes(stacks, axis=0)
            return frequencies, stacks, norm_min, norm_max
    
    return frequencies, stacks, None, None


def train_normal_model(
    normal_frames: Sequence[pd.DataFrame],
    sensor_columns: Sequence[str],
    window_config: WindowingConfig,
    autoencoder_config: AutoencoderConfig,
    val_ratio: float = 0.1,
    random_state: int = 42,
) -> tuple[LSTMAutoencoder, AnomalyDetectionArtifacts]:
    frequencies, features, norm_min, norm_max = create_fft_features(normal_frames, sensor_columns, window_config)
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
    # 윈도우 단위 임계값: 평균 + 4*표준편차 사용 (약 99.99% 정상 데이터 포함)
    # 또는 95% 백분위수 사용 가능: threshold = np.percentile(train_errors, 95)
    mean_error = np.mean(train_errors)
    std_error = np.std(train_errors)
    threshold = mean_error + 4 * std_error
    
    # 프레임 단위 평균 오차 임계값 계산
    # 각 프레임에 대해 평균 오차를 계산
    frame_errors = []
    for frame in normal_frames:
        frame_window_errors = score_anomalies(
            model,
            [frame],
            sensor_columns,
            window_config,
            autoencoder_config,
            norm_min=norm_min,
            norm_max=norm_max,
        )
        frame_errors.append(float(np.mean(frame_window_errors)))
    
    # 프레임 단위 임계값: 평균 + 4*표준편차 사용 (약 99.99% 정상 데이터 포함)
    # 또는 95% 백분위수 사용 가능: frame_threshold = np.percentile(frame_errors, 95)
    if frame_errors:
        mean_frame_error = np.mean(frame_errors)
        std_frame_error = np.std(frame_errors)
        frame_threshold = mean_frame_error + 4 * std_frame_error
    else:
        frame_threshold = threshold

    artifacts = AnomalyDetectionArtifacts(
        frequencies=frequencies,
        train_history=history,
        threshold=threshold,
        frame_threshold=frame_threshold,
        errors=train_errors,
        config=autoencoder_config,
        norm_min=norm_min,
        norm_max=norm_max,
    )
    return model, artifacts


def score_anomalies(
    model: LSTMAutoencoder,
    frames: Sequence[pd.DataFrame],
    sensor_columns: Sequence[str],
    window_config: WindowingConfig,
    config: AutoencoderConfig,
    norm_min: np.ndarray | None = None,
    norm_max: np.ndarray | None = None,
) -> np.ndarray:
    _, features, _, _ = create_fft_features(frames, sensor_columns, window_config, norm_min=norm_min, norm_max=norm_max)
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
    return score_anomalies(
        model, 
        frames, 
        sensor_columns, 
        window_config, 
        artifacts.config,
        norm_min=artifacts.norm_min,
        norm_max=artifacts.norm_max,
    )

