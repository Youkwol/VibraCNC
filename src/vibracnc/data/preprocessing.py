from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq


def ensure_sorted_by_timestamp(frame: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """타임스탬프 기준으로 정렬된 DataFrame을 반환합니다."""
    if timestamp_col not in frame.columns:
        raise ValueError(f"{timestamp_col} 컬럼이 존재하지 않습니다.")
    if frame[timestamp_col].is_monotonic_increasing:
        return frame
    return frame.sort_values(timestamp_col).reset_index(drop=True)


@dataclass(slots=True)
class WindowingConfig:
    window_size: int
    stride: int
    sampling_rate: float

    @classmethod
    def from_seconds(cls, window_seconds: float, step_seconds: float, sampling_rate: float) -> "WindowingConfig":
        window_size = int(window_seconds * sampling_rate)
        stride = int(step_seconds * sampling_rate)
        return cls(window_size=window_size, stride=stride, sampling_rate=sampling_rate)


def sliding_windows(array: np.ndarray, window_size: int, stride: int) -> Iterator[np.ndarray]:
    """1D 또는 2D 배열에서 슬라이딩 윈도우를 생성합니다."""
    if array.ndim == 1:
        array = array[:, None]
    total = array.shape[0]
    if total < window_size:
        raise ValueError("윈도우 크기가 데이터 길이보다 큽니다.")
    for start in range(0, total - window_size + 1, stride):
        yield array[start : start + window_size]


def batch_fft(windows: Iterable[np.ndarray], sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """
    여러 윈도우에 대해 FFT 스펙트럼을 계산합니다.

    Returns
    -------
    frequencies : np.ndarray
        양의 주파수 영역의 주파수 벡터.
    amplitudes : np.ndarray
        각 윈도우의 진폭 스펙트럼. shape = (num_windows, num_freq_bins, num_sensors)
    """
    windows_list = list(windows)
    if not windows_list:
        raise ValueError("FFT를 수행할 윈도우가 없습니다.")

    window_size = windows_list[0].shape[0]
    frequencies = rfftfreq(window_size, d=1 / sampling_rate)
    amplitudes = np.stack([np.abs(rfft(window, axis=0)) for window in windows_list])
    return frequencies, amplitudes


def normalize_amplitudes(amplitudes: np.ndarray, axis: int = -1) -> np.ndarray:
    """Min-Max 정규화."""
    min_val = amplitudes.min(axis=axis, keepdims=True)
    max_val = amplitudes.max(axis=axis, keepdims=True)
    denom = np.where((max_val - min_val) == 0, 1, max_val - min_val)
    return (amplitudes - min_val) / denom


def dataframe_to_windows(
    frame: pd.DataFrame,
    sensor_columns: Sequence[str],
    config: WindowingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    DataFrame을 FFT 학습용 윈도우로 변환합니다.

    Returns
    -------
    frequencies : np.ndarray
    amplitudes : np.ndarray
    """
    sorted_frame = ensure_sorted_by_timestamp(frame)
    signal = sorted_frame[list(sensor_columns)].to_numpy(dtype=float)
    windows = list(sliding_windows(signal, config.window_size, config.stride))
    return batch_fft(windows, config.sampling_rate)

