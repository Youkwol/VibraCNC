from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


STAT_FEATURES = (
    "mean",
    "std",
    "min",
    "max",
    "median",
    "rms",
    "skewness",
    "kurtosis",
    "crest_factor",
    "clearance_factor",
)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def crest_factor(values: np.ndarray) -> float:
    return float(np.max(np.abs(values)) / (rms(values) + 1e-12))


def clearance_factor(values: np.ndarray) -> float:
    numerator = np.max(np.abs(values))
    denominator = np.sqrt(np.mean(np.sqrt(np.abs(values)) ** 2)) + 1e-12
    return float(numerator / denominator)


def compute_stat_features(series: pd.Series) -> dict[str, float]:
    values = series.to_numpy(dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "rms": rms(values),
        "skewness": float(skew(values)),
        "kurtosis": float(kurtosis(values)),
        "crest_factor": crest_factor(values),
        "clearance_factor": clearance_factor(values),
    }


@dataclass(slots=True)
class FeatureExtractionConfig:
    sensor_columns: Sequence[str]
    timestamp_column: str = "timestamp"
    group_column: str | None = None


def extract_features(frame: pd.DataFrame, config: FeatureExtractionConfig) -> pd.Series:
    features: dict[str, float] = {}
    for sensor in config.sensor_columns:
        stats = compute_stat_features(frame[sensor])
        for key, value in stats.items():
            features[f"{sensor}_{key}"] = value
    if config.group_column and config.group_column in frame.columns:
        features["group_id"] = frame[config.group_column].iloc[0]
    return pd.Series(features)

