from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from vibracnc.anomaly.autoencoder import LSTMAutoencoder
from vibracnc.anomaly.pipeline import AnomalyDetectionArtifacts, score_with_artifacts
from vibracnc.config import DatasetConfig
from vibracnc.data.loader import load_csv
from vibracnc.data.preprocessing import WindowingConfig, dataframe_to_windows


@dataclass(slots=True)
class CutSample:
    condition: str
    cut_id: str
    global_index: int
    condition_index: int
    frame: pd.DataFrame


@dataclass(slots=True)
class CutMonitoringRecord:
    condition: str
    cut_id: str
    global_index: int
    condition_index: int
    anomaly_score: float
    is_anomaly: bool
    vibration_rms: float
    timestamp: str | None

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "cut_id": self.cut_id,
            "global_index": self.global_index,
            "condition_index": self.condition_index,
            "anomaly_score": self.anomaly_score,
            "is_anomaly": self.is_anomaly,
            "vibration_rms": self.vibration_rms,
            "timestamp": self.timestamp,
        }


@dataclass(slots=True)
class MonitoringSummary:
    current_state: dict
    anomaly_series: list[dict]
    vibration_series: list[dict]
    fft_snapshot: dict | None
    records: list[dict]

    def to_dict(self) -> dict:
        return {
            "current_state": self.current_state,
            "series": {
                "anomaly_scores": self.anomaly_series,
                "vibration_rms": self.vibration_series,
            },
            "fft_snapshot": self.fft_snapshot,
            "records": self.records,
        }


def load_recent_cut_samples(
    dataset_root: Path,
    dataset_config: DatasetConfig,
    conditions: Sequence[str],
    per_condition_limit: int | None,
    use_recent: bool = True,
) -> list[CutSample]:
    """
    Cut 샘플을 로드합니다.
    
    Args:
        use_recent: True면 마지막 N개, False면 처음 N개를 사용합니다.
            per_condition_limit이 None이면 모든 데이터를 사용합니다.
    """
    samples: list[CutSample] = []
    global_index = 0
    column_names = dataset_config.resolve_column_names()

    for condition in conditions:
        condition_dir = dataset_root / condition
        if not condition_dir.exists():
            continue

        csv_files = sorted(condition_dir.glob("*.csv"))
        if per_condition_limit is not None:
            if use_recent:
                csv_files = csv_files[-per_condition_limit :]  # 마지막 N개
            else:
                csv_files = csv_files[:per_condition_limit]  # 처음 N개

        for condition_index, csv_path in enumerate(csv_files, start=1):
            frame = load_csv(csv_path, column_names=column_names)
            global_index += 1
            samples.append(
                CutSample(
                    condition=condition,
                    cut_id=csv_path.stem,
                    global_index=global_index,
                    condition_index=condition_index,
                    frame=frame,
                )
            )

    return samples


def build_monitoring_summary(
    samples: Sequence[CutSample],
    model: LSTMAutoencoder,
    artifacts: AnomalyDetectionArtifacts,
    dataset_config: DatasetConfig,
    window_config: WindowingConfig,
) -> MonitoringSummary:
    if not samples:
        raise ValueError("분석할 cut 샘플이 없습니다.")

    records: list[CutMonitoringRecord] = []

    for sample in samples:
        errors = score_with_artifacts(
            model,
            [sample.frame],
            dataset_config.fft_columns,
            window_config,
            artifacts,
        )
        anomaly_score = float(np.mean(errors))
        vibration_rms = float(_compute_vibration_rms(sample.frame))
        timestamp = _extract_timestamp(sample.frame, dataset_config.timestamp_column)
        # 프레임 단위 평균 오차는 frame_threshold와 비교
        frame_threshold = getattr(artifacts, 'frame_threshold', artifacts.threshold)
        record = CutMonitoringRecord(
            condition=sample.condition,
            cut_id=sample.cut_id,
            global_index=sample.global_index,
            condition_index=sample.condition_index,
            anomaly_score=anomaly_score,
            is_anomaly=anomaly_score > frame_threshold,
            vibration_rms=vibration_rms,
            timestamp=timestamp,
        )
        records.append(record)

    latest_record = records[-1]
    danger_flag = latest_record.is_anomaly
    frame_threshold = getattr(artifacts, 'frame_threshold', artifacts.threshold)
    danger_level = _danger_level(latest_record.anomaly_score, frame_threshold)

    fft_snapshot = _compute_fft_snapshot(
        samples[-1].frame,
        dataset_config.fft_columns,
        window_config,
    )

    anomaly_series = [
        {
            "cut": record.global_index,
            "condition": record.condition,
            "cut_id": record.cut_id,
            "score": record.anomaly_score,
            "is_anomaly": record.is_anomaly,
        }
        for record in records
    ]

    vibration_series = [
        {
            "cut": record.global_index,
            "condition": record.condition,
            "cut_id": record.cut_id,
            "vibration_rms": record.vibration_rms,
        }
        for record in records
    ]

    frame_threshold = getattr(artifacts, 'frame_threshold', artifacts.threshold)
    current_state = {
        "current_cut": latest_record.global_index,
        "current_condition": latest_record.condition,
        "current_cut_id": latest_record.cut_id,
        "current_anomaly_score": latest_record.anomaly_score,
        "danger_flag": danger_flag,
        "danger_level": danger_level,
        "threshold": float(frame_threshold),
    }

    return MonitoringSummary(
        current_state=current_state,
        anomaly_series=anomaly_series,
        vibration_series=vibration_series,
        fft_snapshot=fft_snapshot,
        records=[record.to_dict() for record in records],
    )


def _compute_vibration_rms(frame: pd.DataFrame, axes: Iterable[str] = ("vx", "vy", "vz")) -> float:
    available_axes = [axis for axis in axes if axis in frame.columns]
    if not available_axes:
        return float("nan")
    values = frame[available_axes].to_numpy(dtype=float)
    return float(np.sqrt(np.mean(values ** 2)))


def _extract_timestamp(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame.columns or frame.empty:
        return None
    value = frame[column].iloc[-1]
    if isinstance(value, (np.datetime64, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    return str(value)


def _danger_level(score: float, threshold: float) -> str:
    if not np.isfinite(score) or threshold <= 0:
        return "unknown"
    ratio = score / threshold
    if ratio >= 1.2:
        return "위험"
    if ratio >= 0.8:
        return "주의"
    return "정상"


def _compute_fft_snapshot(
    frame: pd.DataFrame,
    sensor_columns: Sequence[str],
    window_config: WindowingConfig,
) -> dict | None:
    if frame.empty:
        return None
    try:
        frequencies, amplitudes = dataframe_to_windows(
            frame,
            sensor_columns=sensor_columns,
            config=window_config,
        )
    except ValueError:
        return None

    amplitude_mean = amplitudes.mean(axis=0)  # (freq_bins, sensors)
    amplitude_scalar = amplitude_mean.mean(axis=1)
    return {
        "freq_bins": frequencies.tolist(),
        "amplitude": amplitude_scalar.tolist(),
    }

