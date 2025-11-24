from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from vibracnc.config import DatasetConfig
from vibracnc.data.loader import load_csv
from vibracnc.rul.features import FeatureExtractionConfig, extract_features


@dataclass(slots=True)
class DiagnosticsReport:
    rul_cuts: float
    rul_min_cuts: float
    rul_max_cuts: float
    current_wear: float
    wear_ratio_percent: float
    max_wear_limit: float
    predicted_failure_time: str | None
    wear_actual: list[dict]
    wear_predicted: list[dict]

    def to_dict(self) -> dict:
        return {
            "rul": {
                "cuts": self.rul_cuts,
                "min_cuts": self.rul_min_cuts,
                "max_cuts": self.rul_max_cuts,
                "predicted_failure_time": self.predicted_failure_time,
            },
            "wear": {
                "current": self.current_wear,
                "max_limit": self.max_wear_limit,
                "ratio_percent": self.wear_ratio_percent,
                "actual_series": self.wear_actual,
                "predicted_series": self.wear_predicted,
            },
        }


def build_diagnostics_report(
    dataset_dir: Path,
    dataset_config: DatasetConfig,
    rul_predictions_path: Path,
    wear_csv_path: Optional[Path] = None,
    max_wear_limit: float = 200.0,
    rul_min_padding: float = 0.8,
    rul_max_padding: float = 1.2,
    cut_per_hour: Optional[float] = None,
) -> DiagnosticsReport:
    rul_df = pd.read_csv(rul_predictions_path)
    wear_df = None

    if wear_csv_path is not None and wear_csv_path.exists():
        wear_df = pd.read_csv(wear_csv_path)

    latest_row = rul_df.tail(1).iloc[0]
    rul_cuts = float(latest_row["prediction"])
    rul_min_cuts = max(rul_cuts * rul_min_padding, 0.0)
    rul_max_cuts = max(rul_cuts * rul_max_padding, 0.0)

    current_wear = _estimate_current_wear(dataset_dir, dataset_config, latest_row, wear_df)
    wear_ratio_percent = min(current_wear / max_wear_limit * 100.0, 100.0)

    wear_actual_series = _build_wear_series(wear_df) if wear_df is not None else []
    wear_pred_series = _build_prediction_series(rul_df)

    predicted_failure_time = None
    if cut_per_hour and cut_per_hour > 0:
        hours_left = rul_cuts / cut_per_hour
        predicted_failure_time = (datetime.utcnow() + timedelta(hours=hours_left)).isoformat()

    return DiagnosticsReport(
        rul_cuts=rul_cuts,
        rul_min_cuts=rul_min_cuts,
        rul_max_cuts=rul_max_cuts,
        current_wear=current_wear,
        wear_ratio_percent=wear_ratio_percent,
        max_wear_limit=max_wear_limit,
        predicted_failure_time=predicted_failure_time,
        wear_actual=wear_actual_series,
        wear_predicted=wear_pred_series,
    )


def _estimate_current_wear(
    dataset_dir: Path,
    dataset_config: DatasetConfig,
    latest_prediction_row: pd.Series,
    wear_df: Optional[pd.DataFrame],
) -> float:
    if wear_df is not None and "wear" in wear_df.columns:
        return float(wear_df["wear"].max())

    file_rel_path = Path(latest_prediction_row.get("file", ""))
    if not file_rel_path:
        return 0.0

    data_path = dataset_dir / file_rel_path
    if not data_path.exists():
        return float(latest_prediction_row.get("prediction", 0.0))

    feature_config = FeatureExtractionConfig(sensor_columns=dataset_config.sensor_columns)
    frame = load_csv(data_path, column_names=dataset_config.resolve_column_names())
    features = extract_features(frame, feature_config)

    model_path = dataset_dir / ".." / dataset_config.models_dir / "rul_random_forest.pkl"
    model_path = model_path.resolve()
    if not model_path.exists():
        return float(latest_prediction_row.get("prediction", 0.0))

    saved = joblib.load(model_path)
    model = saved["model"]
    columns = saved["columns"]

    features = features.reindex(columns=columns, fill_value=0.0)
    predicted = model.predict(features.values.reshape(1, -1))[0]
    return float(predicted)


def _build_wear_series(wear_df: pd.DataFrame) -> list[dict]:
    series = []
    if "wear" not in wear_df.columns:
        return series
    for _, row in wear_df.iterrows():
        series.append(
            {
                "cut": int(row.get("cut", len(series) + 1)),
                "wear": float(row["wear"]),
            }
        )
    return series


def _build_prediction_series(rul_df: pd.DataFrame) -> list[dict]:
    series = []
    for idx, row in rul_df.iterrows():
        series.append(
            {
                "cut": int(idx + 1),
                "prediction": float(row.get("prediction", 0.0)),
                "ground_truth": float(row.get("ground_truth", 0.0)),
            }
        )
    return series

