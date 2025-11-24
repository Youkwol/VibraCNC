from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(slots=True)
class AnalysisReport:
    feature_importance: list[dict]
    correlation_matrix: list[dict]
    comparison_table: list[dict]
    important_freq_band: tuple[float, float] | None

    def to_dict(self) -> dict:
        return {
            "feature_importance": self.feature_importance,
            "correlation_matrix": self.correlation_matrix,
            "comparison_table": self.comparison_table,
            "important_freq_band": self.important_freq_band,
        }


def build_analysis_report(
    anomaly_csv: Path,
    feature_importance_csv: Path,
    frequency_band_csv: Path | None = None,
) -> AnalysisReport:
    anomaly_df = pd.read_csv(anomaly_csv)
    feature_importance_df = pd.read_csv(feature_importance_csv)

    # normalize unnamed columns
    if "feature" not in feature_importance_df.columns and len(feature_importance_df.columns) >= 2:
        feature_importance_df = feature_importance_df.rename(
            columns={feature_importance_df.columns[0]: "feature", feature_importance_df.columns[1]: "importance"}
        )

    feature_top5 = _top_feature_importance(feature_importance_df, top_k=5)
    correlation = _sensor_correlation(anomaly_df)
    comparison = _normal_vs_anomaly_stats(anomaly_df)
    freq_band = _important_frequency_band(frequency_band_csv)

    return AnalysisReport(
        feature_importance=feature_top5,
        correlation_matrix=correlation,
        comparison_table=comparison,
        important_freq_band=freq_band,
    )


def _top_feature_importance(df: pd.DataFrame, top_k: int = 5) -> list[dict]:
    importance_col = None
    for candidate in ("importance", "importance_", "value"):
        if candidate in df.columns:
            importance_col = candidate
            break
    if importance_col is None:
        raise KeyError("feature_importance.csv must contain 'importance' column")
    sorted_df = df.sort_values(importance_col, ascending=False).head(top_k)
    return [
        {"feature": row["feature"], "importance": float(row[importance_col])}
        for _, row in sorted_df.iterrows()
    ]


def _sensor_correlation(df: pd.DataFrame) -> list[dict]:
    sensor_cols = [col for col in df.columns if col.startswith("sensor_") or col in {"vx", "vy", "vz", "sx", "sy", "sz"}]
    if not sensor_cols:
        return []
    corr = df[sensor_cols].corr().round(3)
    corr_records = []
    for row_name, row in corr.iterrows():
        corr_records.append({"sensor": row_name, "values": row.to_dict()})
    return corr_records


def _normal_vs_anomaly_stats(df: pd.DataFrame) -> list[dict]:
    if "anomaly" not in df.columns:
        raise KeyError("anomaly column missing in anomaly results")

    metrics = []
    columns = ["vibration_rms", "force_rms", "temperature"]
    for column in columns:
        if column not in df.columns:
            continue
        normal_mean = float(df.loc[df["anomaly"] == 0, column].mean())
        anomaly_mean = float(df.loc[df["anomaly"] == 1, column].mean())
        metrics.append(
            {
                "metric": column,
                "normal_mean": normal_mean,
                "anomaly_mean": anomaly_mean,
            }
        )
    return metrics


def _important_frequency_band(csv_path: Path | None) -> tuple[float, float] | None:
    if csv_path is None or not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if not {"freq_start", "freq_end"}.issubset(df.columns):
        return None
    row = df.sort_values("importance", ascending=False).iloc[0]
    return float(row["freq_start"]), float(row["freq_end"])

