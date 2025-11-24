from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from vibracnc.config import RuleDefinition
from vibracnc.data.preprocessing import WindowingConfig, ensure_sorted_by_timestamp


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


METRIC_FUNCTIONS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda x: float(np.mean(x)),
    "max": lambda x: float(np.max(x)),
    "min": lambda x: float(np.min(x)),
    "std": lambda x: float(np.std(x)),
    "median": lambda x: float(np.median(x)),
    "rms": _rms,
}

OPERATORS: dict[str, Callable[[float, float], bool]] = {
    "gt": lambda value, threshold: value > threshold,
    "ge": lambda value, threshold: value >= threshold,
    "lt": lambda value, threshold: value < threshold,
    "le": lambda value, threshold: value <= threshold,
}


def evaluate_rules_on_frame(
    frame: pd.DataFrame,
    sensor_columns: Sequence[str],
    rules: Sequence[RuleDefinition],
    window_config: WindowingConfig,
    timestamp_col: str = "timestamp",
    window_offset: int = 0,
) -> tuple[pd.DataFrame, int]:
    """
    주어진 DataFrame에 대해 슬라이딩 윈도우를 생성하고 규칙을 평가합니다.

    Returns
    -------
    (rule_results_df, num_windows)
    """
    if not rules:
        raise ValueError("평가할 규칙이 없습니다.")

    sorted_frame = ensure_sorted_by_timestamp(frame, timestamp_col)
    if not set(sensor_columns).issubset(sorted_frame.columns):
        missing = set(sensor_columns) - set(sorted_frame.columns)
        raise ValueError(f"다음 센서 컬럼이 없습니다: {missing}")

    signal = sorted_frame[list(sensor_columns)].to_numpy(dtype=float)
    timestamps = sorted_frame[timestamp_col].to_numpy()

    total = signal.shape[0]
    window_size = window_config.window_size
    stride = window_config.stride
    if total < window_size:
        return (
            pd.DataFrame(
                columns=[
                    "window_index",
                    "rule",
                    "column",
                    "metric",
                    "operator",
                    "threshold",
                    "value",
                    "violated",
                    "description",
                    "start_timestamp",
                    "end_timestamp",
                ]
            ),
            0,
        )

    column_index = {column: idx for idx, column in enumerate(sensor_columns)}
    metric_cache: dict[str, Callable[[np.ndarray], float]] = {}

    records: list[dict] = []
    window_counter = 0

    for start in range(0, total - window_size + 1, stride):
        window = signal[start : start + window_size]
        window_start_ts = timestamps[start]
        window_end_ts = timestamps[start + window_size - 1]

        for rule in rules:
            if rule.column not in column_index:
                raise ValueError(f"규칙 {rule.name}에 정의된 컬럼 '{rule.column}'이 데이터에 없습니다.")
            metric_fn = metric_cache.get(rule.metric)
            if metric_fn is None:
                if rule.metric not in METRIC_FUNCTIONS:
                    raise ValueError(f"지원하지 않는 metric '{rule.metric}' 입니다.")
                metric_fn = METRIC_FUNCTIONS[rule.metric]
                metric_cache[rule.metric] = metric_fn

            operator_fn = OPERATORS.get(rule.operator)
            if operator_fn is None:
                raise ValueError(f"지원하지 않는 operator '{rule.operator}' 입니다.")

            column_values = window[:, column_index[rule.column]]
            metric_value = metric_fn(column_values)
            violated = operator_fn(metric_value, rule.threshold)

            records.append(
                {
                    "window_index": window_counter + window_offset,
                    "window_in_file": window_counter,
                    "rule": rule.name,
                    "column": rule.column,
                    "metric": rule.metric,
                    "operator": rule.operator,
                    "threshold": rule.threshold,
                    "value": metric_value,
                    "violated": violated,
                    "description": rule.description or "",
                    "start_timestamp": window_start_ts,
                    "end_timestamp": window_end_ts,
                }
            )
        window_counter += 1

    return pd.DataFrame(records), window_counter

