from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class RULModelConfig:
    n_estimators: int = 300
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    random_state: int = 42
    n_jobs: int = -1


@dataclass(slots=True)
class EvaluationResult:
    rmse: float
    mae: float
    r2: float
    predictions: np.ndarray
    ground_truth: np.ndarray
    groups: np.ndarray


def prepare_dataset(
    feature_records: Sequence[pd.Series],
    target: Sequence[float],
    groups: Optional[Sequence[str | int]] = None,
) -> tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
    features_df = pd.DataFrame(feature_records).fillna(0.0)
    y = np.asarray(target, dtype=float)
    groups_arr = np.asarray(groups) if groups is not None else None
    if "group_id" in features_df.columns:
        group_values = features_df.pop("group_id").to_numpy()
        if groups_arr is None:
            groups_arr = group_values
    return features_df, y, groups_arr


def train_random_forest_regressor(
    X: pd.DataFrame, y: np.ndarray, config: RULModelConfig
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    model.fit(X, y)
    return model


def leave_one_group_out_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    config: RULModelConfig,
) -> EvaluationResult:
    logo = LeaveOneGroupOut()
    preds: list[float] = []
    actuals: list[float] = []
    group_ids: list[str | int] = []

    feature_names = X.columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    for train_idx, test_idx in logo.split(X_scaled, y, groups):
        X_train = X_scaled_df.iloc[train_idx]
        X_test = X_scaled_df.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = train_random_forest_regressor(X_train, y_train, config)
        y_pred = model.predict(X_test)
        preds.extend(y_pred.tolist())
        actuals.extend(y_test.tolist())
        group_ids.extend(groups[test_idx])

    y_pred_arr = np.asarray(preds)
    y_true_arr = np.asarray(actuals)
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    return EvaluationResult(
        rmse=rmse,
        mae=mae,
        r2=r2,
        predictions=y_pred_arr,
        ground_truth=y_true_arr,
        groups=np.asarray(group_ids),
    )


def compute_feature_importance(model: RandomForestRegressor, feature_names: Sequence[str]) -> pd.Series:
    importances = model.feature_importances_
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)

