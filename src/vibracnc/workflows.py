from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from vibracnc.anomaly.autoencoder import AutoencoderConfig, LSTMAutoencoder, save_model
from vibracnc.anomaly.pipeline import AnomalyDetectionArtifacts, classify_anomalies, score_with_artifacts, train_normal_model
from vibracnc.config import DatasetConfig, ProjectPaths
from vibracnc.data.loader import load_condition, load_csv
from vibracnc.data.preprocessing import WindowingConfig
from vibracnc.data.targets import load_wear_file
from vibracnc.rul.features import FeatureExtractionConfig, extract_features
from vibracnc.rul.trainer import RULModelConfig, compute_feature_importance, leave_one_group_out_cv, prepare_dataset, train_random_forest_regressor


def download_dataset(dataset_config: DatasetConfig) -> Path:
    from vibracnc.data.download import download_phm2010_dataset

    dataset_root = download_phm2010_dataset(dataset_config.root_dir, force=False)
    return dataset_root


def collect_normal_frames(dataset_root: Path, dataset_config: DatasetConfig, per_condition_limit: int = 30) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for condition in dataset_config.normal_conditions:
        frames.extend(
            load_condition(
                dataset_root,
                condition,
                column_names=dataset_config.resolve_column_names(),
                limit=per_condition_limit,
            )
        )
    return frames


def train_anomaly_detection(
    dataset_root: Path,
    dataset_config: DatasetConfig,
    project_paths: ProjectPaths,
    autoencoder_config: Optional[AutoencoderConfig] = None,
    per_condition_limit: int = 30,
) -> tuple[LSTMAutoencoder, AnomalyDetectionArtifacts]:
    window_config = WindowingConfig.from_seconds(
        window_seconds=dataset_config.window_seconds,
        step_seconds=dataset_config.step_seconds,
        sampling_rate=dataset_config.sampling_rate,
    )
    normal_frames = collect_normal_frames(dataset_root, dataset_config, per_condition_limit=per_condition_limit)

    if autoencoder_config is None:
        autoencoder_config = AutoencoderConfig(
            input_dim=len(dataset_config.fft_columns),
            seq_len=128,
        )

    model, artifacts = train_normal_model(
        normal_frames,
        sensor_columns=dataset_config.fft_columns,
        window_config=window_config,
        autoencoder_config=autoencoder_config,
    )

    project_paths.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = project_paths.models_dir / "anomaly_autoencoder.pt"
    metadata_path = project_paths.models_dir / "anomaly_artifacts.json"
    save_model(model, model_path)

    metadata = {
        "threshold": artifacts.threshold,
        "frequencies": artifacts.frequencies.tolist(),
        "train_history": artifacts.train_history,
        "config": asdict(artifacts.config),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return model, artifacts


def evaluate_anomaly_on_condition(
    model: LSTMAutoencoder,
    artifacts: AnomalyDetectionArtifacts,
    dataset_root: Path,
    condition: str,
    dataset_config: DatasetConfig,
    per_condition_limit: Optional[int] = None,
) -> pd.DataFrame:
    window_config = WindowingConfig.from_seconds(
        window_seconds=dataset_config.window_seconds,
        step_seconds=dataset_config.step_seconds,
        sampling_rate=dataset_config.sampling_rate,
    )
    frames = load_condition(
        dataset_root,
        condition,
        column_names=dataset_config.resolve_column_names(),
        limit=per_condition_limit,
    )
    errors = score_with_artifacts(model, frames, dataset_config.fft_columns, window_config, artifacts)
    labels = classify_anomalies(errors, artifacts.threshold)
    return pd.DataFrame({"error": errors, "anomaly": labels})


def train_rul_model(
    dataset_root: Path,
    dataset_config: DatasetConfig,
    project_paths: ProjectPaths,
    model_config: Optional[RULModelConfig] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if model_config is None:
        model_config = RULModelConfig()

    wear_df = load_wear_file(dataset_root / dataset_config.wear_file)
    feature_records: list[pd.Series] = []
    targets: list[float] = []
    groups: list[str] = []

    feature_config = FeatureExtractionConfig(sensor_columns=dataset_config.sensor_columns)

    for _, row in wear_df.iterrows():
        file_rel_path = Path(row["file"])
        data_path = dataset_root / file_rel_path
        if not data_path.exists():
            raise FileNotFoundError(f"{data_path} 경로가 존재하지 않습니다.")
        df = load_csv(data_path, column_names=dataset_config.resolve_column_names())
        features = extract_features(df, feature_config)
        group_value = row.get("group", file_rel_path.parent.name)
        features["group_id"] = group_value
        feature_records.append(features)
        target_value = row.get("wear", row.get("VB", row.get("target")))
        if target_value is None:
            raise KeyError("wear 파일에 'wear' 또는 'VB' 컬럼이 필요합니다.")
        targets.append(float(target_value))
        groups.append(group_value)

    X, y, groups_arr = prepare_dataset(feature_records, targets, groups)
    if groups_arr is None:
        raise ValueError("Leave-one-out 평가를 위해 group 정보가 필요합니다.")
    evaluation = leave_one_group_out_cv(X, y, groups_arr, model_config)

    model = train_random_forest_regressor(X, y, model_config)
    feature_importance = compute_feature_importance(model, X.columns)

    project_paths.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = project_paths.models_dir / "rul_random_forest.pkl"
    joblib.dump({"model": model, "scaler": None, "columns": X.columns.tolist()}, model_path)

    feature_importance_path = project_paths.models_dir / "rul_feature_importance.csv"
    feature_importance.to_csv(feature_importance_path)

    evaluation_df = pd.DataFrame(
        {
            "prediction": evaluation.predictions,
            "ground_truth": evaluation.ground_truth,
            "group": evaluation.groups,
        }
    )

    metrics_df = pd.DataFrame(
        [
            {
                "rmse": evaluation.rmse,
                "mae": evaluation.mae,
                "r2": evaluation.r2,
            }
        ]
    )

    project_paths.figures_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = project_paths.figures_dir / "rul_predictions.csv"
    evaluation_df.to_csv(evaluation_path, index=False)
    metrics_path = project_paths.figures_dir / "rul_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    return evaluation_df, feature_importance.reset_index(names=["feature", "importance"])

