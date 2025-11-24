from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from vibracnc.anomaly.autoencoder import AutoencoderConfig, LSTMAutoencoder, load_model, save_model
from vibracnc.anomaly.pipeline import AnomalyDetectionArtifacts, classify_anomalies, score_with_artifacts, train_normal_model
from vibracnc.anomaly.rules import evaluate_rules_on_frame
from vibracnc.config import DatasetConfig, ProjectPaths, RuleDefinition
from vibracnc.data.loader import discover_csv_files, load_condition, load_csv
from vibracnc.data.preprocessing import WindowingConfig
from vibracnc.data.targets import load_wear_file
from vibracnc.rul.features import FeatureExtractionConfig, extract_features
from vibracnc.rul.trainer import RULModelConfig, compute_feature_importance, leave_one_group_out_cv, prepare_dataset, train_random_forest_regressor


def download_dataset(dataset_config: DatasetConfig) -> Path:
    from vibracnc.data.download import download_phm2010_dataset

    dataset_root = download_phm2010_dataset(dataset_config.root_dir, force=False)
    return dataset_root


def load_anomaly_artifacts(metadata_path: Path) -> AnomalyDetectionArtifacts:
    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} 파일을 찾을 수 없습니다. 먼저 학습을 수행하세요.")

    data = json.loads(metadata_path.read_text())
    config = AutoencoderConfig(**data["config"])
    frequencies = np.array(data["frequencies"], dtype=float)
    threshold = float(data["threshold"])
    frame_threshold = float(data.get("frame_threshold", threshold))  # 하위 호환성
    history = data.get("train_history", {})
    norm_min = np.array(data["norm_min"], dtype=float) if "norm_min" in data and data["norm_min"] is not None else None
    norm_max = np.array(data["norm_max"], dtype=float) if "norm_max" in data and data["norm_max"] is not None else None
    return AnomalyDetectionArtifacts(
        frequencies=frequencies,
        train_history=history,
        threshold=threshold,
        frame_threshold=frame_threshold,
        errors=np.array([]),
        config=config,
        norm_min=norm_min,
        norm_max=norm_max,
    )


def load_anomaly_resources(models_dir: Path) -> tuple[LSTMAutoencoder, AnomalyDetectionArtifacts]:
    metadata_path = models_dir / "anomaly_artifacts.json"
    model_path = models_dir / "anomaly_autoencoder.pt"
    artifacts = load_anomaly_artifacts(metadata_path)
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} 파일이 없습니다. 학습된 가중치를 먼저 생성하세요.")
    model = load_model(model_path, artifacts.config)
    model.eval()
    return model, artifacts


def collect_normal_frames(
    dataset_root: Path, 
    dataset_config: DatasetConfig, 
    per_condition_limit: int | None = None,
    max_wear: float | None = None,
) -> list[pd.DataFrame]:
    """
    정상 상태 데이터를 수집합니다. 초기 cut만 사용합니다.
    
    Args:
        per_condition_limit: 각 조건당 사용할 데이터 개수. 기본값: 20 (초기 20개 cut만 사용).
        max_wear: 정상으로 간주할 최대 마모량 (μm). None이면 wear.csv를 사용하지 않음.
    """
    frames: list[pd.DataFrame] = []
    
    for condition in dataset_config.normal_conditions:
        condition_dir = dataset_root / condition
        if not condition_dir.exists():
            continue
            
        csv_files = sorted(condition_dir.glob("*.csv"))
        
        # per_condition_limit 적용 (초기 cut만 사용)
        if per_condition_limit is not None:
            csv_files = csv_files[:per_condition_limit]
            limit_str = str(per_condition_limit)
        else:
            limit_str = "all"
        
        print(f"[train-anomaly] loading condition '{condition}' (first {limit_str} cuts)", flush=True)
        
        # 프레임 로드
        condition_frames = [
            load_csv(path, column_names=dataset_config.resolve_column_names()) 
            for path in csv_files
        ]
        
        print(f"[train-anomaly] loaded condition '{condition}' -> {len(condition_frames)} frames", flush=True)
        frames.extend(condition_frames)
    
    return frames


def train_anomaly_detection(
    dataset_root: Path,
    dataset_config: DatasetConfig,
    project_paths: ProjectPaths,
    autoencoder_config: Optional[AutoencoderConfig] = None,
    per_condition_limit: int | None = None,
    max_wear: float | None = None,
) -> tuple[LSTMAutoencoder, AnomalyDetectionArtifacts]:
    window_config = WindowingConfig.from_seconds(
        window_seconds=dataset_config.window_seconds,
        step_seconds=dataset_config.step_seconds,
        sampling_rate=dataset_config.sampling_rate,
    )
    normal_frames = collect_normal_frames(
        dataset_root, 
        dataset_config, 
        per_condition_limit=per_condition_limit,
        max_wear=max_wear,
    )

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
        "threshold": float(artifacts.threshold),
        "frame_threshold": float(artifacts.frame_threshold),
        "frequencies": artifacts.frequencies.tolist(),
        "train_history": artifacts.train_history,
        "config": asdict(artifacts.config),
        "norm_min": artifacts.norm_min.tolist() if artifacts.norm_min is not None else None,
        "norm_max": artifacts.norm_max.tolist() if artifacts.norm_max is not None else None,
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


def run_anomaly_inference(
    model: LSTMAutoencoder,
    artifacts: AnomalyDetectionArtifacts,
    dataset_root: Path,
    dataset_config: DatasetConfig,
    condition: str,
    per_condition_limit: Optional[int] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    단일 조건에 대해 이상 탐지를 수행하고 옵션에 따라 CSV로 저장한다.
    """
    results = evaluate_anomaly_on_condition(
        model,
        artifacts,
        dataset_root,
        condition,
        dataset_config,
        per_condition_limit=per_condition_limit,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
    return results


def run_rule_based_detection(
    dataset_root: Path,
    dataset_config: DatasetConfig,
    condition: str,
    rules: Optional[Sequence[RuleDefinition]] = None,
    per_condition_limit: Optional[int] = None,
    window_config: Optional[WindowingConfig] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    if window_config is None:
        window_config = WindowingConfig.from_seconds(
            window_seconds=dataset_config.window_seconds,
            step_seconds=dataset_config.step_seconds,
            sampling_rate=dataset_config.sampling_rate,
        )

    active_rules: Sequence[RuleDefinition] = rules or dataset_config.rule_definitions
    if not active_rules:
        raise ValueError("적용할 규칙 정의가 없습니다.")

    condition_dir = dataset_root / condition
    csv_files = discover_csv_files(condition_dir)
    if per_condition_limit is not None:
        csv_files = csv_files[:per_condition_limit]

    if not csv_files:
        return pd.DataFrame()

    records: list[pd.DataFrame] = []
    window_offset = 0
    column_names = dataset_config.resolve_column_names()

    for path in csv_files:
        frame = load_csv(path, column_names=column_names)
        rule_df, num_windows = evaluate_rules_on_frame(
            frame,
            sensor_columns=dataset_config.sensor_columns,
            rules=active_rules,
            window_config=window_config,
            timestamp_col=dataset_config.timestamp_column,
            window_offset=window_offset,
        )
        if num_windows == 0:
            continue
        rule_df["condition"] = condition
        rule_df["source_file"] = str(path.relative_to(dataset_root))
        records.append(rule_df)
        window_offset += num_windows

    if not records:
        return pd.DataFrame()

    result = pd.concat(records, ignore_index=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
    return result


def summarize_rule_results(result: pd.DataFrame) -> tuple[int, int]:
    if result.empty:
        return 0, 0
    total_windows = result["window_index"].nunique()
    violated_windows = result.loc[result["violated"]]["window_index"].nunique()
    return total_windows, violated_windows


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

    feature_importance_df = feature_importance.reset_index()
    feature_importance_df.columns = ["feature", "importance"]
    return evaluation_df, feature_importance_df

