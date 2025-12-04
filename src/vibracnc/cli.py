from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vibracnc.anomaly.autoencoder import AutoencoderConfig, load_model
from vibracnc.analysis import build_analysis_report
from vibracnc.config import DatasetConfig, ProjectPaths
from vibracnc.diagnostics import build_diagnostics_report
from vibracnc.data.preprocessing import WindowingConfig
from vibracnc.monitoring import build_monitoring_summary, load_recent_cut_samples
from vibracnc.workflows import (
    download_dataset,
    load_anomaly_artifacts,
    run_anomaly_inference,
    run_rule_based_detection,
    summarize_rule_results,
    train_anomaly_detection,
    train_rul_model,
)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Choose --device cpu or install CUDA-enabled PyTorch.")
    return device_arg


def train_anomaly_command(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir).expanduser()
    models_dir = Path(args.models_dir).expanduser()
    figures_dir = Path(args.figures_dir).expanduser()

    dataset_config = DatasetConfig(
        root_dir=dataset_dir,
        sampling_rate=args.sampling_rate,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
    )
    project_paths = ProjectPaths(
        data_dir=dataset_dir,
        models_dir=models_dir,
        figures_dir=figures_dir,
    )

    if args.download:
        download_dataset(dataset_config)

    device = resolve_device(args.device)
    print(f"[train-anomaly] device={device}")

    autoencoder_config = AutoencoderConfig(
        input_dim=len(dataset_config.fft_columns),
        seq_len=args.seq_len,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
    )

    train_anomaly_detection(
        dataset_dir,
        dataset_config,
        project_paths,
        autoencoder_config=autoencoder_config,
        per_condition_limit=args.per_condition_limit,
        max_wear=args.max_wear,
    )


def train_rul_command(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir).expanduser()
    models_dir = Path(args.models_dir).expanduser()
    figures_dir = Path(args.figures_dir).expanduser()

    dataset_config = DatasetConfig(
        root_dir=dataset_dir,
        sampling_rate=args.sampling_rate,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
    )
    project_paths = ProjectPaths(
        data_dir=dataset_dir,
        models_dir=models_dir,
        figures_dir=figures_dir,
    )

    train_rul_model(dataset_dir, dataset_config, project_paths)


def infer_anomaly_command(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir).expanduser()
    models_dir = Path(args.models_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else Path("artifacts/figures/anomaly")

    dataset_config = DatasetConfig(
        root_dir=dataset_dir,
        sampling_rate=args.sampling_rate,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
    )

    metadata_path = models_dir / "anomaly_artifacts.json"
    model_path = models_dir / "anomaly_autoencoder.pt"

    device = resolve_device(args.device)
    artifacts = load_anomaly_artifacts(metadata_path)
    artifacts.config.device = device
    model = load_model(model_path, artifacts.config)

    conditions = args.conditions or dataset_config.normal_conditions
    for condition in conditions:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{condition}_anomaly.csv"

        results = run_anomaly_inference(
            model,
            artifacts,
            dataset_dir,
            dataset_config,
            condition,
            per_condition_limit=args.per_condition_limit,
            output_path=output_path,
        )
        anomaly_ratio = (results["anomaly"].mean() * 100) if not results.empty else 0.0
        print(
            f"[infer-anomaly] condition={condition} windows={len(results)} "
            f"anomaly_ratio={anomaly_ratio:.1f}% threshold={artifacts.threshold:.4f}"
        )


def rule_anomaly_command(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = DatasetConfig(
        root_dir=dataset_dir,
        sampling_rate=args.sampling_rate,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
    )

    conditions = args.conditions or dataset_config.normal_conditions
    for condition in conditions:
        condition_output = output_dir / f"{condition}_rule_based.csv"
        result = run_rule_based_detection(
            dataset_dir,
            dataset_config,
            condition,
            per_condition_limit=args.per_condition_limit,
            rules=dataset_config.rule_definitions,
            window_config=None,
            output_path=condition_output,
        )
        windows, violated = summarize_rule_results(result)
        print(
            f"[rule-anomaly] condition={condition} windows={windows} "
            f"violated_windows={violated} output={condition_output}"
        )


def monitoring_report_command(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir).expanduser()
    models_dir = Path(args.models_dir).expanduser()
    output_path = Path(args.output_path).expanduser()

    dataset_config = DatasetConfig(
        root_dir=dataset_dir,
        sampling_rate=args.sampling_rate,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
    )
    window_config = WindowingConfig.from_seconds(
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        sampling_rate=args.sampling_rate,
    )

    metadata_path = models_dir / "anomaly_artifacts.json"
    model_path = models_dir / "anomaly_autoencoder.pt"
    artifacts = load_anomaly_artifacts(metadata_path)

    device = resolve_device(args.device)
    artifacts.config.device = device
    model = load_model(model_path, artifacts.config)

    conditions = args.conditions or dataset_config.normal_conditions
    # 모니터링 리포트는 최근 데이터를 사용 (실시간 모니터링 시나리오)
    samples = load_recent_cut_samples(
        dataset_dir,
        dataset_config,
        conditions=conditions,
        per_condition_limit=args.per_condition_limit,
        use_recent=True,  # 최근 데이터 사용 (실시간 모니터링)
    )
    if not samples:
        raise ValueError("모니터링할 cut 샘플을 찾을 수 없습니다.")

    summary = build_monitoring_summary(samples, model, artifacts, dataset_config, window_config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[monitoring-report] "
        f"cuts={len(samples)} conditions={','.join(conditions)} output={output_path}"
    )


def diagnostics_report_command(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir).expanduser()
    models_dir = Path(args.models_dir).expanduser()
    rul_predictions = Path(args.rul_predictions).expanduser()
    output_path = Path(args.output_path).expanduser()

    dataset_config = DatasetConfig(
        root_dir=dataset_dir,
        sampling_rate=args.sampling_rate,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
    )

    report = build_diagnostics_report(
        dataset_dir=dataset_dir,
        dataset_config=dataset_config,
        rul_predictions_path=rul_predictions,
        wear_csv_path=(Path(args.wear_csv).expanduser() if args.wear_csv else None),
        max_wear_limit=args.max_wear_limit,
        rul_min_padding=args.rul_min_padding,
        rul_max_padding=args.rul_max_padding,
        cut_per_hour=args.cut_per_hour,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[diagnostics-report] "
        f"rul={report.rul_cuts:.1f} cuts max_wear={report.max_wear_limit:.1f} output={output_path}"
    )


def analysis_report_command(args: argparse.Namespace) -> None:
    anomaly_csv = Path(args.anomaly_csv).expanduser()
    feature_importance_csv = Path(args.feature_importance_csv).expanduser()
    freq_band_csv = Path(args.freq_band_csv).expanduser() if args.freq_band_csv else None
    output_path = Path(args.output_path).expanduser()

    report = build_analysis_report(anomaly_csv, feature_importance_csv, freq_band_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print("[analysis-report]", f"output={output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VibraCNC CLI")
    subparsers = parser.add_subparsers(dest="command")

    anomaly_parser = subparsers.add_parser("train-anomaly", help="이상 탐지 모델 학습")
    anomaly_parser.add_argument("--dataset-dir", default="data/phm2010")
    anomaly_parser.add_argument("--models-dir", default="artifacts/models")
    anomaly_parser.add_argument("--figures-dir", default="artifacts/figures")
    anomaly_parser.add_argument("--download", action="store_true", help="kagglehub로 데이터셋 다운로드")
    anomaly_parser.add_argument("--sampling-rate", type=float, default=25600.0)
    anomaly_parser.add_argument("--window-seconds", type=float, default=0.1)
    anomaly_parser.add_argument("--step-seconds", type=float, default=0.05)
    anomaly_parser.add_argument("--seq-len", type=int, default=128)
    anomaly_parser.add_argument("--latent-dim", type=int, default=32)
    anomaly_parser.add_argument("--hidden-dim", type=int, default=64)
    anomaly_parser.add_argument("--num-layers", type=int, default=2)
    anomaly_parser.add_argument("--dropout", type=float, default=0.1)
    anomaly_parser.add_argument("--learning-rate", type=float, default=1e-3)
    anomaly_parser.add_argument("--epochs", type=int, default=5)
    anomaly_parser.add_argument("--batch-size", type=int, default=64)
    anomaly_parser.add_argument(
        "--per-condition-limit",
        type=int,
        default=10,
        help="각 조건당 사용할 데이터 개수. 기본값: 10 (초기 10개 cut만 사용)",
    )
    anomaly_parser.add_argument(
        "--max-wear",
        type=float,
        default=None,
        help="정상으로 간주할 최대 마모량 (μm). wear.csv를 읽어서 wear < max_wear인 cut만 정상으로 사용. 예: --max-wear 100.0",
    )
    anomaly_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Device to run training on.")
    anomaly_parser.set_defaults(func=train_anomaly_command)

    rul_parser = subparsers.add_parser("train-rul", help="RUL 회귀 모델 학습")
    rul_parser.add_argument("--dataset-dir", default="data/phm2010")
    rul_parser.add_argument("--models-dir", default="artifacts/models")
    rul_parser.add_argument("--figures-dir", default="artifacts/figures")
    rul_parser.add_argument("--sampling-rate", type=float, default=25600.0)
    rul_parser.add_argument("--window-seconds", type=float, default=0.1)
    rul_parser.add_argument("--step-seconds", type=float, default=0.05)
    rul_parser.set_defaults(func=train_rul_command)

    infer_parser = subparsers.add_parser("infer-anomaly", help="학습된 이상 탐지 모델로 재구성 오차 계산")
    infer_parser.add_argument("--dataset-dir", default="data/phm2010")
    infer_parser.add_argument("--models-dir", default="artifacts/models")
    infer_parser.add_argument("--sampling-rate", type=float, default=25600.0)
    infer_parser.add_argument("--window-seconds", type=float, default=0.1)
    infer_parser.add_argument("--step-seconds", type=float, default=0.05)
    infer_parser.add_argument(
        "--conditions",
        nargs="+",
        help="분석할 조건 목록 (미지정 시 DatasetConfig.normal_conditions 사용)",
    )
    infer_parser.add_argument("--per-condition-limit", type=int, default=50)
    infer_parser.add_argument("--output-dir", default="artifacts/figures/anomaly", help="CSV 저장 디렉터리")
    infer_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="추론에 사용할 장치")
    infer_parser.set_defaults(func=infer_anomaly_command)

    monitoring_parser = subparsers.add_parser("monitoring-report", help="실시간 모니터링 리포트 생성")
    monitoring_parser.add_argument("--dataset-dir", default="data/phm2010")
    monitoring_parser.add_argument("--models-dir", default="artifacts/models")
    monitoring_parser.add_argument("--output-path", default="artifacts/monitoring/monitoring_report.json")
    monitoring_parser.add_argument("--sampling-rate", type=float, default=25600.0)
    monitoring_parser.add_argument("--window-seconds", type=float, default=0.1)
    monitoring_parser.add_argument("--step-seconds", type=float, default=0.05)
    monitoring_parser.add_argument(
        "--conditions",
        nargs="+",
        help="모니터링할 조건 목록 (미지정 시 DatasetConfig.normal_conditions 사용)",
    )
    monitoring_parser.add_argument("--per-condition-limit", type=int, default=50)
    monitoring_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="추론에 사용할 장치")
    monitoring_parser.set_defaults(func=monitoring_report_command)

    diagnostics_parser = subparsers.add_parser("diagnostics-report", help="RUL/마모 진단 리포트 생성")
    diagnostics_parser.add_argument("--dataset-dir", default="data/phm2010")
    diagnostics_parser.add_argument("--models-dir", default="artifacts/models")
    diagnostics_parser.add_argument(
        "--rul-predictions",
        default="artifacts/figures/rul_predictions.csv",
        help="RUL 예측 CSV 경로",
    )
    diagnostics_parser.add_argument("--wear-csv", help="wear.csv 경로 (선택)")
    diagnostics_parser.add_argument("--output-path", default="artifacts/monitoring/diagnostics_report.json")
    diagnostics_parser.add_argument("--sampling-rate", type=float, default=25600.0)
    diagnostics_parser.add_argument("--window-seconds", type=float, default=0.1)
    diagnostics_parser.add_argument("--step-seconds", type=float, default=0.05)
    diagnostics_parser.add_argument("--max-wear-limit", type=float, default=200.0)
    diagnostics_parser.add_argument("--rul-min-padding", type=float, default=0.8)
    diagnostics_parser.add_argument("--rul-max-padding", type=float, default=1.2)
    diagnostics_parser.add_argument("--cut-per-hour", type=float, help="시간당 컷 수행 개수 (예상 고장 시각 계산용)")
    diagnostics_parser.set_defaults(func=diagnostics_report_command)

    analysis_parser = subparsers.add_parser("analysis-report", help="심층 분석 리포트 생성")
    analysis_parser.add_argument("--anomaly-csv", default="artifacts/figures/c1_anomaly.csv", help="이상 탐지 결과 CSV")
    analysis_parser.add_argument(
        "--feature-importance-csv",
        default="artifacts/models/rul_feature_importance.csv",
        help="feature importance CSV",
    )
    analysis_parser.add_argument("--freq-band-csv", help="주파수 대역 정보 CSV (선택)")
    analysis_parser.add_argument("--output-path", default="artifacts/monitoring/analysis_report.json")
    analysis_parser.set_defaults(func=analysis_report_command)

    rule_parser = subparsers.add_parser("rule-anomaly", help="규칙 기반 이상 탐지 실행")
    rule_parser.add_argument("--dataset-dir", default="data/phm2010")
    rule_parser.add_argument("--sampling-rate", type=float, default=25600.0)
    rule_parser.add_argument("--window-seconds", type=float, default=0.1)
    rule_parser.add_argument("--step-seconds", type=float, default=0.05)
    rule_parser.add_argument(
        "--conditions",
        nargs="+",
        help="분석할 조건 목록 (미지정 시 DatasetConfig.normal_conditions 사용)",
    )
    rule_parser.add_argument("--per-condition-limit", type=int, default=50)
    rule_parser.add_argument("--output-dir", default="artifacts/figures/rule_based", help="규칙 결과 CSV 디렉터리")
    rule_parser.set_defaults(func=rule_anomaly_command)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()

