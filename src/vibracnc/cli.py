from __future__ import annotations

import argparse
from pathlib import Path

from vibracnc.anomaly.autoencoder import AutoencoderConfig
from vibracnc.config import DatasetConfig, ProjectPaths
from vibracnc.workflows import download_dataset, train_anomaly_detection, train_rul_model


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
    )

    train_anomaly_detection(
        dataset_dir,
        dataset_config,
        project_paths,
        autoencoder_config=autoencoder_config,
        per_condition_limit=args.per_condition_limit,
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
    anomaly_parser.add_argument("--epochs", type=int, default=50)
    anomaly_parser.add_argument("--batch-size", type=int, default=64)
    anomaly_parser.add_argument("--per-condition-limit", type=int, default=30)
    anomaly_parser.set_defaults(func=train_anomaly_command)

    rul_parser = subparsers.add_parser("train-rul", help="RUL 회귀 모델 학습")
    rul_parser.add_argument("--dataset-dir", default="data/phm2010")
    rul_parser.add_argument("--models-dir", default="artifacts/models")
    rul_parser.add_argument("--figures-dir", default="artifacts/figures")
    rul_parser.add_argument("--sampling-rate", type=float, default=25600.0)
    rul_parser.add_argument("--window-seconds", type=float, default=0.1)
    rul_parser.add_argument("--step-seconds", type=float, default=0.05)
    rul_parser.set_defaults(func=train_rul_command)

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

