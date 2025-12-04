from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


SENSOR_COLUMNS = ["force_x", "force_y", "force_z", "vibration_x", "vibration_y", "vibration_z", "ae_rms"]
DEFAULT_CONDITIONS = ["c1", "c4", "c6", "c2", "c3", "c5"]


class Seq2SeqAutoencoder(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden_size: int, num_layers: int = 2) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.encoder = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        _, (hidden, cell) = self.encoder(x)
        decoder_input = torch.zeros(batch_size, self.seq_len, self.input_dim, device=x.device, dtype=x.dtype)
        decoded, _ = self.decoder(decoder_input, (hidden, cell))
        return self.output_layer(decoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="사전 계산된 이상 점수 생성기")
    parser.add_argument("--dataset-root", default="data/phm2010", help="PHM 2010 데이터셋 루트 경로")
    parser.add_argument("--model-path", default="artifacts/models/best_anomaly_model.pth", help="학습된 모델 체크포인트")
    parser.add_argument("--output-dir", default="artifacts/results", help="이상 점수 .npy 저장 경로")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS, help="처리할 조건 목록")
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--downsample-factor", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024, help="추론 배치 크기")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 장치를 찾을 수 없습니다. --device cpu 로 다시 실행하세요.")
    return torch.device(name)


def iter_condition_frames(condition_dir: Path, downsample: int) -> Iterable[pd.DataFrame]:
    csv_files = sorted(condition_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"{condition_dir} 에 CSV 파일이 없습니다.")
    for csv in csv_files:
        df = pd.read_csv(csv, header=None, names=SENSOR_COLUMNS)
        yield df.iloc[::downsample]


def compute_global_min_max(dataset_root: Path, conditions: list[str], downsample: int) -> tuple[np.ndarray, np.ndarray]:
    print("[1/4] 정규화 기준 계산 중...")
    global_min = None
    global_max = None
    for condition in conditions:
        cond_dir = dataset_root / condition
        for frame in iter_condition_frames(cond_dir, downsample):
            values = frame.values
            if global_min is None:
                global_min = values.min(axis=0)
                global_max = values.max(axis=0)
            else:
                global_min = np.minimum(global_min, values.min(axis=0))
                global_max = np.maximum(global_max, values.max(axis=0))

    if global_min is None:
        raise RuntimeError("정규화 기준을 계산할 수 없습니다. 조건을 확인하세요.")

    global_range = global_max - global_min
    zero_mask = global_range <= 1e-8
    if np.any(zero_mask):
        global_range[zero_mask] = 1.0
    return global_min, global_range


def build_sequences(array: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    total = (len(array) - seq_len) // stride + 1
    if total <= 0:
        raise ValueError("시퀀스를 만들 수 있을 만큼 데이터가 부족합니다.")
    sequences = np.zeros((total, seq_len, array.shape[1]), dtype=np.float32)
    for i, idx in enumerate(range(0, len(array) - seq_len + 1, stride)):
        sequences[i] = array[idx : idx + seq_len]
    return sequences


def score_sequences(
    model: nn.Module,
    device: torch.device,
    sequences: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    criterion = nn.L1Loss(reduction="none")
    feature_errors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_np = sequences[start : start + batch_size]
            batch = torch.from_numpy(batch_np).to(device, dtype=torch.float32, non_blocking=device.type == "cuda")
            recon = model(batch)
            loss = criterion(recon, batch).mean(dim=1).cpu().numpy()  # (batch, sensors)
            feature_errors.append(loss)
    feature_array = np.concatenate(feature_errors, axis=0)
    total_errors = feature_array.mean(axis=1)
    return total_errors, feature_array


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"[0/4] 모델 로드 (device={device})")

    model = Seq2SeqAutoencoder(len(SENSOR_COLUMNS), args.seq_len, args.hidden_size).to(device)
    checkpoint_path = Path(args.model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"모델 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    global_min, global_range = compute_global_min_max(dataset_root, args.conditions[:3], args.downsample_factor)

    for idx, condition in enumerate(args.conditions, start=1):
        print(f"[{idx+1}/4] {condition} 처리 중...", end=" ", flush=True)
        cond_dir = dataset_root / condition
        frames = list(iter_condition_frames(cond_dir, args.downsample_factor))
        
        # 각 cut 파일의 경계 정보 계산 (각 cut 파일이 몇 개의 step을 생성하는지)
        cut_boundaries = []  # 각 cut 파일의 시작 step 인덱스
        current_step = 0
        
        for frame in frames:
            # 각 cut 파일의 다운샘플링된 길이
            downsampled_len = len(frame)
            # 이 cut 파일에서 생성되는 시퀀스 수
            num_sequences = (downsampled_len - args.seq_len) // args.stride + 1
            if num_sequences > 0:
                cut_boundaries.append(current_step)
                current_step += num_sequences
        
        full_df = pd.concat(frames, ignore_index=True)
        normalized = (full_df.values - global_min) / global_range
        sequences = build_sequences(normalized, args.seq_len, args.stride)
        total_errors, feature_errors = score_sequences(model, device, sequences, args.batch_size)
        score_path = output_dir / f"{condition}.npy"
        feature_path = output_dir / f"{condition}_features.npy"
        boundaries_path = output_dir / f"{condition}_cut_boundaries.json"
        
        np.save(score_path, total_errors.astype(np.float32))
        np.save(feature_path, feature_errors.astype(np.float32))
        
        # cut 경계 정보 저장
        with open(boundaries_path, "w") as f:
            json.dump({"cut_boundaries": cut_boundaries, "total_cuts": len(cut_boundaries)}, f)
        
        print(
            f"저장 완료 (steps={len(total_errors)}, cuts={len(cut_boundaries)}, score={score_path}, features={feature_path}, boundaries={boundaries_path})"
        )

    print("✅ 모든 조건 처리 완료")


if __name__ == "__main__":
    main()

