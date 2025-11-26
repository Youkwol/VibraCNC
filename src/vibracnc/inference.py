from __future__ import annotations

import argparse
from pathlib import Path
import platform
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
from sklearn.preprocessing import MinMaxScaler

# 학습 코드에 있던 클래스들을 그대로 가져오거나 import 해야 합니다.
# 편의를 위해 여기에 필요한 정의를 포함합니다.
# (실제로는 experiments/lstm_autoencoder_c1.py 에서 import 하는 것이 좋습니다)

SENSOR_COLUMNS = ["force_x", "force_y", "force_z", "vibration_x", "vibration_y", "vibration_z", "ae_rms"]

# --- [재사용되는 클래스 및 함수 정의 시작] ---
# (주의: 학습 코드와 동일한 구조여야 합니다.)

class Seq2SeqAutoencoder(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_size, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.encoder = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_dim)

    def forward(self, x):
        batch_size = x.size(0)
        _, (hidden, cell) = self.encoder(x)
        decoder_input = torch.zeros(batch_size, self.seq_len, self.input_dim, device=x.device, dtype=x.dtype)
        decoded, _ = self.decoder(decoder_input, (hidden, cell))
        return self.output_layer(decoded)

def load_condition_series(dataset_root: Path, condition: str) -> pd.DataFrame:
    data_dir = dataset_root / condition
    csv_files = sorted(data_dir.glob("*.csv"))
    frames = [pd.read_csv(p, header=None, names=SENSOR_COLUMNS) for p in csv_files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def build_sequences(array: np.ndarray, seq_len: int, stride: int = 1) -> np.ndarray:
    n_samples = (len(array) - seq_len) // stride + 1
    if n_samples <= 0:
        raise ValueError("Not enough samples to build sequences.")
    sequences = np.zeros((n_samples, seq_len, array.shape[1]), dtype=array.dtype)
    for i, idx in enumerate(range(0, len(array) - seq_len + 1, stride)):
        sequences[i] = array[idx : idx + seq_len]
    return sequences

class SequenceDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# --- [재사용 정의 끝] ---

def evaluate_condition(model, condition, args, scaler, threshold, amp_enabled, amp_dtype):
    print(f"Evaluating {condition}...")
    # 데이터 로드
    df = load_condition_series(Path(args.dataset_root), condition)
    if df.empty:
        print(f"Skipping {condition} (no data)")
        return
    
    if args.downsample_factor > 1:
        df = df.iloc[::args.downsample_factor].reset_index(drop=True)
    
    normalized = scaler.transform(df.values)
    
    sequences = build_sequences(normalized, args.seq_len, stride=args.stride)
    dataset = SequenceDataset(sequences)
    num_workers = 0 if platform.system() == "Windows" else (os.cpu_count() or 4)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=args.use_cuda,
    )
    
    device = torch.device("cuda" if args.use_cuda else "cpu")
    model.eval()
    criterion = nn.L1Loss(reduction="none")
    errors = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with autocast(
                device_type=device.type,
                enabled=amp_enabled,
                dtype=amp_dtype if amp_enabled else torch.float32,
            ):
                recon = model(batch)
            
            # 배치별 에러 계산
            batch_loss = criterion(recon, batch).mean(dim=(1, 2)).cpu().numpy()
            errors.append(batch_loss)
            
    return np.concatenate(errors)

def fit_scaler(dataset_root: Path, conditions: list[str], downsample_factor: int) -> MinMaxScaler:
    scaler = MinMaxScaler()
    total_rows = 0
    for condition in conditions:
        df = load_condition_series(dataset_root, condition)
        if df.empty:
            continue
        if downsample_factor > 1:
            df = df.iloc[::downsample_factor].reset_index(drop=True)
        scaler.partial_fit(df.values)
        total_rows += len(df)
    if total_rows == 0:
        raise RuntimeError("No data available to fit scaler.")
    return scaler


def main(args):
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    amp_enabled = args.use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    
    # 모델 초기화 (학습 때와 파라미터가 동일해야 함)
    model = Seq2SeqAutoencoder(
        input_dim=len(SENSOR_COLUMNS),
        seq_len=args.seq_len,
        hidden_size=args.hidden_size
    ).to(device)
    
    # 가중치 로드
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded model from epoch {checkpoint['epoch']}")

    # 학습 조건 데이터를 사용해 scaler를 적합 (학습 시와 동일 조건 사용 권장)
    scaler = fit_scaler(Path(args.dataset_root), args.train_conditions, args.downsample_factor)
    
    # 비교할 데이터 목록
    test_conditions = ["c1", "c4", "c6", "c2", "c3", "c5"] 
    
    plt.figure(figsize=(15, 10))
    
    for i, cond in enumerate(test_conditions):
        errors = evaluate_condition(
            model,
            cond,
            args,
            scaler,
            args.threshold,
            amp_enabled,
            amp_dtype,
        )
        
        plt.subplot(2, 3, i+1)
        plt.plot(errors, label='Reconstruction Error', color='blue', alpha=0.7)
        plt.axhline(y=args.threshold, color='red', linestyle='--', label='Threshold')
        
        # 스무딩 (이동 평균) - 트렌드를 더 잘 보기 위함
        smooth_errors = pd.Series(errors).rolling(window=100).mean()
        plt.plot(smooth_errors, color='orange', linewidth=2, label='Moving Avg (100)')
        
        plt.title(f"Condition: {cond}")
        plt.xlabel("Time Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="data/phm2010")
    parser.add_argument("--checkpoint-path", type=str, default="artifacts/models/best_anomaly_model.pth")
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=96) # 학습 때 쓴 값
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--stride", type=int, default=5, help="Stride for sequence generation (match training).")
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=2,
        help="Downsample factor applied to both scaler fitting and evaluation data.",
    )
    parser.add_argument(
        "--train-conditions",
        type=str,
        nargs="+",
        default=["c1", "c4", "c6"],
        help="Conditions used to fit the scaler. Should match training.",
    )
    
    # 중요: 학습 완료 후 터미널에 출력된 '95th percentile threshold' 값을 여기에 넣으세요!
    parser.add_argument("--threshold", type=float, default=0.15) 
    
    args = parser.parse_args()
    main(args)