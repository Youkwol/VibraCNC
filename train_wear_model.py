from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# --- 설정 ---
SENSOR_COLUMNS = ["force_x", "force_y", "force_z", "vibration_x", "vibration_y", "vibration_z", "ae_rms"]
DATA_ROOT = "data/phm2010"
OUTPUT_DIR = "artifacts/results"
MODEL_SAVE_PATH = "artifacts/models/wear_regressor.pth"

# 학습 데이터 (정답지 wear.csv가 있는 폴더들)
TRAIN_CONDITIONS = ["c1", "c4", "c6"]
ALL_CONDITIONS = ["c1", "c4", "c6", "c2", "c3", "c5"]

SEQ_LEN = 50
STRIDE = 10
DOWNSAMPLE_FACTOR = 10
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2  # 검증 데이터 비율
EARLY_STOPPING_PATIENCE = 10  # Early stopping patience
MIN_DELTA = 1e-6  # Early stopping 최소 개선량
GRADIENT_CLIP = 1.0  # Gradient clipping 값

# --- 모델 정의 (CNN-LSTM) ---
class CNNLSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, seq_len: int) -> None:
        super().__init__()
        # 1. CNN (특징 추출)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # CNN 거치면 길이가 줄어듦 (50 -> 25 -> 12)
        cnn_out_len = seq_len // 4

        # 2. LSTM (시계열 회귀)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)

        # 3. Output (마모량 1개 값)
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # 출력: mm 단위 마모량
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Seq, Feat) -> (Batch, Feat, Seq) for CNN
        x = x.permute(0, 2, 1)

        features = self.cnn(x)

        # (Batch, Feat, Seq) -> (Batch, Seq, Feat) for LSTM
        features = features.permute(0, 2, 1)

        _, (hidden, _) = self.lstm(features)

        # Last hidden state
        out = self.regressor(hidden[-1])
        return out.squeeze()


# --- 데이터 로딩 함수 ---
def load_data_with_labels(
    conditions: list[str], is_training: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    시퀀스 데이터와 라벨을 로드합니다.
    
    Returns:
        sequences: (N, seq_len, 7) 시퀀스 배열
        labels: (N,) 마모량 라벨 (cut 내 위치 기반 보간)
        cut_groups: (N,) cut 그룹 ID (데이터 분할용)
    """
    sequences: list[np.ndarray] = []
    labels: list[float] = []
    cut_groups: list[int] = []  # cut 단위 그룹화용

    global_min: np.ndarray | None = None
    global_max: np.ndarray | None = None

    # wear.csv 파일 로드 (루트 디렉토리에 있음)
    wear_df: pd.DataFrame | None = None
    wear_file_path = Path(DATA_ROOT) / "wear.csv"
    if wear_file_path.exists():
        try:
            wear_df = pd.read_csv(wear_file_path)
            print(f"wear.csv 로드 완료: {len(wear_df)} 행")
        except Exception as e:
            print(f"Warning: wear.csv 읽기 실패: {e}")
    else:
        if is_training:
            print(f"Warning: {wear_file_path} 없음")
        wear_df = None

    # 정규화를 위한 Min/Max 계산 (학습 데이터만 사용)
    if is_training:
        print("데이터 로딩 및 전처리 중...")
        all_sensor_data: list[np.ndarray] = []
        for cond in TRAIN_CONDITIONS:
            cond_dir = Path(DATA_ROOT) / cond
            csv_files = sorted([f for f in cond_dir.glob("*.csv") if f.name != "wear.csv"])
            for csv in csv_files:
                try:
                    df = pd.read_csv(csv, header=None, names=SENSOR_COLUMNS).iloc[::DOWNSAMPLE_FACTOR]
                    all_sensor_data.append(df.values)
                except Exception:
                    continue

        if len(all_sensor_data) == 0:
            raise ValueError("학습 데이터를 찾을 수 없습니다.")

        # 메모리 효율적인 Min/Max 계산 (큰 데이터셋 대응)
        global_min = None
        global_max = None
        for data in all_sensor_data:
            if global_min is None:
                global_min = data.min(axis=0)
                global_max = data.max(axis=0)
            else:
                global_min = np.minimum(global_min, data.min(axis=0))
                global_max = np.maximum(global_max, data.max(axis=0))
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        np.save("artifacts/models/wear_scaler_params.npy", np.array([global_min, global_max]))
    else:
        scaler_path = "artifacts/models/wear_scaler_params.npy"
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"{scaler_path} 파일이 없습니다. 먼저 학습을 실행하세요.")
        scaler_data = np.load(scaler_path)
        global_min, global_max = scaler_data[0], scaler_data[1]

    range_val = np.maximum(global_max - global_min, 1e-8)

    # 데이터셋 구축
    group_id = 0  # cut 그룹 ID
    for cond in conditions:
        cond_dir = Path(DATA_ROOT) / cond
        csv_files = sorted([f for f in cond_dir.glob("*.csv") if f.name != "wear.csv"])

        # 해당 조건의 wear 데이터 필터링 및 정렬된 리스트 생성
        wear_list: list[tuple[str, float]] = []
        if wear_df is not None:
            cond_wear = wear_df[wear_df["condition"] == cond].sort_values("cut")
            for _, row in cond_wear.iterrows():
                file_path = row["file"]
                file_name = Path(file_path).name
                wear_um = float(row["wear"])
                wear_mm = wear_um / 1000.0
                wear_list.append((file_name, wear_mm))

        if is_training and len(wear_list) == 0:
            print(f"Warning: {cond}에 대한 wear 데이터 없음, 건너뜀")
            continue

        # wear 딕셔너리 생성 (빠른 조회용)
        wear_dict: dict[str, float] = {name: wear for name, wear in wear_list}

        # 이전 cut의 wear 값 저장 (보간용)
        prev_wear = 0.0

        for csv_idx, csv in enumerate(csv_files):
            try:
                df = pd.read_csv(csv, header=None, names=SENSOR_COLUMNS).iloc[::DOWNSAMPLE_FACTOR]
                norm_data = (df.values - global_min) / range_val

                csv_name = csv.name
                current_wear = wear_dict.get(csv_name, prev_wear) if wear_dict else prev_wear

                # 이전 cut의 wear 값 결정 (시퀀스 생성 전에 설정)
                if csv_idx > 0:
                    # 이전 cut의 wear 값 찾기
                    prev_csv_name = csv_files[csv_idx - 1].name
                    prev_wear = wear_dict.get(prev_csv_name, prev_wear) if wear_dict else prev_wear
                # csv_idx == 0인 경우 prev_wear는 이미 0.0으로 설정됨

                # 시퀀스 생성 및 보간 라벨링
                data_len = len(norm_data)

                for idx in range(0, data_len - SEQ_LEN + 1, STRIDE):
                    seq = norm_data[idx : idx + SEQ_LEN]
                    sequences.append(seq)

                    # cut 내 시퀀스 위치 비율 계산 (0.0 ~ 1.0)
                    # 시퀀스의 중간 지점을 기준으로 위치 계산
                    seq_center = idx + SEQ_LEN // 2
                    position_ratio = seq_center / max(data_len - 1, 1)

                    # 이전 cut과 현재 cut 사이 선형 보간
                    # position_ratio = 0.0이면 prev_wear, 1.0이면 current_wear
                    interpolated_wear = prev_wear + (current_wear - prev_wear) * position_ratio
                    labels.append(interpolated_wear)

                    # cut 그룹 ID 할당 (같은 cut의 모든 시퀀스는 같은 그룹)
                    cut_groups.append(group_id)

                # 다음 cut을 위해 현재 wear를 prev_wear로 저장
                prev_wear = current_wear
                group_id += 1  # 다음 cut으로 그룹 ID 증가

            except Exception as e:
                print(f"Warning: {csv} 처리 실패: {e}")
                continue

    if len(sequences) == 0:
        raise ValueError("시퀀스 데이터를 생성할 수 없습니다.")

    return (
        np.array(sequences, dtype=np.float32),
        np.array(labels, dtype=np.float32),
        np.array(cut_groups, dtype=np.int32),
    )


# --- 메인 실행 ---
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 데이터 로드 (학습용)
    print(">>> 1. 학습 데이터 준비")
    try:
        X_train, y_train, cut_groups = load_data_with_labels(TRAIN_CONDITIONS, is_training=True)
        print(f"학습 데이터: {X_train.shape}, 라벨: {y_train.shape}, Cut 그룹: {len(np.unique(cut_groups))}개")
    except Exception as e:
        print(f"에러: {e}")
        print("wear.csv 파일이 없거나 데이터 형식이 맞지 않습니다.")
        print("대신 더미 데이터로 학습을 진행합니다...")
        # 더미 데이터 생성 (테스트용)
        X_train = np.random.randn(1000, SEQ_LEN, 7).astype(np.float32)
        y_train = np.random.rand(1000).astype(np.float32) * 0.2  # 0~0.2mm 범위
        cut_groups = np.arange(1000)  # 더미 그룹

    # Tensor 변환
    X_tensor = torch.tensor(X_train).to(device)
    y_tensor = torch.tensor(y_train).to(device)

    # Cut 단위로 그룹화하여 검증 데이터셋 분리 (같은 cut의 시퀀스가 섞이지 않도록)
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=42)
    train_idx, val_idx = next(gss.split(X_train, y_train, groups=cut_groups))

    train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
    val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"학습 데이터: {len(train_idx)} (Cut 그룹: {len(np.unique(cut_groups[train_idx]))}개), "
          f"검증 데이터: {len(val_idx)} (Cut 그룹: {len(np.unique(cut_groups[val_idx]))}개)")

    # 2. 모델 학습
    print(f">>> 2. 모델 학습 시작 (Epochs: {EPOCHS})")
    model = CNNLSTMRegressor(input_dim=7, seq_len=SEQ_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        # 학습
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            train_loss += loss.item()

        # 검증
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # 학습률 스케줄러 업데이트
        scheduler.step(val_loss)

        # Early stopping 및 체크포인팅
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"Best Val Loss: {best_val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1} (patience: {EARLY_STOPPING_PATIENCE})")
            break

    # 최고 성능 모델 저장
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f">>> 모델 저장 완료! (Best Val Loss: {best_val_loss:.6f})")

    # 3. 전체 데이터에 대해 마모량 예측 및 저장 (Inference)
    print(">>> 3. 전체 시나리오 마모량 예측 및 저장 (.npy)")
    model.eval()

    # Scaler 정보 로드
    try:
        scaler_data = np.load("artifacts/models/wear_scaler_params.npy")
        global_min, global_max = scaler_data[0], scaler_data[1]
        range_val = np.maximum(global_max - global_min, 1e-8)
    except Exception:
        print("Warning: Scaler 파라미터 없음, 기본값 사용")
        global_min = np.zeros(7)
        global_max = np.ones(7)
        range_val = np.ones(7)

    for cond in ALL_CONDITIONS:
        print(f"Predicting {cond}...", end=" ")
        cond_dir = Path(DATA_ROOT) / cond
        csv_files = sorted([f for f in cond_dir.glob("*.csv") if f.name != "wear.csv"])

        if len(csv_files) == 0:
            print("No CSV files found")
            continue

        all_preds: list[float] = []
        inference_batch_size = 512  # 추론용 배치 크기 (메모리 효율)

        for csv_idx, csv in enumerate(csv_files):
            try:
                df = pd.read_csv(csv, header=None, names=SENSOR_COLUMNS).iloc[::DOWNSAMPLE_FACTOR]
                norm_data = (df.values - global_min) / range_val

                # 시퀀스 만들기
                seqs: list[np.ndarray] = []
                for idx in range(0, len(norm_data) - SEQ_LEN + 1, STRIDE):
                    seqs.append(norm_data[idx : idx + SEQ_LEN])

                if len(seqs) == 0:
                    continue

                # 배치 단위로 추론 (메모리 효율)
                seqs_array = np.array(seqs, dtype=np.float32)
                with torch.no_grad():
                    for batch_start in range(0, len(seqs_array), inference_batch_size):
                        batch_end = min(batch_start + inference_batch_size, len(seqs_array))
                        batch_seqs = seqs_array[batch_start:batch_end]
                        X_batch = torch.tensor(batch_seqs, dtype=torch.float32).to(device)

                        preds = model(X_batch).cpu().numpy()
                        # 음수 방지 (마모량은 음수일 수 없음)
                        preds = np.maximum(preds, 0)
                        all_preds.extend(preds.tolist())

                # 진행 상황 표시 (큰 데이터셋 대응)
                if (csv_idx + 1) % 10 == 0 or csv_idx == len(csv_files) - 1:
                    print(f"[{csv_idx + 1}/{len(csv_files)}]", end=" ")

            except Exception as e:
                print(f"Warning: {csv} 처리 실패: {e}")
                continue

        # 저장: {scenario}_wear.npy
        if len(all_preds) > 0:
            save_path = os.path.join(OUTPUT_DIR, f"{cond}_wear.npy")
            np.save(save_path, np.array(all_preds, dtype=np.float32))
            print(f"Saved to {save_path} (shape: {len(all_preds)})")
        else:
            print("No data")


if __name__ == "__main__":
    main()

