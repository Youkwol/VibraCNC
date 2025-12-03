from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import messagebox, ttk

import torch
import torch.nn as nn


SENSOR_COLUMNS = ["force_x", "force_y", "force_z", "vibration_x", "vibration_y", "vibration_z", "ae_rms"]
MODEL_PATH = "artifacts/models/best_anomaly_model.pth"
DATA_ROOT = "data/phm2010"
TRAIN_CONDITIONS = ["c1", "c4", "c6"]
SEQ_LEN = 50
HIDDEN_SIZE = 64
STRIDE = 5
DOWNSAMPLE_FACTOR = 2
THRESHOLD_DEFAULT = 0.0674

_GLOBAL_MIN_MAX: dict[str, np.ndarray | None] = {"min": None, "range": None}


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


def load_ai_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqAutoencoder(len(SENSOR_COLUMNS), SEQ_LEN, HIDDEN_SIZE).to(device)
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model, device
    return None, None


def compute_global_min_max(train_conditions, downsample):
    if _GLOBAL_MIN_MAX["min"] is not None and _GLOBAL_MIN_MAX["range"] is not None:
        return _GLOBAL_MIN_MAX["min"], _GLOBAL_MIN_MAX["range"]

    global_min = None
    global_max = None
    data_dir = Path(DATA_ROOT)
    for condition in train_conditions:
        cond_dir = data_dir / condition
        csv_files = sorted(cond_dir.glob("*.csv"))
        for csv in csv_files:
            try:
                df = pd.read_csv(csv, header=None, names=SENSOR_COLUMNS)
                if downsample > 1:
                    df = df.iloc[::downsample]
                val = df.values
                if global_min is None:
                    global_min = val.min(axis=0)
                    global_max = val.max(axis=0)
                else:
                    global_min = np.minimum(global_min, val.min(axis=0))
                    global_max = np.maximum(global_max, val.max(axis=0))
            except Exception as exc:
                print(f"[compute_global_min_max] Failed to read {csv}: {exc}")

    if global_min is None:
        global_min = np.zeros(len(SENSOR_COLUMNS))
        global_range = np.ones(len(SENSOR_COLUMNS))
    else:
        global_range = global_max - global_min
        zero_mask = global_range <= 1e-8
        if np.any(zero_mask):
            global_range[zero_mask] = 1.0

    _GLOBAL_MIN_MAX["min"] = global_min
    _GLOBAL_MIN_MAX["range"] = global_range
    return global_min, global_range


def process_data(condition, model, device):
    data_dir = Path(DATA_ROOT) / condition
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"{condition} 조건 CSV 파일을 찾을 수 없습니다.")

    frames = []
    for path in csv_files:
        try:
            df = pd.read_csv(path, header=None, names=SENSOR_COLUMNS).iloc[::DOWNSAMPLE_FACTOR]
            frames.append(df)
        except Exception as exc:
            raise ValueError(f"{path.name} 로드 실패: {exc}") from exc

    full_df = pd.concat(frames, ignore_index=True)

    g_min, g_range = compute_global_min_max(TRAIN_CONDITIONS, DOWNSAMPLE_FACTOR)
    normalized = (full_df.values - g_min) / g_range

    total = (len(normalized) - SEQ_LEN) // STRIDE + 1
    if total <= 0:
        raise ValueError("시퀀스를 만들 수 있을 만큼 데이터가 부족합니다.")

    sequences = np.zeros((total, SEQ_LEN, len(SENSOR_COLUMNS)), dtype=np.float32)
    for i, idx in enumerate(range(0, len(normalized) - SEQ_LEN + 1, STRIDE)):
        sequences[i] = normalized[idx : idx + SEQ_LEN]

    errors = []
    batch_size = 512
    criterion = nn.L1Loss(reduction="none")

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_np = sequences[i : i + batch_size]
            batch = torch.from_numpy(batch_np).to(
                device,
                dtype=torch.float32,
                non_blocking=device.type == "cuda",
            )
            recon = model(batch)
            loss = criterion(recon, batch).mean(dim=(1, 2)).cpu().numpy()
            errors.extend(loss)

    return np.array(errors, dtype=np.float32)


class CNCMonitorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CNC AI 예지보전 모니터링 시스템")
        self.root.geometry("1200x800")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Danger.TLabel", foreground="red", font=("Helvetica", 16, "bold"))
        style.configure("Normal.TLabel", foreground="green", font=("Helvetica", 16, "bold"))
        style.configure("KPI.TLabel", font=("Helvetica", 12))
        style.configure("Value.TLabel", font=("Helvetica", 14, "bold"))

        self.is_running = False
        self.current_step = 0
        self.error_scores = None
        self.threshold = tk.DoubleVar(value=THRESHOLD_DEFAULT)
        self.speed = tk.IntVar(value=30)

        self.status_var = tk.StringVar(value="모델 로딩 중...")
        self.model, self.device = load_ai_model()
        if not self.model:
            messagebox.showerror("Error", "모델 파일을 찾을 수 없습니다.")
            self.root.destroy()
            return

        self.setup_ui()
        self.status_var.set("준비 완료 (데이터를 선택하세요)")

    def setup_ui(self) -> None:
        control_frame = ttk.LabelFrame(self.root, text="제어 패널", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(control_frame, text="시나리오:").pack(side="left", padx=5)
        self.combo_scenario = ttk.Combobox(
            control_frame, values=["c1", "c4", "c6", "c2", "c3", "c5"], state="readonly", width=5
        )
        self.combo_scenario.current(0)
        self.combo_scenario.pack(side="left", padx=5)
        self.combo_scenario.bind("<<ComboboxSelected>>", self.on_scenario_change)

        ttk.Label(control_frame, text="임계값:").pack(side="left", padx=10)
        ttk.Entry(control_frame, textvariable=self.threshold, width=8).pack(side="left")

        ttk.Button(control_frame, text="▶ 시작", command=self.start_sim).pack(side="left", padx=10)
        ttk.Button(control_frame, text="⏸ 일시정지", command=self.pause_sim).pack(side="left")
        ttk.Button(control_frame, text="⏹ 초기화", command=self.reset_sim).pack(side="left", padx=5)

        kpi_frame = ttk.Frame(self.root, padding=10)
        kpi_frame.pack(fill="x", padx=10)

        f1 = ttk.Frame(kpi_frame, borderwidth=2, relief="groove")
        f1.pack(side="left", expand=True, fill="both", padx=5)
        ttk.Label(f1, text="현재 작업 (Cut)", style="KPI.TLabel").pack()
        self.lbl_cut = ttk.Label(f1, text="-", style="Value.TLabel")
        self.lbl_cut.pack()

        f2 = ttk.Frame(kpi_frame, borderwidth=2, relief="groove")
        f2.pack(side="left", expand=True, fill="both", padx=5)
        ttk.Label(f2, text="이상 점수", style="KPI.TLabel").pack()
        self.lbl_score = ttk.Label(f2, text="-", style="Value.TLabel")
        self.lbl_score.pack()

        f3 = ttk.Frame(kpi_frame, borderwidth=2, relief="groove")
        f3.pack(side="left", expand=True, fill="both", padx=5)
        ttk.Label(f3, text="상태 (Status)", style="KPI.TLabel").pack()
        self.lbl_status = ttk.Label(f3, text="대기", style="Normal.TLabel")
        self.lbl_status.pack()

        f4 = ttk.Frame(kpi_frame, borderwidth=2, relief="groove")
        f4.pack(side="left", expand=True, fill="both", padx=5)
        ttk.Label(f4, text="남은 수명 (RUL)", style="KPI.TLabel").pack()
        self.lbl_rul = ttk.Label(f4, text="-", style="Value.TLabel")
        self.lbl_rul.pack()

        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Real-time Anomaly Score")
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Error")
        self.ax.grid(True, alpha=0.3)

        self.line_score, = self.ax.plot([], [], "b-", label="Score", linewidth=1)
        self.line_thresh = self.ax.axhline(y=self.threshold.get(), color="r", linestyle="--", label="Threshold")
        self.ax.legend(loc="upper left")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        ttk.Label(self.root, textvariable=self.status_var, relief="sunken").pack(side="bottom", fill="x")

        self.on_scenario_change(None)

    def on_scenario_change(self, _event) -> None:
        scenario = self.combo_scenario.get()
        self.status_var.set(f"{scenario} 데이터 로딩 및 분석 중... (잠시만 기다려주세요)")
        self.root.update()

        try:
            self.error_scores = process_data(scenario, self.model, self.device)
        except ValueError as exc:
            messagebox.showerror("Error", str(exc))
            self.status_var.set("데이터 로드 실패")
            return

        self.current_step = 0
        self.is_running = False
        self.update_gui_once()
        self.status_var.set(f"{scenario} 로드 완료. '시작' 버튼을 누르세요.")

    def start_sim(self) -> None:
        if not self.is_running:
            self.is_running = True
            self.run_loop()

    def pause_sim(self) -> None:
        self.is_running = False

    def reset_sim(self) -> None:
        self.is_running = False
        self.current_step = 0
        self.update_gui_once()

    def run_loop(self) -> None:
        if self.is_running and self.error_scores is not None:
            if self.current_step < len(self.error_scores):
                self.update_gui_once()
                self.current_step += 1
                self.root.after(self.speed.get(), self.run_loop)
            else:
                self.is_running = False
                self.status_var.set("시뮬레이션 완료")

    def update_gui_once(self) -> None:
        if self.error_scores is None:
            return

        idx = self.current_step
        score = self.error_scores[idx]
        thresh = self.threshold.get()

        approx_cut = int((idx * STRIDE) / SEQ_LEN) + 1
        self.lbl_cut.config(text=f"#{approx_cut}")
        self.lbl_score.config(text=f"{score:.4f}")

        total = len(self.error_scores)
        rul = int((total - idx) / 50)
        self.lbl_rul.config(text=f"{rul} 회")

        if score > thresh:
            self.lbl_status.config(text="⚠️ 위험", style="Danger.TLabel")
        else:
            self.lbl_status.config(text="✅ 정상", style="Normal.TLabel")

        start = max(0, idx - 200)
        y_data = self.error_scores[start : idx + 1]
        x_data = range(start, idx + 1)

        self.line_score.set_data(x_data, y_data)
        self.line_thresh.set_ydata([thresh] * 2)

        self.ax.set_xlim(start, max(start + 200, idx + 1))
        max_y = max(np.max(y_data) if len(y_data) > 0 else 0.1, thresh)
        self.ax.set_ylim(0, max_y * 1.2)

        self.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    app = CNCMonitorApp(root)
    root.mainloop()

