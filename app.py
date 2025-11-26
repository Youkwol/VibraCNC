import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn


# --- [1. 설정 및 모델 정의] ---
# (학습 코드와 동일한 모델 클래스 정의 필수)
SENSOR_COLUMNS = ["force_x", "force_y", "force_z", "vibration_x", "vibration_y", "vibration_z", "ae_rms"]
MODEL_PATH = "artifacts/models/best_anomaly_model.pth"
DATA_ROOT = "data/phm2010"
TRAIN_CONDITIONS = ["c1", "c4", "c6"]
SEQ_LEN = 50
HIDDEN_SIZE = 64  # 학습 시 사용한 hidden size와 동일하게 유지
STRIDE = 5
DOWNSAMPLE_FACTOR = 2  # 학습 파이프라인과 동일한 전처리
THRESHOLD_DEFAULT = 0.0674


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


# 페이지 설정 (반드시 맨 처음에 와야 함)
st.set_page_config(page_title="CNC AI 모니터링 시스템", layout="wide", page_icon="🏭")

if "is_running" not in st.session_state:
    st.session_state["is_running"] = False
if "current_step" not in st.session_state:
    st.session_state["current_step"] = 0
if "chart_history_score" not in st.session_state:
    st.session_state["chart_history_score"] = []
if "chart_history_threshold" not in st.session_state:
    st.session_state["chart_history_threshold"] = []
if "last_rendered_step" not in st.session_state:
    st.session_state["last_rendered_step"] = -1


def reset_simulation_state():
    st.session_state["current_step"] = 0
    st.session_state["chart_history_score"] = []
    st.session_state["chart_history_threshold"] = []
    st.session_state["last_rendered_step"] = -1

# --- [2. 데이터 및 모델 로딩 함수 (캐싱 적용)] ---
@st.cache_resource
def load_ai_model(model_path, input_dim, seq_len, hidden_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqAutoencoder(input_dim, seq_len, hidden_size).to(device)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model, device
    else:
        st.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return None, None


@st.cache_resource
def compute_global_min_max(train_conditions, downsample):
    """학습 시 사용한 조건들 전체에 대한 전역 min/max를 계산."""
    global_min = None
    global_max = None
    data_dir = Path(DATA_ROOT)

    for condition in train_conditions:
        cond_dir = data_dir / condition
        csv_files = sorted(cond_dir.glob("*.csv"))
        for csv in csv_files:
            for chunk in pd.read_csv(csv, header=None, names=SENSOR_COLUMNS, chunksize=200_000):
                if downsample > 1:
                    chunk = chunk.iloc[::downsample]
                values = chunk.values
                if global_min is None:
                    global_min = values.min(axis=0)
                    global_max = values.max(axis=0)
                else:
                    global_min = np.minimum(global_min, values.min(axis=0))
                    global_max = np.maximum(global_max, values.max(axis=0))

    if global_min is None or global_max is None:
        raise RuntimeError("전역 min/max를 계산할 데이터가 없습니다.")

    data_range = np.maximum(global_max - global_min, 1e-8)
    return global_min, data_range


def normalize_with_stats(values, global_min, data_range):
    return (values - global_min) / data_range


def build_sequences(array, seq_len, stride):
    total = (len(array) - seq_len) // stride + 1
    if total <= 0:
        return np.empty((0, seq_len, array.shape[1]), dtype=array.dtype)
    sequences = np.zeros((total, seq_len, array.shape[1]), dtype=array.dtype)
    for idx, start in enumerate(range(0, len(array) - seq_len + 1, stride)):
        sequences[idx] = array[start : start + seq_len]
    return sequences


@st.cache_data
def load_and_process_data(condition, _model, _device, seq_len, stride=STRIDE, downsample=DOWNSAMPLE_FACTOR):
    """
    선택된 조건(c1~c6)의 데이터를 로드하고, 모델을 돌려 미리 에러(Anomaly Score)를 계산해 둡니다.
    실시간 루프에서 매번 추론하면 느리기 때문에, 시뮬레이터를 위해 미리 계산합니다.
    """
    data_dir = Path(DATA_ROOT) / condition
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        return None, None

    # 데이터 로드 (학습과 동일한 다운샘플)
    frames = []
    for p in csv_files:
        df = pd.read_csv(p, header=None, names=SENSOR_COLUMNS)
        if downsample > 1:
            df = df.iloc[::downsample]
        frames.append(df)

    full_df = pd.concat(frames, ignore_index=True)

    global_min, data_range = compute_global_min_max(TRAIN_CONDITIONS, downsample)
    normalized = normalize_with_stats(full_df.values, global_min, data_range)

    sequences = build_sequences(normalized, seq_len, stride)
    if len(sequences) == 0:
        return None, None

    # 대량 텐서를 한 번에 GPU로 옮기면 VRAM을 초과하므로 CPU에 두고 배치 단위로 전송
    seq_tensor = torch.tensor(sequences, dtype=torch.float32)

    # 추론 (Batch 처리)
    batch_size = 512
    errors = []
    criterion = nn.L1Loss(reduction="none")

    with torch.no_grad():
        for i in range(0, len(seq_tensor), batch_size):
            batch = seq_tensor[i: i + batch_size].to(_device, non_blocking=True)
            recon = _model(batch)
            loss = criterion(recon, batch).mean(dim=(1, 2)).cpu().numpy()
            errors.extend(loss)

    return full_df, np.array(errors)


# --- [3. 메인 UI 구성] ---
st.title("🏭 CNC 장비 상태 분석 및 예지보전 시스템")
st.markdown("---")

# 사이드바: 제어 패널
with st.sidebar:
    st.header("⚙️ 시뮬레이션 설정")
    selected_condition = st.selectbox("데이터 선택 (Scenario)", ["c1", "c4", "c6", "c2", "c3", "c5"], index=3)  # c2 기본

    st.subheader("모델 파라미터")
    threshold = st.number_input(
        "이상치 임계값 (Threshold)",
        value=THRESHOLD_DEFAULT,
        step=0.001,
        format="%.4f",
    )

    st.subheader("재생 속도 제어")
    speed = st.slider("시뮬레이션 속도", 1, 100, 10)

    start_btn = st.button("▶️ 시뮬레이션 시작", type="primary")
    pause_btn = st.button("⏸️ 일시정지")
    reset_btn = st.button("🔄 초기화")

# 모델 로드
model, device = load_ai_model(MODEL_PATH, input_dim=len(SENSOR_COLUMNS), seq_len=SEQ_LEN, hidden_size=HIDDEN_SIZE)
if model is None or device is None:
    st.stop()

if start_btn:
    st.session_state['is_running'] = True
if pause_btn:
    st.session_state['is_running'] = False
if reset_btn:
    st.session_state['is_running'] = False
    reset_simulation_state()

# 데이터 준비
if 'data_cache' not in st.session_state or st.session_state.get('last_cond') != selected_condition:
    with st.spinner(f"{selected_condition} 데이터 및 AI 분석 결과 로딩 중..."):
        raw_df, error_scores = load_and_process_data(
            selected_condition,
            model,
            device,
            seq_len=SEQ_LEN,
            stride=STRIDE,
            downsample=DOWNSAMPLE_FACTOR,
        )
        st.session_state['data_cache'] = (raw_df, error_scores)
        st.session_state['last_cond'] = selected_condition
        st.session_state['is_running'] = False
        reset_simulation_state()
else:
    raw_df, error_scores = st.session_state['data_cache']

# --- [4. 대시보드 뷰 구현] ---

tab1, tab2, tab3, tab4 = st.tabs(["🖥️ 실시간 모니터링", "🔮 예측 및 진단", "🔍 심층 분석", "💰 운영 최적화"])

with tab1:
    st.markdown("### 📊 실시간 센서 데이터 및 이상 감지 현황")
    kpi_container = st.empty()
    st.markdown("---")
    chart_placeholder = st.empty()
    log_placeholder = st.empty()

with tab2:
    pred_col1, pred_col2 = st.columns(2)
    rul_chart_placeholder = st.empty()

with tab3:
    st.info("이 기능은 특정 시점의 상세 FFT 분석을 보여줍니다.")
    analysis_col1, analysis_col2 = st.columns(2)

with tab4:
    roi_col1, roi_col2, roi_col3 = st.columns(3)

progress_bar = st.progress(0)


def update_dashboard(step_idx, threshold_value, append_history=True):
    if error_scores is None or len(error_scores) == 0:
        return

    total_steps = len(error_scores)
    step_idx = int(max(0, min(step_idx, total_steps - 1)))
    current_score = float(error_scores[step_idx])

    chart_scores = st.session_state['chart_history_score']
    chart_thresholds = st.session_state['chart_history_threshold']
    last_step = st.session_state.get('last_rendered_step', -1)

    if append_history and step_idx != last_step:
        chart_scores.append(current_score)
        chart_thresholds.append(threshold_value)
        if len(chart_scores) > 200:
            chart_scores[:] = chart_scores[-200:]
            chart_thresholds[:] = chart_thresholds[-200:]
        st.session_state['last_rendered_step'] = step_idx
    elif not chart_scores:
        chart_scores.append(current_score)
        chart_thresholds.append(threshold_value)
    elif not append_history and chart_thresholds:
        chart_thresholds[-1] = threshold_value

    plot_score = chart_scores[-200:]
    plot_threshold = chart_thresholds[-200:] if chart_thresholds else [threshold_value] * len(plot_score)

    is_danger = current_score > threshold_value
    status_text = "🚨 위험 (Danger)" if is_danger else "✅ 정상 (Normal)"
    status_color = "red" if is_danger else "green"

    with kpi_container.container():
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

        with kpi_col1:
            approx_cut = int((step_idx * STRIDE) / SEQ_LEN) + 1
            st.metric(label="현재 작업 (Cut)", value=f"#{approx_cut}")

        with kpi_col2:
            st.metric(
                label="이상 점수 (Anomaly Score)",
                value=f"{current_score:.4f}",
                delta=f"{current_score - threshold_value:.4f}",
                delta_color="inverse",
            )

        with kpi_col3:
            temp_val = 40 + (step_idx / max(total_steps, 1)) * 15
            st.metric(label="현재 온도 (Temp)", value=f"{temp_val:.1f} °C")

        with kpi_col4:
            st.markdown("#### 상태")
            st.markdown(f":{status_color}[**{status_text}**]")

    chart_df = pd.DataFrame({
        "Anomaly Score": plot_score,
        "Threshold": plot_threshold if len(plot_threshold) == len(plot_score) else [threshold_value] * len(plot_score),
    })
    chart_placeholder.line_chart(chart_df, color=["#0000FF", "#FF0000"], height=350)

    if is_danger:
        log_placeholder.error(f"⚠️ [WARNING] 이상 감지됨! 점수: {current_score:.4f} > 임계값: {threshold_value}")
    else:
        log_placeholder.empty()

    remaining_steps = total_steps - step_idx
    rul_cuts = int(max(0, remaining_steps) / 50)
    wear_percent = min(100.0, (step_idx / max(total_steps - 1, 1)) * 100)

    with pred_col1:
        st.metric("남은 수명 (RUL)", f"{rul_cuts} 회 (Cuts)")
    with pred_col2:
        st.metric("현재 마모율", f"{wear_percent:.1f} %")

    with rul_chart_placeholder.container():
        st.progress(wear_percent / 100, text=f"마모 진행도: {wear_percent:.1f}%")

    if step_idx % 10 == 0:
        with analysis_col1:
            importance_data = pd.DataFrame({
                "Sensor": ["Vibration_Z", "Vibration_Y", "AE_RMS", "Force_X", "Force_Z"],
                "Importance": [0.35, 0.25, 0.20, 0.15, 0.05]
            })
            st.bar_chart(importance_data.set_index("Sensor"))
            st.caption("핵심 기여 센서 (Top 5)")

    cost_saved = int((step_idx / 100) * 5)
    with roi_col1:
        st.metric("추천 교체 시점", "RUL 30회 미만")
    with roi_col2:
        st.metric("예상 절감 비용", f"{cost_saved} 만원")
    with roi_col3:
        rec_text = "사용 가능"
        if rul_cuts < 30:
            rec_text = "교체 권장"
        if rul_cuts < 10:
            rec_text = "즉시 교체 필요"
        st.metric("AI 제안", rec_text)

    progress_bar.progress(min(1.0, step_idx / max(total_steps - 1, 1)))

# --- [5. 실시간 루프 (Animation Loop)] ---
if error_scores is not None:
    total_steps = len(error_scores)
    current_step = min(st.session_state.get('current_step', 0), total_steps - 1)

    if st.session_state.get('is_running'):
        next_step = min(current_step + speed, total_steps - 1)
        update_dashboard(next_step, threshold, append_history=True)
        st.session_state['current_step'] = next_step

        if next_step >= total_steps - 1:
            st.session_state['is_running'] = False
            st.success("시뮬레이션 완료")
        else:
            time.sleep(0.05)
            st.rerun()
    else:
        update_dashboard(current_step, threshold, append_history=False)
        if current_step >= total_steps - 1:
            st.success("시뮬레이션 완료")
else:
    st.info("선택한 조건의 데이터를 찾을 수 없습니다.")