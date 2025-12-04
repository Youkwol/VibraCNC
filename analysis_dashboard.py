from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from vibracnc.anomaly.autoencoder import load_model
from vibracnc.anomaly.pipeline import score_with_artifacts
from vibracnc.config import DatasetConfig
from vibracnc.data.loader import discover_csv_files, load_csv
from vibracnc.data.preprocessing import WindowingConfig, dataframe_to_windows
from vibracnc.workflows import load_anomaly_artifacts

# 페이지 설정
st.set_page_config(
    page_title="VibraCNC 분석 리포트",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 기본 경로 설정
DEFAULT_MONITORING_PATH = Path("artifacts/monitoring/monitoring_report.json")
DEFAULT_DIAGNOSTICS_PATH = Path("artifacts/monitoring/diagnostics_report.json")
DEFAULT_ANALYSIS_PATH = Path("artifacts/monitoring/analysis_report.json")
DEFAULT_MODELS_DIR = Path("artifacts/models")
DEFAULT_FIGURES_DIR = Path("artifacts/figures")
DEFAULT_DATASET_DIR = Path("data/phm2010")


@st.cache_data(show_spinner=False)
def load_static_condition_data(
    dataset_dir: str,
    models_dir: str,
    condition: str,
    num_cuts: int,
    sampling_rate: float,
    window_seconds: float,
    step_seconds: float,
) -> tuple[list[dict], float]:
    dataset_root = Path(dataset_dir)
    models_path = Path(models_dir)
    dataset_config = DatasetConfig(
        root_dir=dataset_root,
        sampling_rate=sampling_rate,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
    )
    window_config = WindowingConfig.from_seconds(
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        sampling_rate=sampling_rate,
    )

    artifacts = load_anomaly_artifacts(models_path / "anomaly_artifacts.json")
    model = load_model(models_path / "anomaly_autoencoder.pt", artifacts.config)

    csv_files = discover_csv_files(dataset_root / condition)
    if num_cuts is not None:
        csv_files = csv_files[:num_cuts]

    records: list[dict] = []
    for idx, path in enumerate(csv_files, start=1):
        frame = load_csv(path, column_names=dataset_config.resolve_column_names())
        errors = score_with_artifacts(
            model,
            [frame],
            dataset_config.fft_columns,
            window_config,
            artifacts,
        )
        score = float(np.mean(errors))
        vibration_values = frame[list(dataset_config.fft_columns)].to_numpy(dtype=float)
        vibration_rms = float(np.sqrt(np.mean(np.square(vibration_values))))

        records.append(
            {
                "cut": idx,
                "file": path.name,
                "score": score,
                "is_anomaly": bool(score > artifacts.threshold),
                "vibration_rms": vibration_rms,
            }
        )

    return records, float(artifacts.threshold)


@st.cache_data(show_spinner=False)
def compute_fft_spectrum(
    dataset_dir: str,
    condition: str,
    cut_index: int,
    sampling_rate: float,
    window_seconds: float,
    step_seconds: float,
    fft_columns: tuple[str, ...],
) -> tuple[list[float], list[float], str] | None:
    dataset_root = Path(dataset_dir)
    dataset_config = DatasetConfig(
        root_dir=dataset_root,
        sampling_rate=sampling_rate,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
    )
    window_config = WindowingConfig.from_seconds(
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        sampling_rate=sampling_rate,
    )
    files = discover_csv_files(dataset_root / condition)
    if cut_index < 1 or cut_index > len(files):
        return None

    path = files[cut_index - 1]
    frame = load_csv(path, column_names=dataset_config.resolve_column_names())
    frequencies, amplitudes = dataframe_to_windows(frame, fft_columns, window_config)
    spectrum = amplitudes.mean(axis=(0, 2))
    return frequencies.tolist(), spectrum.tolist(), path.name


def render_static_condition_section(
    records: list[dict],
    threshold: float,
    condition: str,
    fft_payload: tuple[list[float], list[float], str] | None,
    selected_cut: int,
    freq_band: tuple[int, int],
):
    st.header("📌 Cut 기반 정적 분석")

    if not records:
        st.info("선택한 조건에서 사용할 수 있는 컷 데이터가 없습니다.")
        return

    df = pd.DataFrame(records)
    selected_cut = min(max(selected_cut, 1), len(df))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 Cut 수", len(df))
    with col2:
        st.metric("이상 탐지 수", int(df["is_anomaly"].sum()))
    with col3:
        ratio = df["is_anomaly"].mean() * 100 if len(df) > 0 else 0.0
        st.metric("이상 비율", f"{ratio:.1f}%")
    with col4:
        st.metric("임계값", f"{threshold:.4f}")

    st.subheader("1) 위험 점수 추이")
    fig_risk = go.Figure()
    fig_risk.add_trace(
        go.Scatter(
            x=df["cut"],
            y=df["score"],
            mode="lines+markers",
            name="위험 점수",
            line=dict(color="#1f77b4"),
        )
    )
    anomaly_df = df[df["is_anomaly"]]
    if not anomaly_df.empty:
        fig_risk.add_trace(
            go.Scatter(
                x=anomaly_df["cut"],
                y=anomaly_df["score"],
                mode="markers",
                name="이상 Cut",
                marker=dict(color="#d62728", size=8),
            )
        )
    fig_risk.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Threshold {threshold:.4f}",
    )
    fig_risk.update_layout(
        xaxis_title="Cut 번호",
        yaxis_title="위험 점수",
        height=350,
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    st.subheader("2) 평균 진동 세기 (RMS)")
    fig_rms = px.line(
        df,
        x="cut",
        y="vibration_rms",
        markers=True,
        labels={"cut": "Cut 번호", "vibration_rms": "진동 RMS"},
        title="평균 진동 세기 추이",
    )
    fig_rms.update_layout(height=300)
    st.plotly_chart(fig_rms, use_container_width=True)

    st.subheader("3) 고장 원인 주파수 분석")
    if fft_payload is None:
        st.info("선택한 Cut에 대한 FFT 데이터를 가져올 수 없습니다.")
        return

    frequencies, spectrum, file_name = fft_payload
    freq_min, freq_max = freq_band
    highlight = [
        freq_min <= freq <= freq_max for freq in frequencies
    ]
    colors = ["#ef553b" if flag else "#636efa" for flag in highlight]

    fig_fft = go.Figure(
        go.Bar(
            x=frequencies,
            y=spectrum,
            marker_color=colors,
            name="Amplitude",
        )
    )
    fig_fft.update_layout(
        title=f"FFT 스펙트럼 - Cut {selected_cut} ({file_name})",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude",
        height=400,
    )
    st.plotly_chart(fig_fft, use_container_width=True)


def load_json(path: Path) -> dict | None:
    """JSON 파일을 로드합니다."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"파일 로드 오류: {e}")
        return None


def render_anomaly_detection_section(monitoring_data: dict, models_dir: Path):
    """이상 탐지 분석 섹션"""
    st.header("🔍 이상 탐지 분석")
    
    st.subheader("1. 이상 탐지 모델 개요")
    
    # 모델 메타데이터 로드
    metadata_path = models_dir / "anomaly_artifacts.json"
    if metadata_path.exists():
        metadata = load_json(metadata_path)
        if metadata:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("임계값", f"{metadata.get('threshold', 0):.6f}")
            with col2:
                config = metadata.get("config", {})
                st.metric("입력 차원", config.get("input_dim", "N/A"))
            with col3:
                st.metric("은닉 차원", config.get("hidden_dim", "N/A"))
            
            # 학습 히스토리
            train_history = metadata.get("train_history", {})
            if train_history:
                st.subheader("2. 모델 학습 과정")
                epochs = range(1, len(train_history.get("train_loss", [])) + 1)
                df_history = pd.DataFrame({
                    "epoch": epochs,
                    "train_loss": train_history.get("train_loss", []),
                    "val_loss": train_history.get("val_loss", []),
                })
                fig_history = go.Figure()
                fig_history.add_trace(
                    go.Scatter(x=df_history["epoch"], y=df_history["train_loss"], 
                             mode="lines", name="Train Loss", line=dict(color="blue"))
                )
                if df_history["val_loss"].notna().any():
                    fig_history.add_trace(
                        go.Scatter(x=df_history["epoch"], y=df_history["val_loss"], 
                                 mode="lines", name="Validation Loss", line=dict(color="red"))
                    )
                fig_history.update_layout(
                    title="학습 손실 추이",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    height=400,
                )
                st.plotly_chart(fig_history, width="stretch")
    
    # 현재 상태 분석
    current = monitoring_data.get("current_state", {})
    series = monitoring_data.get("series", {})
    anomaly_scores = series.get("anomaly_scores", [])
    
    if anomaly_scores:
        st.subheader("3. 이상 탐지 결과 분석")
        
        df_anomaly = pd.DataFrame(anomaly_scores)
        threshold = current.get("threshold", 0.0)
        
        # 통계 요약
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_cuts = len(df_anomaly)
            st.metric("총 Cut 수", total_cuts)
        with col2:
            anomaly_count = df_anomaly["is_anomaly"].sum()
            st.metric("이상 탐지 수", anomaly_count)
        with col3:
            anomaly_ratio = (anomaly_count / total_cuts * 100) if total_cuts > 0 else 0
            st.metric("이상 비율", f"{anomaly_ratio:.1f}%")
        with col4:
            avg_score = df_anomaly["score"].mean()
            st.metric("평균 이상 점수", f"{avg_score:.6f}")
        
        # 이상 비율이 너무 높을 경우 경고
        if anomaly_ratio >= 90:
            st.error(
                f"🚨 **심각한 문제 발견:** 이상 탐지 비율이 {anomaly_ratio:.1f}%입니다!\n\n"
                "**원인 분석:**\n\n"
                "1. **학습 데이터**: 각 조건(c1, c4, c6)의 **처음 30개 cut**만 사용 (초기 정상 상태)\n"
                "2. **테스트 데이터**: 각 조건의 **마지막 50개 cut** 사용 (마모 진행된 후반부)\n"
                "3. **결과**: 마모 진행된 데이터는 정상 패턴과 다르므로 모두 이상으로 판단됨\n\n"
                "**이것은 정상적인 동작입니다!** 마모가 진행된 데이터는 실제로 정상 상태가 아니므로 "
                "이상으로 탐지되는 것이 맞습니다. 다만, 이는 **고장 예측**의 목적이지 "
                "**초기 이상 탐지**의 목적과는 다릅니다.\n\n"
                "**해결 방법:**\n"
                "- 초기 이상 탐지가 목적이라면: 테스트 데이터도 초기 정상 데이터만 사용\n"
                "- 마모 진행 모니터링이 목적이라면: 현재 결과가 정상 (마모 진행 = 이상 상태)\n"
                "- 더 정확한 분석을 원한다면: 학습 데이터에 다양한 상태(초기+중간+후반) 포함"
            )
        elif anomaly_ratio <= 10:
            st.info(
                f"ℹ️ 이상 탐지 비율이 {anomaly_ratio:.1f}%로 매우 낮습니다. "
                "임계값이 너무 높거나 모델이 너무 보수적일 수 있습니다."
            )
        
        # 조건별 분석
        st.subheader("4. 조건별 이상 탐지 통계")
        condition_stats = df_anomaly.groupby("condition").agg({
            "score": ["mean", "std", "min", "max"],
            "is_anomaly": "sum",
            "cut": "count"
        }).round(6)
        condition_stats.columns = ["평균 점수", "표준편차", "최소값", "최대값", "이상 수", "총 Cut 수"]
        st.dataframe(condition_stats, width="stretch")
        
        # 이상 점수 분포
        st.subheader("5. 이상 점수 분포")
        fig_dist = px.histogram(
            df_anomaly,
            x="score",
            nbins=30,
            title="이상 점수 히스토그램",
            labels={"score": "이상 점수", "count": "빈도"},
        )
        fig_dist.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"임계값: {threshold:.6f}",
        )
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, width="stretch")
        
        # 조건별 비교
        st.subheader("6. 조건별 이상 점수 비교")
        fig_box = px.box(
            df_anomaly,
            x="condition",
            y="score",
            title="조건별 이상 점수 분포",
            labels={"condition": "조건", "score": "이상 점수"},
        )
        fig_box.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"임계값: {threshold:.6f}",
        )
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, width="stretch")


def render_rul_prediction_section(diagnostics_data: dict, analysis_data: dict, models_dir: Path, figures_dir: Path):
    """고장 예측 분석 섹션"""
    st.header("🔮 고장 예측 (RUL) 분석")
    
    st.subheader("1. RUL 예측 모델 개요")
    
    rul_data = diagnostics_data.get("rul", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("예측 남은 수명", f"{rul_data.get('cuts', 0):.1f} 컷")
    with col2:
        st.metric("예측 범위 (최소)", f"{rul_data.get('min_cuts', 0):.1f} 컷")
    with col3:
        st.metric("예측 범위 (최대)", f"{rul_data.get('max_cuts', 0):.1f} 컷")
    
    # 특성 중요도
    feature_importance = analysis_data.get("feature_importance", [])
    if feature_importance:
        st.subheader("2. RUL 예측에 사용된 주요 특성")
        st.markdown(
            "고장 예측 모델은 다음 특성들을 사용하여 남은 수명을 예측합니다. "
            "특성 중요도가 높을수록 예측에 더 큰 영향을 미칩니다."
        )
        df_feat = pd.DataFrame(feature_importance)
        fig_feat = px.bar(
            df_feat,
            x="importance",
            y="feature",
            orientation="h",
            title="특성 중요도 Top 5",
            labels={"importance": "중요도", "feature": "특성"},
            color="importance",
            color_continuous_scale="Blues",
        )
        fig_feat.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_feat, width="stretch")
        
        # 특성 설명
        st.markdown("**주요 특성 설명:**")
        feature_descriptions = {
            "vz_rms": "Z축 진동의 RMS (Root Mean Square) 값 - 진동 에너지의 크기",
            "vz_std": "Z축 진동의 표준편차 - 진동 변동성",
            "sy_rms": "Y축 힘의 RMS 값 - 가공력의 크기",
            "sy_std": "Y축 힘의 표준편차 - 가공력 변동성",
            "sx_max": "X축 힘의 최대값 - 최대 가공력",
        }
        for feat in df_feat["feature"].head(5):
            desc = feature_descriptions.get(feat, "센서 데이터 통계 특성")
            st.markdown(f"- **{feat}**: {desc}")
    
    # 마모 분석
    wear_data = diagnostics_data.get("wear", {})
    if wear_data:
        st.subheader("3. 마모 진행 분석")
        st.markdown(
            "마모 데이터를 기반으로 공구의 상태를 분석하고, "
            "예측 모델이 이를 활용하여 남은 수명을 계산합니다."
        )
        
        actual_series = wear_data.get("actual_series", [])
        predicted_series = wear_data.get("predicted_series", [])
        
        col1, col2 = st.columns(2)
        with col1:
            current_wear = wear_data.get("current", 0.0)
            max_limit = wear_data.get("max_limit", 200.0)
            ratio = wear_data.get("ratio_percent", 0.0)
            st.metric("현재 마모량", f"{current_wear:.2f}")
            st.metric("마모 한계", f"{max_limit:.1f}")
            st.metric("마모율", f"{ratio:.1f}%")
        
        with col2:
            st.info(
                f"**분석 결과:**\n\n"
                f"- 현재 마모량은 한계의 {ratio:.1f}%에 도달했습니다.\n"
                f"- 예측된 남은 수명: {rul_data.get('cuts', 0):.1f} 컷\n"
                f"- 예상 고장 시각: {rul_data.get('predicted_failure_time', 'N/A')[:19] if rul_data.get('predicted_failure_time') else 'N/A'}"
            )
        
        # 마모 시계열
        if actual_series or predicted_series:
            fig_wear = go.Figure()
            if actual_series:
                df_actual = pd.DataFrame(actual_series)
                wear_col = "wear" if "wear" in df_actual.columns else df_actual.columns[1] if len(df_actual.columns) > 1 else None
                if wear_col:
                    fig_wear.add_trace(
                        go.Scatter(
                            x=df_actual["cut"],
                            y=df_actual[wear_col],
                            mode="lines+markers",
                            name="실제 마모",
                            line=dict(color="blue", width=2),
                        )
                    )
            if predicted_series:
                df_pred = pd.DataFrame(predicted_series)
                wear_col = "wear" if "wear" in df_pred.columns else ("prediction" if "prediction" in df_pred.columns else df_pred.columns[1] if len(df_pred.columns) > 1 else None)
                if wear_col:
                    fig_wear.add_trace(
                        go.Scatter(
                            x=df_pred["cut"],
                            y=df_pred[wear_col],
                            mode="lines+markers",
                            name="예측 마모",
                            line=dict(color="red", dash="dash", width=2),
                        )
                    )
            if max_limit:
                fig_wear.add_hline(
                    y=max_limit,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"한계: {max_limit:.1f}",
                )
            fig_wear.update_layout(
                title="마모 진행 추이 (실제 vs 예측)",
                xaxis_title="Cut 번호",
                yaxis_title="마모량",
                height=500,
            )
            st.plotly_chart(fig_wear, width="stretch")
    
    # 모델 성능 지표
    metrics_path = figures_dir / "rul_metrics.csv"
    if metrics_path.exists():
        st.subheader("4. 모델 성능 지표")
        df_metrics = pd.read_csv(metrics_path)
        st.dataframe(df_metrics, width="stretch", hide_index=True)


def render_correlation_analysis(analysis_data: dict):
    """상관관계 분석 섹션"""
    st.header("🔗 센서 상관관계 분석")
    
    correlation_matrix = analysis_data.get("correlation_matrix", [])
    if correlation_matrix:
        st.markdown(
            "센서 간 상관관계를 분석하여 어떤 센서들이 함께 변화하는지 확인합니다. "
            "이는 이상 탐지와 고장 예측에 중요한 정보를 제공합니다."
        )
        
        if isinstance(correlation_matrix, list) and len(correlation_matrix) > 0:
            if isinstance(correlation_matrix[0], dict):
                df_corr = pd.DataFrame(correlation_matrix)
            else:
                df_corr = pd.DataFrame(correlation_matrix)
            
            if not df_corr.empty:
                fig_corr = px.imshow(
                    df_corr,
                    title="센서 상관 행렬",
                    aspect="auto",
                    color_continuous_scale="RdBu",
                    labels=dict(color="상관계수"),
                )
                fig_corr.update_layout(height=600)
                st.plotly_chart(fig_corr, width="stretch")
    
    comparison_table = analysis_data.get("comparison_table", [])
    if comparison_table:
        st.subheader("정상 vs 위험 구간 비교")
        st.markdown("정상 구간과 위험 구간의 센서 특성을 비교하여 이상 패턴을 파악합니다.")
        df_comp = pd.DataFrame(comparison_table)
        st.dataframe(df_comp, width="stretch")


def render_summary_section(monitoring_data: dict, diagnostics_data: dict):
    """종합 요약 섹션"""
    st.header("📋 종합 분석 요약")
    
    current = monitoring_data.get("current_state", {})
    rul_data = diagnostics_data.get("rul", {})
    wear_data = diagnostics_data.get("wear", {})
    
    st.subheader("현재 상태")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**이상 탐지 결과**")
        danger_level = current.get("danger_level", "unknown")
        st.metric("위험 수준", danger_level)
        st.metric("이상 점수", f"{current.get('current_anomaly_score', 0):.6f}")
    with col2:
        st.markdown("**고장 예측 결과**")
        st.metric("남은 수명", f"{rul_data.get('cuts', 0):.1f} 컷")
        st.metric("예상 고장 시각", rul_data.get("predicted_failure_time", "N/A")[:19] if rul_data.get("predicted_failure_time") else "N/A")
    with col3:
        st.markdown("**마모 상태**")
        st.metric("현재 마모율", f"{wear_data.get('ratio_percent', 0):.1f}%")
        st.metric("마모 한계", f"{wear_data.get('max_limit', 0):.1f}")
    
    st.divider()
    
    st.subheader("분석 방법론")
    st.markdown("""
    ### 이상 탐지 방법
    1. **LSTM Autoencoder 모델**을 사용하여 정상 상태의 진동 패턴을 학습
    2. 새로운 데이터의 **재구성 오차**를 계산하여 이상 여부 판단
    3. 임계값을 초과하는 경우 이상으로 분류
    
    ### 고장 예측 방법
    1. 센서 데이터에서 **통계적 특성** 추출 (RMS, 표준편차, 최대값 등)
    2. **Random Forest 회귀 모델**을 사용하여 마모량과 남은 수명 예측
    3. **교차 검증**을 통해 모델 성능 평가
    
    ### 데이터 활용
    - **진동 센서 (vx, vy, vz)**: 이상 탐지의 주요 입력
    - **힘 센서 (sx, sy, sz)**: 고장 예측의 보조 특성
    - **마모 데이터**: 예측 모델의 학습 및 검증에 사용
    """)


def main():
    st.title("📈 VibraCNC 분석 리포트")
    st.markdown("이상 탐지와 고장 예측 분석 결과를 종합적으로 보여줍니다.")
    st.markdown("---")
    
    # 사이드바: 파일 경로 설정
    with st.sidebar:
        st.header("설정")
        monitoring_path = Path(
            st.text_input(
                "모니터링 리포트 경로",
                value=str(DEFAULT_MONITORING_PATH),
            )
        )
        diagnostics_path = Path(
            st.text_input(
                "진단 리포트 경로",
                value=str(DEFAULT_DIAGNOSTICS_PATH),
            )
        )
        analysis_path = Path(
            st.text_input(
                "분석 리포트 경로",
                value=str(DEFAULT_ANALYSIS_PATH),
            )
        )
        models_dir = Path(
            st.text_input(
                "모델 디렉터리",
                value=str(DEFAULT_MODELS_DIR),
            )
        )
        figures_dir = Path(
            st.text_input(
                "결과 디렉터리",
                value=str(DEFAULT_FIGURES_DIR),
            )
        )
        st.subheader("정적 분석 설정")
        dataset_dir = Path(
            st.text_input(
                "데이터셋 경로",
                value=str(DEFAULT_DATASET_DIR),
            )
        )
        static_condition = st.selectbox(
            "분석 조건",
            options=["c1", "c2", "c3", "c4", "c5", "c6"],
            index=1,
        )
        num_cuts = st.slider("분석할 Cut 수", min_value=10, max_value=200, value=60, step=5)
        sampling_rate = st.number_input("샘플링 레이트 (Hz)", value=25600.0, step=100.0)
        window_seconds = st.number_input("윈도우 길이 (초)", value=0.1, step=0.01, format="%.3f")
        step_seconds = st.number_input("윈도우 스텝 (초)", value=0.05, step=0.01, format="%.3f")
        selected_cut = st.number_input(
            "FFT 분석 Cut 번호",
            min_value=1,
            max_value=num_cuts,
            value=num_cuts,
            step=1,
        )
        freq_band = st.slider(
            "강조할 주파수 대역 (Hz)",
            min_value=0,
            max_value=5000,
            value=(100, 300),
            step=50,
        )
    
    # 정적 분석 데이터 준비
    try:
        static_records, static_threshold = load_static_condition_data(
            str(dataset_dir),
            str(models_dir),
            static_condition,
            num_cuts,
            sampling_rate,
            window_seconds,
            step_seconds,
        )
    except FileNotFoundError as exc:
        st.error(f"정적 분석 데이터를 불러올 수 없습니다: {exc}")
        static_records, static_threshold = [], 0.0

    fft_payload = None
    if static_records:
        fft_payload = compute_fft_spectrum(
            str(dataset_dir),
            static_condition,
            int(selected_cut),
            sampling_rate,
            window_seconds,
            step_seconds,
            ("vx", "vy", "vz"),
        )

    render_static_condition_section(
        records=static_records,
        threshold=static_threshold,
        condition=static_condition,
        fft_payload=fft_payload,
        selected_cut=int(selected_cut),
        freq_band=freq_band,
    )

    st.divider()

    # 데이터 로드
    monitoring_data = load_json(monitoring_path)
    diagnostics_data = load_json(diagnostics_path)
    analysis_data = load_json(analysis_path)
    
    if not monitoring_data:
        st.error(f"모니터링 리포트를 찾을 수 없습니다: {monitoring_path}")
        return
    
    if not diagnostics_data:
        st.error(f"진단 리포트를 찾을 수 없습니다: {diagnostics_path}")
        return
    
    if not analysis_data:
        st.error(f"분석 리포트를 찾을 수 없습니다: {analysis_path}")
        return
    
    # 섹션 렌더링
    render_summary_section(monitoring_data, diagnostics_data)
    st.divider()
    render_anomaly_detection_section(monitoring_data, models_dir)
    st.divider()
    render_rul_prediction_section(diagnostics_data, analysis_data, models_dir, figures_dir)
    st.divider()
    render_correlation_analysis(analysis_data)


if __name__ == "__main__":
    main()

