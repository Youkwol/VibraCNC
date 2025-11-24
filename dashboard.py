from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="VibraCNC 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 기본 경로 설정
DEFAULT_MONITORING_PATH = Path("artifacts/monitoring/monitoring_report.json")
DEFAULT_DIAGNOSTICS_PATH = Path("artifacts/monitoring/diagnostics_report.json")
DEFAULT_ANALYSIS_PATH = Path("artifacts/monitoring/analysis_report.json")


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


def render_monitoring_view(data: dict):
    """실시간 모니터링 뷰"""
    st.header("📊 실시간 모니터링")

    current = data.get("current_state", {})
    series = data.get("series", {})
    fft_snapshot = data.get("fft_snapshot")

    # 현재 상태 카드
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("현재 Cut", current.get("current_cut", "N/A"))
    with col2:
        st.metric("조건", current.get("current_condition", "N/A"))
    with col3:
        danger_level = current.get("danger_level", "unknown")
        danger_color = {"정상": "🟢", "주의": "🟡", "위험": "🔴"}.get(danger_level, "⚪")
        st.metric("위험 수준", f"{danger_color} {danger_level}")
    with col4:
        score = current.get("current_anomaly_score", 0.0)
        threshold = current.get("threshold", 0.0)
        st.metric("이상 점수", f"{score:.4f}", delta=f"임계값: {threshold:.4f}")

    st.divider()

    # 이상 점수 시계열 그래프
    anomaly_scores = series.get("anomaly_scores", [])
    if anomaly_scores:
        df_anomaly = pd.DataFrame(anomaly_scores)
        fig_anomaly = go.Figure()
        fig_anomaly.add_trace(
            go.Scatter(
                x=df_anomaly["cut"],
                y=df_anomaly["score"],
                mode="lines+markers",
                name="이상 점수",
                line=dict(color="blue"),
            )
        )
        threshold = current.get("threshold", 0.0)
        fig_anomaly.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"임계값: {threshold:.4f}",
        )
        fig_anomaly.update_layout(
            title="이상 점수 시계열",
            xaxis_title="Cut 번호",
            yaxis_title="이상 점수",
            height=400,
        )
        st.plotly_chart(fig_anomaly, width="stretch")

    # 진동 RMS 시계열 그래프
    vibration_series = series.get("vibration_rms", [])
    if vibration_series:
        df_vib = pd.DataFrame(vibration_series)
        fig_vib = px.line(
            df_vib,
            x="cut",
            y="vibration_rms",
            title="진동 RMS 시계열",
            labels={"cut": "Cut 번호", "vibration_rms": "RMS 진동"},
        )
        fig_vib.update_layout(height=400)
        st.plotly_chart(fig_vib, width="stretch")

    # FFT 스냅샷
    if fft_snapshot:
        st.subheader("FFT 스냅샷")
        freq_bins = fft_snapshot.get("freq_bins", [])
        amplitude = fft_snapshot.get("amplitude", [])
        if freq_bins and amplitude:
            fig_fft = px.bar(
                x=freq_bins[:50],  # 처음 50개만 표시
                y=amplitude[:50],
                title="주파수 스펙트럼",
                labels={"x": "주파수 (Hz)", "y": "진폭"},
            )
            fig_fft.update_layout(height=400)
            st.plotly_chart(fig_fft, width="stretch")


def render_diagnostics_view(data: dict):
    """예측 및 진단 뷰"""
    st.header("🔮 예측 및 진단")

    rul_data = data.get("rul", {})
    wear_data = data.get("wear", {})

    # RUL 정보 카드
    col1, col2, col3 = st.columns(3)
    with col1:
        rul_cuts = rul_data.get("cuts", 0.0)
        st.metric("남은 수명 (컷)", f"{rul_cuts:.1f}")
    with col2:
        rul_min = rul_data.get("min_cuts", 0.0)
        rul_max = rul_data.get("max_cuts", 0.0)
        st.metric("예측 범위", f"{rul_min:.1f} ~ {rul_max:.1f}")
    with col3:
        failure_time = rul_data.get("predicted_failure_time")
        if failure_time:
            st.metric("예상 고장 시각", failure_time[:19])  # 날짜만 표시
        else:
            st.metric("예상 고장 시각", "N/A")

    st.divider()

    # 마모 정보
    col1, col2 = st.columns(2)
    with col1:
        current_wear = wear_data.get("current", 0.0)
        max_limit = wear_data.get("max_limit", 200.0)
        ratio = wear_data.get("ratio_percent", 0.0)
        st.metric("현재 마모량", f"{current_wear:.2f}", delta=f"한계: {max_limit:.1f}")
    with col2:
        st.metric("마모율", f"{ratio:.1f}%")

    # 마모 시계열 그래프
    actual_series = wear_data.get("actual_series", [])
    predicted_series = wear_data.get("predicted_series", [])
    if actual_series or predicted_series:
        fig_wear = go.Figure()
        if actual_series:
            df_actual = pd.DataFrame(actual_series)
            fig_wear.add_trace(
                go.Scatter(
                    x=df_actual["cut"],
                    y=df_actual["wear"],
                    mode="lines+markers",
                    name="실제 마모",
                    line=dict(color="blue"),
                )
            )
        if predicted_series:
            df_pred = pd.DataFrame(predicted_series)
            pred_value_col = (
                "wear"
                if "wear" in df_pred.columns
                else ("prediction" if "prediction" in df_pred.columns else None)
            )
            if pred_value_col is not None:
                fig_wear.add_trace(
                    go.Scatter(
                        x=df_pred["cut"],
                        y=df_pred[pred_value_col],
                        mode="lines+markers",
                        name="예측 마모",
                        line=dict(color="red", dash="dash"),
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
            title="마모 시계열 (실제 vs 예측)",
            xaxis_title="Cut 번호",
            yaxis_title="마모량",
            height=500,
        )
        st.plotly_chart(fig_wear, width="stretch")


def render_analysis_view(data: dict):
    """심층 분석 및 진단 뷰"""
    st.header("🔍 심층 분석 및 진단")

    # Feature Importance Top 5
    feature_importance = data.get("feature_importance", [])
    if feature_importance:
        st.subheader("Feature Importance Top 5")
        df_feat = pd.DataFrame(feature_importance)
        fig_feat = px.bar(
            df_feat,
            x="feature",
            y="importance",
            title="특성 중요도",
            labels={"feature": "특성", "importance": "중요도"},
        )
        fig_feat.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_feat, width="stretch")

    # 상관 행렬
    correlation_matrix = data.get("correlation_matrix", [])
    if correlation_matrix:
        st.subheader("센서 상관 행렬")
        # correlation_matrix가 리스트 형태라면 DataFrame으로 변환
        if isinstance(correlation_matrix, list) and len(correlation_matrix) > 0:
            if isinstance(correlation_matrix[0], dict):
                df_corr = pd.DataFrame(correlation_matrix)
            else:
                df_corr = pd.DataFrame(correlation_matrix)
            if not df_corr.empty:
                fig_corr = px.imshow(
                    df_corr,
                    title="상관 행렬 히트맵",
                    aspect="auto",
                    color_continuous_scale="RdBu",
                )
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, width="stretch")

    # 비교 테이블
    comparison_table = data.get("comparison_table", [])
    if comparison_table:
        st.subheader("정상 vs 위험 구간 비교")
        df_comp = pd.DataFrame(comparison_table)
        st.dataframe(df_comp, use_container_width=True)

    # 핵심 주파수 대역
    important_freq_band = data.get("important_freq_band")
    if important_freq_band:
        st.subheader("핵심 주파수 대역")
        st.json(important_freq_band)


def render_operations_view(data: dict):
    """운영 및 활용 최적화 뷰"""
    st.header("⚙️ 운영 및 활용 최적화")

    rul_data = data.get("rul", {})
    wear_data = data.get("wear", {})

    # 경제적 추천 시점 (간단한 계산)
    rul_cuts = rul_data.get("cuts", 0.0)
    current_wear = wear_data.get("current", 0.0)
    max_limit = wear_data.get("max_limit", 200.0)
    ratio = wear_data.get("ratio_percent", 0.0)

    col1, col2, col3 = st.columns(3)
    with col1:
        # 최적 교체 시점 (마모율 80% 기준)
        optimal_ratio = 80.0
        if ratio >= optimal_ratio:
            st.metric("교체 권장", "즉시 교체 권장", delta="마모율 초과")
        else:
            remaining_ratio = optimal_ratio - ratio
            st.metric("교체 권장", f"마모율 {optimal_ratio}% 도달 시", delta=f"남은 여유: {remaining_ratio:.1f}%")
    with col2:
        st.metric("공정 안정성", "정상" if ratio < 80 else "주의 필요", delta=f"마모율: {ratio:.1f}%")
    with col3:
        st.metric("예상 남은 수명", f"{rul_cuts:.1f} 컷")

    st.divider()

    # 운영 지표 요약
    st.subheader("운영 지표 요약")
    summary_data = {
        "항목": ["현재 마모량", "마모 한계", "마모율", "남은 수명 (컷)", "예상 고장 시각"],
        "값": [
            f"{current_wear:.2f}",
            f"{max_limit:.1f}",
            f"{ratio:.1f}%",
            f"{rul_cuts:.1f}",
            rul_data.get("predicted_failure_time", "N/A")[:19] if rul_data.get("predicted_failure_time") else "N/A",
        ],
    }
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    # 교체 권장 메시지
    st.subheader("교체 권장 메시지")
    if ratio >= 100:
        st.error("⚠️ 즉시 교체가 필요합니다. 마모율이 한계를 초과했습니다.")
    elif ratio >= 80:
        st.warning("⚠️ 곧 교체가 필요합니다. 마모율이 80%를 초과했습니다.")
    elif ratio >= 60:
        st.info("ℹ️ 교체 계획을 수립하시기 바랍니다. 마모율이 60%를 초과했습니다.")
    else:
        st.success("✅ 정상 운영 중입니다. 현재 마모율이 안전 범위 내에 있습니다.")


def main():
    st.title("VibraCNC 대시보드")
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
        st.divider()
        
        # 시뮬레이션 모드
        st.subheader("시뮬레이션 모드")
        auto_refresh = st.checkbox("자동 새로고침", value=False, help="주기적으로 리포트를 자동으로 새로고침합니다")
        if auto_refresh:
            refresh_interval = st.slider("새로고침 간격 (초)", min_value=1, max_value=60, value=5)
            
            # 파일 수정 시간 체크를 위한 session state
            if "last_file_time" not in st.session_state:
                st.session_state.last_file_time = {}
            
            # 각 파일의 수정 시간 체크
            files_to_check = {
                "monitoring": monitoring_path,
                "diagnostics": diagnostics_path,
                "analysis": analysis_path,
            }
            
            file_changed = False
            for key, path in files_to_check.items():
                if path.exists():
                    current_mtime = path.stat().st_mtime
                    last_mtime = st.session_state.last_file_time.get(key, 0)
                    if current_mtime > last_mtime:
                        st.session_state.last_file_time[key] = current_mtime
                        file_changed = True
            
            if file_changed:
                st.success("🔄 새로운 데이터 감지됨! 페이지를 새로고침합니다...")
                time.sleep(0.5)  # 메시지 표시를 위한 짧은 대기
                st.rerun()
            
            # 자동 새로고침을 위한 JavaScript
            st.markdown(
                f"""
                <script>
                    setTimeout(function(){{
                        window.location.reload();
                    }}, {refresh_interval * 1000});
                </script>
                """,
                unsafe_allow_html=True
            )
            st.info(f"⏱️ {refresh_interval}초마다 자동 새로고침 중... (파일 변경 시 즉시 업데이트)")
        
        st.divider()
        if st.button("🔄 새로고침", type="primary"):
            # session state 초기화하여 강제 새로고침
            if "last_file_time" in st.session_state:
                st.session_state.last_file_time = {}
            st.rerun()
        
        # 시뮬레이터 실행 안내
        st.divider()
        st.info(
            "💡 **실시간 시뮬레이션 사용법:**\n\n"
            "1. 별도 터미널에서 시뮬레이터 실행:\n"
            "   ```bash\n"
            "   python -m vibracnc.simulator --interval 5\n"
            "   ```\n"
            "2. 이 대시보드에서 '자동 새로고침'을 켜세요"
        )

    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 실시간 모니터링", "🔮 예측 및 진단", "🔍 심층 분석", "⚙️ 운영 최적화"]
    )

    with tab1:
        monitoring_data = load_json(monitoring_path)
        if monitoring_data:
            render_monitoring_view(monitoring_data)
        else:
            st.error(f"모니터링 리포트를 찾을 수 없습니다: {monitoring_path}")

    with tab2:
        diagnostics_data = load_json(diagnostics_path)
        if diagnostics_data:
            render_diagnostics_view(diagnostics_data)
        else:
            st.error(f"진단 리포트를 찾을 수 없습니다: {diagnostics_path}")

    with tab3:
        analysis_data = load_json(analysis_path)
        if analysis_data:
            render_analysis_view(analysis_data)
        else:
            st.error(f"분석 리포트를 찾을 수 없습니다: {analysis_path}")

    with tab4:
        diagnostics_data = load_json(diagnostics_path)
        if diagnostics_data:
            render_operations_view(diagnostics_data)
        else:
            st.error(f"진단 리포트를 찾을 수 없습니다: {diagnostics_path}")


if __name__ == "__main__":
    main()

