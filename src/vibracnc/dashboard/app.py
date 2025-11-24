from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from vibracnc.anomaly.autoencoder import load_model
from vibracnc.config import DatasetConfig, ProjectPaths
from vibracnc.workflows import evaluate_anomaly_on_condition, load_anomaly_artifacts


def render_sidebar() -> tuple[Path, Path, DatasetConfig, ProjectPaths]:
    st.sidebar.header("설정")
    dataset_root = Path(st.sidebar.text_input("데이터셋 경로", value="data/phm2010"))
    models_dir = Path(st.sidebar.text_input("모델 디렉터리", value="artifacts/models"))
    figures_dir = Path(st.sidebar.text_input("결과 디렉터리", value="artifacts/figures"))

    sampling_rate = st.sidebar.number_input("샘플링 레이트 (Hz)", value=25600.0)
    window_seconds = st.sidebar.number_input("윈도우 길이 (초)", value=0.1)
    step_seconds = st.sidebar.number_input("윈도우 스텝 (초)", value=0.05)

    dataset_config = DatasetConfig(
        root_dir=dataset_root,
        sampling_rate=sampling_rate,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
    )
    project_paths = ProjectPaths(
        data_dir=dataset_root,
        models_dir=models_dir,
        figures_dir=figures_dir,
    )
    return dataset_root, models_dir, dataset_config, project_paths


def anomaly_page(dataset_root: Path, models_dir: Path, dataset_config: DatasetConfig) -> None:
    st.subheader("이상 탐지 결과")
    model_path = models_dir / "anomaly_autoencoder.pt"
    metadata_path = models_dir / "anomaly_artifacts.json"
    if not model_path.exists() or not metadata_path.exists():
        st.info("학습된 이상 탐지 모델이 없습니다. 먼저 학습 워크플로우를 실행하세요.")
        return

    artifacts = load_anomaly_artifacts(metadata_path)
    model = load_model(model_path, artifacts.config)

    condition = st.selectbox("조건 선택", dataset_config.normal_conditions + ("c2", "c3", "c5"))
    limit = st.number_input("분석할 컷 개수", min_value=1, max_value=200, value=50)

    if st.button("이상 점수 계산"):
        with st.spinner("이상 점수를 계산 중입니다..."):
            results = evaluate_anomaly_on_condition(
                model,
                artifacts,
                dataset_root,
                condition,
                dataset_config,
                per_condition_limit=int(limit),
            )
        st.success("계산 완료")
        st.dataframe(results.head(20))

        fig = px.line(results.reset_index(), x="index", y="error", title="재구성 오차")
        fig.add_hline(y=artifacts.threshold, line_dash="dash", annotation_text="임계값")
        st.plotly_chart(fig, use_container_width=True)

        anomaly_ratio = results["anomaly"].mean() * 100
        st.metric("이상 판정 비율", f"{anomaly_ratio:.1f}%")


def rul_page(project_paths: ProjectPaths) -> None:
    st.subheader("RUL 예측 결과")
    predictions_path = project_paths.figures_dir / "rul_predictions.csv"
    metrics_path = project_paths.figures_dir / "rul_metrics.csv"
    feature_importance_path = project_paths.models_dir / "rul_feature_importance.csv"

    if not predictions_path.exists() or not metrics_path.exists():
        st.info("RUL 모델 결과 파일이 없습니다. 학습 워크플로우를 먼저 실행하세요.")
        return

    preds = pd.read_csv(predictions_path)
    metrics = pd.read_csv(metrics_path)
    st.write("평가 지표", metrics)

    fig = px.scatter(preds, x="ground_truth", y="prediction", color="group", title="예측 vs 실제")
    fig.add_trace(
        px.line(x=[preds["ground_truth"].min(), preds["ground_truth"].max()],
                y=[preds["ground_truth"].min(), preds["ground_truth"].max()]).data[0]
    )
    st.plotly_chart(fig, use_container_width=True)

    if feature_importance_path.exists():
        fi = pd.read_csv(feature_importance_path)
        st.write("피처 중요도", fi.head(20))
        fig_imp = px.bar(fi.head(20), x="importance", y="feature", orientation="h", title="상위 중요 특징")
        st.plotly_chart(fig_imp, use_container_width=True)


def main() -> None:
    st.title("VibraCNC 대시보드")
    dataset_root, models_dir, dataset_config, project_paths = render_sidebar()

    tab1, tab2 = st.tabs(["이상 탐지", "RUL 예측"])
    with tab1:
        anomaly_page(dataset_root, models_dir, dataset_config)
    with tab2:
        rul_page(project_paths)


if __name__ == "__main__":
    main()

