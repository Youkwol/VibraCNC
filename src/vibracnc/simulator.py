from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

from vibracnc.anomaly.autoencoder import AutoencoderConfig, load_model
from vibracnc.config import DatasetConfig, ProjectPaths
from vibracnc.data.preprocessing import WindowingConfig
from vibracnc.monitoring import build_monitoring_summary, load_recent_cut_samples
from vibracnc.workflows import load_anomaly_artifacts


class MonitoringSimulator:
    """실시간 모니터링 시뮬레이터"""

    def __init__(
        self,
        dataset_dir: Path,
        models_dir: Path,
        output_dir: Path,
        dataset_config: DatasetConfig,
        conditions: list[str],
        initial_cuts: int = 10,
        max_cuts: int = 50,
    ):
        self.dataset_dir = dataset_dir
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_config = dataset_config
        self.conditions = conditions
        self.current_cuts = initial_cuts
        self.max_cuts = max_cuts
        self.output_path = output_dir / "monitoring_report.json"
        
        # 모델과 아티팩트를 한 번만 로드
        self._model = None
        self._artifacts = None
        self._window_config = None

    def _load_model_once(self):
        """모델과 아티팩트를 한 번만 로드합니다."""
        if self._model is None or self._artifacts is None:
            metadata_path = self.models_dir / "anomaly_artifacts.json"
            model_path = self.models_dir / "anomaly_autoencoder.pt"
            
            if not metadata_path.exists() or not model_path.exists():
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.models_dir}")
            
            self._artifacts = load_anomaly_artifacts(metadata_path)
            self._artifacts.config.device = "cuda"
            self._model = load_model(model_path, self._artifacts.config)
            
            # 윈도우 설정도 한 번만
            self._window_config = WindowingConfig.from_seconds(
                window_seconds=self.dataset_config.window_seconds,
                step_seconds=self.dataset_config.step_seconds,
                sampling_rate=self.dataset_config.sampling_rate,
            )
    
    def generate_report(self, num_cuts: int) -> bool:
        """지정된 개수의 cut으로 모니터링 리포트를 생성합니다."""
        try:
            # 모델을 한 번만 로드
            self._load_model_once()
            
            # 샘플 로드
            samples = load_recent_cut_samples(
                self.dataset_dir,
                self.dataset_config,
                conditions=self.conditions,
                per_condition_limit=num_cuts,
            )
            
            if not samples:
                print("오류: 샘플을 찾을 수 없습니다.")
                return False
            
            # 리포트 생성
            summary = build_monitoring_summary(
                samples, self._model, self._artifacts, self.dataset_config, self._window_config
            )
            
            # JSON 저장
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(
                json.dumps(summary.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            
            return True
        except Exception as e:
            print(f"리포트 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def step(self) -> bool:
        """다음 cut을 추가하고 리포트를 갱신합니다."""
        if self.current_cuts >= self.max_cuts:
            return False
        self.current_cuts += 1
        print(f"[시뮬레이터] Cut {self.current_cuts}/{self.max_cuts} 생성 중...")
        return self.generate_report(self.current_cuts)

    def run(self, interval_seconds: float = 5.0, auto_stop: bool = True):
        """시뮬레이션을 실행합니다."""
        print(f"[시뮬레이터] 시작: 초기 {self.current_cuts}개 cut")
        if not self.generate_report(self.current_cuts):
            print("[시뮬레이터] 초기 리포트 생성 실패")
            return

        while self.current_cuts < self.max_cuts:
            time.sleep(interval_seconds)
            if not self.step():
                break
            print(f"[시뮬레이터] Cut {self.current_cuts} 완료")

        print(f"[시뮬레이터] 완료: 총 {self.current_cuts}개 cut 생성")


def run_simulation(
    dataset_dir: str = "data/phm2010",
    models_dir: str = "artifacts/models",
    output_dir: str = "artifacts/monitoring",
    conditions: Optional[list[str]] = None,
    initial_cuts: int = 10,
    max_cuts: int = 50,
    interval_seconds: float = 5.0,
):
    """시뮬레이션을 실행하는 헬퍼 함수"""
    if conditions is None:
        conditions = ["c1", "c4", "c6"]

    dataset_config = DatasetConfig(root_dir=Path(dataset_dir))
    simulator = MonitoringSimulator(
        dataset_dir=Path(dataset_dir),
        models_dir=Path(models_dir),
        output_dir=Path(output_dir),
        dataset_config=dataset_config,
        conditions=conditions,
        initial_cuts=initial_cuts,
        max_cuts=max_cuts,
    )
    simulator.run(interval_seconds=interval_seconds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="모니터링 시뮬레이터")
    parser.add_argument("--dataset-dir", default="data/phm2010")
    parser.add_argument("--models-dir", default="artifacts/models")
    parser.add_argument("--output-dir", default="artifacts/monitoring")
    parser.add_argument("--conditions", nargs="+", default=["c1", "c4", "c6"])
    parser.add_argument("--initial-cuts", type=int, default=10)
    parser.add_argument("--max-cuts", type=int, default=50)
    parser.add_argument("--interval", type=float, default=5.0, help="각 cut 간격 (초)")

    args = parser.parse_args()
    run_simulation(
        dataset_dir=args.dataset_dir,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        conditions=args.conditions,
        initial_cuts=args.initial_cuts,
        max_cuts=args.max_cuts,
        interval_seconds=args.interval,
    )

