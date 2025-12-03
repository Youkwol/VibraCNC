from __future__ import annotations

import numpy as np
from pathlib import Path

RESULT_DIR = Path("artifacts/results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 테스트용 시나리오 생성 (400 스텝 정도면 충분)
steps = 400
print(f"테스트 시나리오 생성 중... (steps={steps})")

# 이상 점수: 0.01에서 0.08까지 천천히 증가 (임계값 0.0674 근처에서 위험 상태로 전환)
scores = np.linspace(0.01, 0.08, steps) + np.random.normal(0, 0.002, steps)
scores = np.clip(scores, 0.0, 0.15)  # 음수 방지 및 최대값 제한

# 센서별 기여도 (7개 센서)
# 각 센서는 기본 점수에 약간의 변동을 주어 다양성 추가
features = np.stack(
    [
        scores * (1.0 + 0.1 * np.sin(np.linspace(0, 4 * np.pi, steps))),  # Force X (사인파 패턴)
        scores * 0.8,  # Force Y
        scores * 1.2,  # Force Z (가장 높은 기여도)
        scores * 0.5,  # Vib X
        scores * 0.6,  # Vib Y
        scores * 0.9,  # Vib Z
        scores * 0.3,  # AE (가장 낮은 기여도)
    ],
    axis=1,
)

# 음수 방지
features = np.clip(features, 0.0, None)

# 파일 저장
score_path = RESULT_DIR / "test_fast.npy"
feature_path = RESULT_DIR / "test_fast_features.npy"

np.save(score_path, scores.astype(np.float32))
np.save(feature_path, features.astype(np.float32))

print(f"✅ 테스트 시나리오 생성 완료!")
print(f"   - {score_path} (shape: {scores.shape})")
print(f"   - {feature_path} (shape: {features.shape})")
print(f"\n이제 cnc_viewer.py에서 'test_fast' 시나리오를 선택하세요!")

