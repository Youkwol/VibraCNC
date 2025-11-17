# VibraCNC

FFT 기반 CNC 진동 이상 탐지 및 RUL(Remaining Useful Life) 예측 파이프라인입니다.  
PHM 2010 밀링 데이터셋을 기반으로 두 가지 프로젝트 목표를 지원합니다.

1. **프로젝트 1 – 이상 탐지:** FFT 특징과 LSTM AutoEncoder를 이용하여 공구 마모로 인한 비정상 진동 패턴을 감지합니다.
2. **프로젝트 2 – RUL 예측:** 센서 통계 특징을 추출해 Random Forest 회귀 모델로 공구의 잔여 수명을 추정합니다.

## ⚙️ 환경 구성

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> `kagglehub`를 사용해 데이터를 내려받으려면 Kaggle API 토큰(`~/.kaggle/kaggle.json`)을 먼저 설정하세요.

## 📦 데이터 다운로드

```bash
python -m vibracnc.cli train-anomaly --download
```

위 명령은 Kaggle에서 PHM 2010 데이터셋을 내려받아 `data/phm2010` 경로에 복사합니다. 데이터 구조는 다음과 같이 구성하는 것을 권장합니다.

```
data/phm2010/
├─ c1/
│  ├─ 0001.csv
│  ├─ ...
├─ c4/
├─ c6/
└─ wear.csv
```

- `c1`, `c4`, `c6`의 앞부분 컷(기본 30개)을 정상 상태로 간주합니다.
- `wear.csv`는 각 컷 파일 경로와 마모량(`wear` 또는 `VB`)을 포함해야 합니다.
- CSV에 헤더가 없다면 `timestamp`, `vx`, `vy`, `vz`, `sx`, `sy`, `sz`, `temp` 순으로 컬럼이 배치되어야 합니다.
- 필요 시 `src/vibracnc/config.py`에서 센서/FFT 컬럼 및 샘플링 주기를 조정하세요.

## 🧠 이상 탐지 학습

```bash
python -m vibracnc.cli train-anomaly \
  --dataset-dir data/phm2010 \
  --models-dir artifacts/models \
  --per-condition-limit 40 \
  --device cuda
```

산출물:
- `artifacts/models/anomaly_autoencoder.pt`
- `artifacts/models/anomaly_artifacts.json`

> Use `--device` to choose `auto`/`cpu`/`cuda` (default `auto`). CUDA GPUs are used automatically when available.

## 🔮 RUL 예측 학습

```bash
python -m vibracnc.cli train-rul \
  --dataset-dir data/phm2010 \
  --models-dir artifacts/models \
  --figures-dir artifacts/figures
```

산출물:
- `artifacts/models/rul_random_forest.pkl`
- `artifacts/models/rul_feature_importance.csv`
- `artifacts/figures/rul_predictions.csv`
- `artifacts/figures/rul_metrics.csv`

## 📊 Streamlit 대시보드

```bash
streamlit run src/vibracnc/dashboard/app.py
```

기능:
- 학습된 LSTM AutoEncoder 로드 후 조건별 이상 점수 시각화
- RUL 예측 결과 및 피처 중요도 확인

## 📚 프로젝트 구조

```
src/vibracnc/
├─ data/           # 데이터 다운로드 및 전처리 도구
├─ anomaly/        # FFT 기반 LSTM AutoEncoder 구현
├─ rul/            # RUL 특징 추출 및 회귀 모델
├─ dashboard/      # Streamlit 앱
├─ workflows.py    # 학습/평가 워크플로우
└─ cli.py          # 명령행 인터페이스
```

## ✅ TODO

- [ ] 실제 데이터 구조에 맞춰 `DatasetConfig` 조정
- [ ] 학습 결과 검증 및 임계값 튜닝
- [ ] 추가 모델(예: Isolation Forest, GRU) 비교 실험
- [ ] 대시보드에 실시간 데이터 스트림 연동