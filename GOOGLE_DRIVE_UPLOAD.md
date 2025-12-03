# 구글 드라이브 업로드 가이드

다른 컴퓨터에서 VibraCNC 프로젝트를 사용하기 위해 구글 드라이브에 업로드해야 할 파일 목록입니다.

## 📦 필수 파일 (반드시 업로드 필요)

### 1. 데이터셋 (`data/phm2010/`)

**경로:** `data/phm2010/`  
**크기:** 약 수 GB (전체 데이터셋)

**구조:**
```
data/phm2010/
├─ c1/
│  ├─ c_1_001.csv
│  ├─ c_1_002.csv
│  └─ ... (모든 cut 파일)
├─ c4/
│  ├─ c_4_001.csv
│  └─ ...
├─ c6/
│  ├─ c_6_001.csv
│  └─ ...
└─ wear.csv  (중요: 모든 조건의 마모량 정보 포함)
```

**업로드 방법:**
- 전체 `data/phm2010/` 폴더를 압축하여 업로드하거나
- 각 조건 폴더(`c1/`, `c4/`, `c6/`)와 `wear.csv`를 개별적으로 업로드

### 2. 학습된 모델 파일들 (`artifacts/models/`)

**경로:** `artifacts/models/`  
**크기:** 약 수십 MB

**필수 파일:**
- ✅ `best_anomaly_model.pth` - 이상 탐지 LSTM AutoEncoder 모델 가중치 (최신 버전)
- ✅ `anomaly_autoencoder.pt` - 이상 탐지 LSTM AutoEncoder 모델 가중치 (구버전, 호환성)
- ✅ `anomaly_artifacts.json` - 이상 탐지 모델 메타데이터 (임계값, 정규화 파라미터, 학습 히스토리)
- ✅ `wear_regressor.pth` - 마모 예측 CNN-LSTM 모델 가중치 (새로 추가)
- ✅ `wear_scaler_params.npy` - 마모 모델 정규화 파라미터 (새로 추가)
- ✅ `rul_random_forest.pkl` - RUL 예측 Random Forest 모델
- ✅ `rul_feature_importance.csv` - RUL 피처 중요도

**생성 방법:**
```bash
# 이상 탐지 모델 학습 (기존 CLI 방식)
$env:PYTHONPATH = "$PWD\src"
python -m vibracnc.cli train-anomaly --dataset-dir data/phm2010 --models-dir artifacts/models --per-condition-limit 10 --epochs 10 --device cuda

# 또는 새로운 스크립트 방식 (generate_results.py 사용 시)
# generate_results.py는 best_anomaly_model.pth를 사용합니다

# 마모 예측 모델 학습 (새로 추가)
python train_wear_model.py

# RUL 예측 모델 학습
python -m vibracnc.cli train-rul --dataset-dir data/phm2010 --models-dir artifacts/models --figures-dir artifacts/figures
```

## 📊 선택적 파일 (대시보드 사용 시 필요)

### 3. 사전 계산된 결과 파일들 (`artifacts/results/`)

**경로:** `artifacts/results/`  
**크기:** 약 수백 MB

**파일 목록 (cnc_viewer.py 사용 시 필요):**
- `c1.npy`, `c2.npy`, `c3.npy`, `c4.npy`, `c5.npy`, `c6.npy` - 각 조건별 이상 점수
- `c1_features.npy`, `c2_features.npy`, ... - 각 조건별 센서별 기여도 (feature-wise error)
- `c1_wear.npy`, `c2_wear.npy`, ... - 각 조건별 마모 예측 결과

**생성 방법:**
```bash
# 이상 점수 및 센서별 기여도 계산
python generate_results.py

# 마모 예측 결과 생성 (train_wear_model.py가 자동으로 생성)
python train_wear_model.py
```

### 4. 생성된 리포트 파일들

**경로:** `artifacts/monitoring/` 및 `artifacts/figures/`

**파일 목록:**
- `artifacts/monitoring/monitoring_report.json` - 실시간 모니터링 리포트
- `artifacts/monitoring/diagnostics_report.json` - RUL 진단 리포트
- `artifacts/monitoring/analysis_report.json` - 심층 분석 리포트
- `artifacts/figures/rul_predictions.csv` - RUL 예측 결과
- `artifacts/figures/anomaly/c1_anomaly.csv` - 이상 탐지 결과 (조건별)
- `artifacts/figures/anomaly/c4_anomaly.csv`
- `artifacts/figures/anomaly/c6_anomaly.csv`

**생성 방법:**
```bash
# 모니터링 리포트
python -m vibracnc.cli monitoring-report --dataset-dir data/phm2010 --models-dir artifacts/models --output-path artifacts/monitoring/monitoring_report.json

# 진단 리포트 (train-rul 먼저 실행 필요)
python -m vibracnc.cli train-rul --dataset-dir data/phm2010 --models-dir artifacts/models --figures-dir artifacts/figures
python -m vibracnc.cli diagnostics-report --dataset-dir data/phm2010 --models-dir artifacts/models --rul-predictions artifacts/figures/rul_predictions.csv --output-path artifacts/monitoring/diagnostics_report.json

# 분석 리포트 (infer-anomaly 먼저 실행 필요)
python -m vibracnc.cli infer-anomaly --dataset-dir data/phm2010 --models-dir artifacts/models --conditions c1 c4 c6 --output-dir artifacts/figures/anomaly
python -m vibracnc.cli analysis-report --anomaly-csv artifacts/figures/anomaly/c1_anomaly.csv --feature-importance-csv artifacts/models/rul_feature_importance.csv --output-path artifacts/monitoring/analysis_report.json
```

## 🚫 업로드 불필요한 파일

다음 파일들은 GitHub에 이미 있거나 각 컴퓨터에서 새로 생성할 수 있으므로 업로드할 필요가 없습니다:

- ❌ 소스 코드 (`src/` 폴더) - GitHub에 있음
- ❌ `requirements.txt` - GitHub에 있음
- ❌ `README.md` - GitHub에 있음
- ❌ 가상환경 (`.venv312/`) - 각 컴퓨터에서 새로 생성
- ❌ `__pycache__/` - 자동 생성됨
- ❌ 임시 분석 스크립트 (`check_*.py`, `analyze_*.py` 등)

## 📥 다운로드 후 배치 방법

1. **데이터셋 배치:**
   ```bash
   # 구글 드라이브에서 다운로드한 data/phm2010/ 폴더를 프로젝트 루트에 배치
   ```

2. **모델 파일 배치:**
   ```bash
   # artifacts/models/ 폴더 생성
   mkdir -p artifacts/models
   
   # 구글 드라이브에서 다운로드한 모델 파일들을 artifacts/models/에 복사
   ```

3. **리포트 파일 배치 (선택사항):**
   ```bash
   # artifacts/monitoring/ 폴더 생성
   mkdir -p artifacts/monitoring
   
   # artifacts/figures/ 폴더 생성
   mkdir -p artifacts/figures/anomaly
   
   # 구글 드라이브에서 다운로드한 리포트 파일들을 해당 경로에 복사
   ```

## ✅ 검증 방법

다운로드 후 다음 명령어로 확인:

```bash
# 모델 파일 확인
ls artifacts/models/
# anomaly_autoencoder.pt, anomaly_artifacts.json, rul_random_forest.pkl, rul_feature_importance.csv가 있어야 함

# 데이터셋 확인
ls data/phm2010/
# c1/, c4/, c6/, wear.csv가 있어야 함

# 대시보드 실행 테스트
streamlit run dashboard.py
```

## 📝 요약

**최소 필수 파일:**
1. `data/phm2010/` (전체 폴더)
2. `artifacts/models/best_anomaly_model.pth` (또는 `anomaly_autoencoder.pt`)
3. `artifacts/models/anomaly_artifacts.json`
4. `artifacts/models/wear_regressor.pth` (마모 예측 사용 시)
5. `artifacts/models/wear_scaler_params.npy` (마모 예측 사용 시)
6. `artifacts/models/rul_random_forest.pkl` (RUL 예측 사용 시)
7. `artifacts/models/rul_feature_importance.csv` (RUL 예측 사용 시)

**cnc_viewer.py 사용 시 추가 필요:**
- `artifacts/results/*.npy` (모든 조건의 이상 점수, feature-wise error, 마모 예측)

**대시보드 사용 시 추가 필요:**
- `artifacts/monitoring/*.json`
- `artifacts/figures/rul_predictions.csv`
- `artifacts/figures/anomaly/*.csv`

