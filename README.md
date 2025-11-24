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
$env:PYTHONPATH = "$PWD\src"
python -m vibracnc.cli train-anomaly \
  --dataset-dir data/phm2010 \
  --models-dir artifacts/models \
  --per-condition-limit 10 \
  --epochs 10 \
  --device cuda
```

**중요 사항:**
- `--per-condition-limit 10`: 초기 10개 cut만 정상 데이터로 사용합니다. PHM 데이터셋의 특성상 초기 cut만 정상 상태로 간주해야 합니다.
- `--max-wear`: 마모량 기반 필터링도 가능합니다 (예: `--max-wear 100.0`).
- 학습 시 정규화 파라미터(`norm_min`, `norm_max`)가 자동으로 저장되어 추론 시 일관되게 적용됩니다.
- 학습 데이터의 정규화 파라미터를 추론 데이터에도 동일하게 적용해야 정확한 이상 탐지가 가능합니다.

산출물:
- `artifacts/models/anomaly_autoencoder.pt`: 학습된 LSTM AutoEncoder 모델
- `artifacts/models/anomaly_artifacts.json`: 임계값, 정규화 파라미터, 학습 히스토리 등 메타데이터

> Use `--device` to choose `auto`/`cpu`/`cuda` (default `auto`). CUDA GPUs are used automatically when available.

## 🔍 이상 탐지 추론

```bash
python -m vibracnc.cli infer-anomaly \
  --dataset-dir data/phm2010 \
  --models-dir artifacts/models \
  --conditions c2 c3 \
  --per-condition-limit 60 \
  --output-dir artifacts/figures/anomaly
  --device cpu
```

- 각 조건별 재구성 오차/이상 판정을 계산한 CSV가 `output-dir`(`artifacts/figures/anomaly` 기본값)에 저장됩니다.
- 콘솔에는 윈도우 수와 이상 비율이 요약되어 출력됩니다. `--conditions`를 생략하면 `DatasetConfig.normal_conditions`가 사용됩니다.
- GPU 가중치로 학습한 모델을 CPU에서 추론하려면 `--device cpu`를 명시해 주세요(`auto` 기본값은 GPU가 있을 때 CUDA를 사용).

## 🧾 규칙 기반 이상 탐지

```bash
python -m vibracnc.cli rule-anomaly \
  --dataset-dir data/phm2010 \
  --conditions c2 c3 \
  --per-condition-limit 40 \
  --output-dir artifacts/figures/rule_based
```

- `src/vibracnc/config.py`에 정의된 `RuleDefinition` 목록(예: 온도 65 °C 초과, 축별 RMS 초과 등)을 이용해 윈도우별 규칙 위반 여부를 계산합니다.
- 조건별 결과는 `output-dir/<condition>_rule_based.csv`에 저장되며, CSV에는 규칙 이름·임계값·실측값·위반 여부가 모두 포함됩니다.
- 규칙을 변경하고 싶다면 `config.py`의 `DEFAULT_RULES` 값을 수정하거나 새로운 `RuleDefinition`을 추가하세요.

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

## 📊 리포트 생성

### 모니터링 리포트 (실시간 모니터링용)

```bash
$env:PYTHONPATH = "$PWD\src"
python -m vibracnc.cli monitoring-report \
  --dataset-dir data/phm2010 \
  --models-dir artifacts/models \
  --output-path artifacts/monitoring/monitoring_report.json \
  --device cuda
```

최근 N개 cut에 대한 이상 탐지 결과를 생성합니다. 실시간 모니터링 대시보드에서 사용됩니다.

### 진단 리포트 (RUL 예측 및 마모 진단)

```bash
$env:PYTHONPATH = "$PWD\src"
python -m vibracnc.cli diagnostics-report \
  --dataset-dir data/phm2010 \
  --models-dir artifacts/models \
  --rul-predictions artifacts/figures/rul_predictions.csv \
  --output-path artifacts/monitoring/diagnostics_report.json \
  --max-wear-limit 200.0 \
  --cut-per-hour 10.0
```

RUL 예측 결과와 마모 진단 정보를 생성합니다. `train-rul`을 먼저 실행해야 합니다.

### 분석 리포트 (심층 분석용)

```bash
$env:PYTHONPATH = "$PWD\src"
python -m vibracnc.cli analysis-report \
  --anomaly-csv artifacts/figures/anomaly/c1_anomaly.csv \
  --feature-importance-csv artifacts/models/rul_feature_importance.csv \
  --output-path artifacts/monitoring/analysis_report.json
```

피처 중요도, 센서 상관관계, 정상/이상 통계 등을 분석합니다. `infer-anomaly`와 `train-rul`을 먼저 실행해야 합니다.

## 📊 Streamlit 대시보드

### 실시간 모니터링 대시보드

```bash
streamlit run dashboard.py
```

기능:
- 실시간 모니터링 뷰: 최근 cut들의 이상 탐지 상태, FFT 스펙트럼, 이상 점수 추이
- 예측 및 진단 뷰: RUL 예측 결과, 마모 진행 상황, 잔여 수명 추정
- 심층 분석 뷰: 피처 중요도, 센서 상관관계, 정상/이상 통계 비교
- 운영 및 활용 최적화 뷰: (구현 예정)

### 정적 분석 대시보드

```bash
streamlit run analysis_dashboard.py
```

기능:
- 이상 탐지 모델 상세 분석: 모델 구조, 학습 과정, 임계값 설정 방법 설명
- RUL 예측 모델 상세 분석: 피처 중요도, 예측 정확도, 마모 진행 패턴 분석
- 전체 데이터셋 통계: 조건별 마모 진행, 이상 탐지 비율, 예측 오차 분포

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

## 🔧 주요 구현 사항 및 해결한 문제들

### 정규화 문제 해결 (2024)

**문제:** 학습 데이터와 추론 데이터를 각각 독립적으로 정규화하여 "reconstruction error inversion" 현상 발생
- 마모가 진행된 데이터가 정상 데이터보다 더 낮은 재구성 오차를 보이는 현상
- 이상 탐지율이 0% 또는 100%로 극단적으로 나타남

**해결 방법:**
- 학습 데이터에서 정규화 파라미터(`norm_min`, `norm_max`) 계산
- 추론 시 동일한 정규화 파라미터 적용
- `AnomalyDetectionArtifacts`에 정규화 파라미터 저장 및 로드

**관련 파일:**
- `src/vibracnc/anomaly/pipeline.py`: `create_fft_features` 함수 수정
- `src/vibracnc/workflows.py`: 정규화 파라미터 저장/로드 로직 추가

### 임계값 설정 개선

**변경 사항:**
- Percentile 기반 임계값 → `mean + N * std` 기반 임계값으로 변경
- 윈도우 단위 임계값(`threshold`)과 프레임 단위 임계값(`frame_threshold`) 분리
- 기본값: `mean_error + 1 * std_error`

### 대시보드 구현

**구현된 기능:**
1. **실시간 모니터링 대시보드** (`dashboard.py`)
   - 최근 cut들의 이상 탐지 상태 실시간 표시
   - FFT 스펙트럼 시각화
   - 이상 점수 추이 그래프
   - 자동 새로고침 기능

2. **정적 분석 대시보드** (`analysis_dashboard.py`)
   - 이상 탐지 모델 상세 분석 및 설명
   - RUL 예측 모델 상세 분석
   - 전체 데이터셋 통계 및 시각화

## 🚀 다른 컴퓨터에서 사용하기

### 1. GitHub에서 코드 클론

```bash
git clone https://github.com/Youkwol/VibraCNC.git
cd VibraCNC
```

### 2. 가상환경 설정

```bash
python -m venv .venv312
.venv312\Scripts\activate  # Windows PowerShell
python -m pip install -r requirements.txt
```

### 3. PYTHONPATH 설정

PowerShell에서:
```powershell
$env:PYTHONPATH = "$PWD\src"
```

또는 세션별로 자동 설정하려면 `.venv312\Scripts\Activate.ps1`에 다음 추가:
```powershell
$env:PYTHONPATH = "$PWD\src"
```

### 4. 데이터 및 모델 파일 다운로드

**구글 드라이브에 올려야 할 파일들:**

1. **데이터셋** (`data/phm2010/`)
   - 전체 PHM 2010 데이터셋 폴더
   - 크기: 약 수 GB
   - 구조:
     ```
     data/phm2010/
     ├─ c1/
     ├─ c4/
     ├─ c6/
     └─ wear.csv
     ```

2. **학습된 모델 파일들** (`artifacts/models/`)
   - `anomaly_autoencoder.pt`: 이상 탐지 모델 가중치
   - `anomaly_artifacts.json`: 이상 탐지 모델 메타데이터 (임계값, 정규화 파라미터 포함)
   - `rul_random_forest.pkl`: RUL 예측 모델
   - `rul_feature_importance.csv`: RUL 피처 중요도

3. **생성된 리포트 파일들** (선택사항)
   - `artifacts/monitoring/monitoring_report.json`
   - `artifacts/monitoring/diagnostics_report.json`
   - `artifacts/monitoring/analysis_report.json`
   - `artifacts/figures/rul_predictions.csv`
   - `artifacts/figures/anomaly/*.csv`

**다운로드 후 배치:**
```bash
# 데이터셋을 data/phm2010/에 배치
# 모델 파일들을 artifacts/models/에 배치
# 리포트 파일들을 해당 경로에 배치
```

### 5. 실행

```bash
# 대시보드 실행
streamlit run dashboard.py

# 또는 리포트 재생성
$env:PYTHONPATH = "$PWD\src"
python -m vibracnc.cli monitoring-report --dataset-dir data/phm2010 --models-dir artifacts/models --output-path artifacts/monitoring/monitoring_report.json
```

## ✅ TODO

- [x] 정규화 문제 해결
- [x] 임계값 설정 개선
- [x] 실시간 모니터링 대시보드 구현
- [x] 정적 분석 대시보드 구현
- [ ] 운영 및 활용 최적화 뷰 완성
- [ ] 추가 모델(예: Isolation Forest, GRU) 비교 실험
- [ ] 대시보드에 실제 실시간 데이터 스트림 연동