# 구글 드라이브 업로드 체크리스트

## 📦 필수 파일 (반드시 업로드)

### 1. 데이터셋 폴더
```
📁 data/phm2010/
├── 📁 c1/ (모든 CSV 파일)
├── 📁 c2/ (모든 CSV 파일)
├── 📁 c3/ (모든 CSV 파일)
├── 📁 c4/ (모든 CSV 파일)
├── 📁 c5/ (모든 CSV 파일)
├── 📁 c6/ (모든 CSV 파일)
└── 📄 wear.csv (중요!)
```
**크기:** 수 GB  
**압축 권장:** `data/phm2010.zip`으로 압축하여 업로드

---

### 2. 모델 가중치 파일들
```
📁 artifacts/models/
├── ✅ best_anomaly_model.pth (이상 탐지 모델 - 필수)
├── ✅ wear_regressor.pth (마모 예측 모델 - 필수)
├── ✅ wear_scaler_params.npy (마모 모델 정규화 파라미터 - 필수)
├── ✅ anomaly_artifacts.json (이상 탐지 메타데이터 - 필수)
├── ⚠️ anomaly_autoencoder.pt (구버전, 선택사항)
├── ⚠️ rul_random_forest.pkl (RUL 예측 모델, 선택사항)
└── ⚠️ rul_feature_importance.csv (RUL 피처 중요도, 선택사항)
```
**크기:** 수십 MB  
**경로:** `artifacts/models/` 폴더 전체를 압축하여 업로드

---

### 3. 사전 계산된 결과 파일들 (cnc_viewer.py 사용 시 필수)
```
📁 artifacts/results/
├── ✅ c1.npy, c2.npy, c3.npy, c4.npy, c5.npy, c6.npy (이상 점수)
├── ✅ c1_features.npy, c2_features.npy, ... (센서별 기여도)
└── ✅ c1_wear.npy, c2_wear.npy, ... (마모 예측 결과)
```
**크기:** 수백 MB  
**생성 방법:**
```bash
python generate_results.py
python train_wear_model.py
```

---

## 📊 선택적 파일 (대시보드 사용 시)

### 4. 리포트 파일들
```
📁 artifacts/monitoring/
├── monitoring_report.json
├── diagnostics_report.json
└── analysis_report.json

📁 artifacts/figures/
├── rul_predictions.csv
└── anomaly/
    ├── c1_anomaly.csv
    ├── c4_anomaly.csv
    └── c6_anomaly.csv
```

---

## 📝 업로드 방법

### 방법 1: 폴더별 압축 업로드 (권장)
1. 각 폴더를 개별적으로 압축
   - `data/phm2010.zip`
   - `artifacts_models.zip` (artifacts/models/ 폴더)
   - `artifacts_results.zip` (artifacts/results/ 폴더)
2. 구글 드라이브에 업로드
3. 다운로드 후 해당 경로에 압축 해제

### 방법 2: 전체 폴더 업로드
1. 구글 드라이브에 폴더 구조 그대로 업로드
2. 다운로드 후 프로젝트 루트에 배치

---

## ✅ 최소 필수 파일 요약

**cnc_viewer.py 실행에 필요한 최소 파일:**
1. ✅ `data/phm2010/` (전체 폴더)
2. ✅ `artifacts/models/best_anomaly_model.pth`
3. ✅ `artifacts/models/wear_regressor.pth`
4. ✅ `artifacts/models/wear_scaler_params.npy`
5. ✅ `artifacts/models/anomaly_artifacts.json`
6. ✅ `artifacts/results/*.npy` (모든 조건의 결과 파일)

**또는 결과 파일을 직접 생성:**
```bash
# 데이터와 모델만 있으면 결과 파일 생성 가능
python generate_results.py
python train_wear_model.py
```

---

## 🔍 파일 확인 명령어

업로드 전 파일 존재 여부 확인:
```bash
# 모델 파일 확인
ls artifacts/models/best_anomaly_model.pth
ls artifacts/models/wear_regressor.pth
ls artifacts/models/wear_scaler_params.npy

# 데이터 확인
ls data/phm2010/wear.csv
ls data/phm2010/c1/

# 결과 파일 확인 (생성된 경우)
ls artifacts/results/c1.npy
ls artifacts/results/c1_features.npy
ls artifacts/results/c1_wear.npy
```

