# GitHub 업로드 가이드

## 📋 사전 준비

1. **GitHub 저장소 생성**
   - GitHub에서 새 저장소 생성 (예: `VibraCNC`)
   - 저장소 URL 확인 (예: `https://github.com/YourUsername/VibraCNC.git`)

2. **Git 설치 확인**
   ```bash
   git --version
   ```

## 🚀 업로드 절차

### 1. Git 초기화 (처음 한 번만)

```bash
cd D:\VibraCNC

# Git 저장소 초기화
git init

# 원격 저장소 연결 (YourUsername과 저장소 이름을 실제 값으로 변경)
git remote add origin https://github.com/YourUsername/VibraCNC.git
```

### 2. 파일 추가 및 커밋

```bash
# 모든 파일 추가 (.gitignore에 제외된 파일은 자동으로 제외됨)
git add .

# 커밋 메시지와 함께 커밋
git commit -m "Initial commit: CNC 예지보전 시스템

- 이상 탐지 모델 (LSTM AutoEncoder)
- 마모 예측 모델 (CNN-LSTM)
- 실시간 모니터링 뷰어 (Tkinter)
- 종합 분석 리포트
- 데이터 전처리 및 결과 생성 스크립트"
```

### 3. GitHub에 푸시

```bash
# 메인 브랜치로 푸시
git branch -M main
git push -u origin main
```

## 📁 업로드되는 파일 목록

### ✅ 업로드되는 파일
- 모든 Python 소스 코드 (`.py` 파일)
- `requirements.txt`
- `README.md`
- `GOOGLE_DRIVE_UPLOAD.md`
- `.gitignore`
- 프로젝트 구조 파일들

### ❌ 업로드되지 않는 파일 (.gitignore에 의해 제외)
- 가상환경 (`.venv/`, `.venv312/`)
- 모델 가중치 파일 (`*.pth`, `*.pt`, `*.pkl`)
- 데이터 파일 (`data/phm2010/`)
- 사전 계산된 결과 (`artifacts/results/*.npy`)
- 캐시 파일 (`__pycache__/`)
- IDE 설정 파일 (`.vscode/`, `.idea/`)

## 🔄 이후 업데이트 방법

코드를 수정한 후:

```bash
# 변경된 파일 확인
git status

# 변경된 파일 추가
git add .

# 커밋
git commit -m "변경 사항 설명"

# GitHub에 푸시
git push
```

## ⚠️ 주의사항

1. **큰 파일은 GitHub에 올리지 마세요**
   - 모델 파일 (`.pth`, `.pt`, `.pkl`)은 구글 드라이브에 업로드
   - 데이터 파일 (`data/phm2010/`)은 구글 드라이브에 업로드
   - 자세한 내용은 `GOOGLE_DRIVE_UPLOAD.md` 참고

2. **민감한 정보 확인**
   - API 키, 비밀번호 등은 절대 커밋하지 마세요
   - `.gitignore`에 추가하거나 환경 변수로 관리

3. **커밋 메시지 작성**
   - 명확하고 간결하게 작성
   - 무엇을 변경했는지 설명

## 📝 체크리스트

업로드 전 확인:
- [ ] `.gitignore` 파일이 올바르게 설정되어 있는지 확인
- [ ] 민감한 정보(API 키, 비밀번호)가 코드에 없는지 확인
- [ ] `README.md`가 최신 정보를 반영하는지 확인
- [ ] `requirements.txt`가 모든 의존성을 포함하는지 확인
- [ ] 테스트 파일이나 임시 파일이 제외되었는지 확인

