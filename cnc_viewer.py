from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tkinter as tk
from matplotlib import font_manager, rcParams
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import messagebox, ttk


RESULT_DIR = "artifacts/results"
CONDITIONS = ["test_fast", "c1", "c4", "c6", "c2", "c3", "c5"]
STRIDE = 10
SEQ_LEN = 50
DOWNSAMPLE_FACTOR = 10
THRESHOLD_DEFAULT = 0.0674

# 실제 데이터 샘플링 속도 (PHM 2010: 25600 Hz)
SAMPLING_RATE_HZ = 25600.0
# 실제 step 간격 계산: (STRIDE / (SAMPLING_RATE_HZ / DOWNSAMPLE_FACTOR)) * 1000 (밀리초)
REAL_TIME_STEP_MS = (STRIDE / (SAMPLING_RATE_HZ / DOWNSAMPLE_FACTOR)) * 1000  # 약 3.906 밀리초
# 실제 데이터 순서: [vx, vy, vz, sx, sy, sz, temp]
SENSOR_NAMES = ["Vib X", "Vib Y", "Vib Z", "Force X", "Force Y", "Force Z", "Temp"]
SENSOR_DESCRIPTIONS = {
    "Vib X": "X축 진동 (g)",
    "Vib Y": "Y축 진동 (g)",
    "Vib Z": "Z축 진동 (g)",
    "Force X": "X축 절삭력 (N)",
    "Force Y": "Y축 절삭력 (N)",
    "Force Z": "Z축 절삭력 (N)",
    "Temp": "온도 (℃)",
}


def configure_font() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    preferred = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    for font in preferred:
        if font in available:
            rcParams["font.family"] = font
            break
    rcParams["axes.unicode_minus"] = False


configure_font()


class CNCViewerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CNC AI 예지보전 시스템 (Enterprise Ver.)")
        self.root.geometry("1400x950")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Danger.TLabel", foreground="red", font=("Helvetica", 20, "bold"))
        style.configure("Normal.TLabel", foreground="green", font=("Helvetica", 20, "bold"))
        style.configure("KPI.TLabel", font=("Helvetica", 11))
        style.configure("Value.TLabel", font=("Helvetica", 16, "bold"))
        style.configure("Tab.TLabel", font=("Helvetica", 14, "bold"))

        self.is_running = False
        self.current_step = 0
        self.error_scores: np.ndarray | None = None
        self.feature_errors: np.ndarray | None = None
        self.y_max_limit = 0.1
        self.failure_step = 0
        self.cut_boundaries: list[int] = []  # 각 cut 파일의 시작 step 인덱스

        self.threshold = tk.DoubleVar(value=THRESHOLD_DEFAULT)
        # GUI 업데이트 최적화: 실제 step 간격 설정
        # GUI 오버헤드를 고려하여 실제 속도로 설정
        self.speed = max(1, int(REAL_TIME_STEP_MS))  # 약 3.9ms
        # GUI 업데이트 빈도 조절: 매 N step마다 한 번만 업데이트 (성능 향상)
        self.update_interval = 5  # 5 step마다 한 번만 GUI 업데이트
        self.status_var = tk.StringVar(value="준비 완료")

        # 비용 변수 (Tab 4용)
        self.cost_failure = tk.IntVar(value=5000)  # 고장 비용 (만원)
        self.cost_replace = tk.IntVar(value=200)  # 교체 비용 (만원)

        # [추가] 정상 기준값 계산 (Baseline Calculation)
        # c1_features.npy 파일이 있으면 그걸 읽어서 평균을 냄
        self.normal_baseline = np.full(7, 0.005)  # 기본값 (파일 없을 때 대비)
        self.calc_baseline()

        self.setup_ui()
        self.root.after(100, lambda: self.on_scenario_change(None))

    def setup_ui(self) -> None:
        # 1. 상단 공통 제어 패널
        control_frame = ttk.LabelFrame(self.root, text="시스템 제어", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(control_frame, text="시나리오:").pack(side="left", padx=5)
        self.combo_scenario = ttk.Combobox(control_frame, values=CONDITIONS, state="readonly", width=10)
        self.combo_scenario.current(0)  # test_fast 기본값
        self.combo_scenario.pack(side="left", padx=5)
        self.combo_scenario.bind("<<ComboboxSelected>>", self.on_scenario_change)

        ttk.Button(control_frame, text="▶ 시작", command=self.start_sim).pack(side="left", padx=20)
        ttk.Button(control_frame, text="⏸ 일시정지", command=self.pause_sim).pack(side="left")
        ttk.Button(control_frame, text="⏹ 초기화", command=self.reset_sim).pack(side="left", padx=5)

        ttk.Label(control_frame, text=" |  재생 속도:").pack(side="left", padx=10)
        # 실제 데이터 속도(약 3.9ms)를 기준으로 스케일 범위 설정
        # Tkinter Scale은 from_ < to 이어야 하므로, 빠른 속도(작은 값) ~ 느린 속도(큰 값) 순서
        # update_interval을 고려하여 실제 속도 범위 설정
        min_speed = max(1, int(REAL_TIME_STEP_MS / 10))  # 10배속
        max_speed = int(REAL_TIME_STEP_MS * 2)  # 0.5배속
        self.scale_speed = ttk.Scale(control_frame, from_=min_speed, to=max_speed, command=self.update_speed)
        self.scale_speed.set(self.speed)  # 기본값 사용
        self.scale_speed.pack(side="left", padx=5)
        
        # 실제 속도 버튼 추가
        ttk.Button(control_frame, text="실제 속도", command=self.set_real_time_speed).pack(side="left", padx=5)

        # 2. 탭 구성 (핵심)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # 탭 생성
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.tab4 = ttk.Frame(self.notebook)

        self.notebook.add(self.tab1, text="  🖥️ 실시간 모니터링  ")
        self.notebook.add(self.tab2, text="  🔮 예측 및 진단  ")
        self.notebook.add(self.tab3, text="  🔍 심층 분석 (Why?)  ")
        self.notebook.add(self.tab4, text="  💰 운영 최적화 (ROI)  ")

        # 각 탭 UI 구성 함수 호출
        self.setup_tab1()
        self.setup_tab2()
        self.setup_tab3()
        self.setup_tab4()

        # 하단 상태바
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w").pack(side="bottom", fill="x")

    # --- [Tab 1] 실시간 모니터링 ---
    def setup_tab1(self) -> None:
        # KPI 영역
        kpi_frame = ttk.Frame(self.tab1, padding=10)
        kpi_frame.pack(fill="x")

        self.lbl_cut = self.create_kpi_box(kpi_frame, "현재 작업 (Cut)", 0)
        self.lbl_score = self.create_kpi_box(kpi_frame, "현재 위험 점수", 1)
        self.lbl_status = self.create_kpi_box(kpi_frame, "장비 상태", 2, is_status=True)

        # 그래프 영역
        plot_frame = ttk.Frame(self.tab1)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.fig1 = Figure(figsize=(8, 4), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_title("Real-time Anomaly Trend")
        self.ax1.set_ylabel("Anomaly Score")
        self.ax1.grid(True, alpha=0.3)

        self.line_score, = self.ax1.plot([], [], "b-", lw=1.5)
        self.line_thresh = self.ax1.axhline(y=self.threshold.get(), color="r", ls="--", lw=2)

        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=plot_frame)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(fill="both", expand=True)

    # --- [Tab 2] 예측 및 진단 ---
    def setup_tab2(self) -> None:
        frame = ttk.Frame(self.tab2, padding=10)
        frame.pack(fill="both", expand=True)

        # 1. 왼쪽: 수치 정보
        left_panel = ttk.Frame(frame)
        left_panel.pack(side="left", fill="y", padx=20)

        ttk.Label(left_panel, text="예측 마모량 (Predicted Wear)", font=("Helvetica", 14)).pack(pady=(20, 10))
        self.lbl_wear_val = ttk.Label(left_panel, text="- mm", font=("Helvetica", 36, "bold"), foreground="#e67e22")
        self.lbl_wear_val.pack()

        ttk.Label(left_panel, text="교체 한계 (Limit)", font=("Helvetica", 12)).pack(pady=(20, 5))
        self.lbl_limit = ttk.Label(left_panel, text="0.20 mm", font=("Helvetica", 18, "bold"), foreground="red")
        self.lbl_limit.pack()

        # RUL 표시
        ttk.Label(left_panel, text="남은 수명 (RUL)", font=("Helvetica", 12)).pack(pady=(20, 5))
        self.lbl_rul_big = ttk.Label(left_panel, text="--- 회", font=("Helvetica", 24, "bold"), foreground="#2980b9")
        self.lbl_rul_big.pack()

        # 예측 메시지
        self.lbl_pred_msg = ttk.Label(
            left_panel, text="아직 충분히 사용할 수 있습니다.", font=("Helvetica", 12), foreground="green"
        )
        self.lbl_pred_msg.pack(pady=(20, 0))

        # 2. 오른쪽: 마모량 그래프 (Matplotlib)
        right_panel = ttk.Frame(frame)
        right_panel.pack(side="right", fill="both", expand=True)

        self.fig2 = Figure(figsize=(6, 4), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_title("Wear Degradation Trend")
        self.ax2.set_ylabel("Wear (mm)")
        self.ax2.set_xlabel("Time Step")
        self.ax2.grid(True, alpha=0.3)

        # 선 그리기
        self.line_wear, = self.ax2.plot([], [], color="#e67e22", linewidth=2, label="Predicted Wear")
        self.line_limit = self.ax2.axhline(y=0.2, color="r", linestyle="--", label="Limit (0.2mm)")
        self.ax2.legend(loc="upper left")
        self.ax2.set_ylim(0, 0.25)  # 0.25mm 까지 고정

        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=right_panel)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(fill="both", expand=True)

    # --- [Tab 3] 심층 분석 (기획서 3번 내용) ---
    def setup_tab3(self) -> None:
        content = ttk.Frame(self.tab3, padding=10)
        content.pack(fill="both", expand=True)

        # 상단: 바 차트 (센서 중요도)
        top_frame = ttk.Frame(content)
        top_frame.pack(fill="both", expand=True, side="top")

        self.fig3 = Figure(figsize=(10, 4), dpi=100)
        self.ax3_bar = self.fig3.add_subplot(121)
        self.ax3_bar.set_title("실시간 센서별 위험 기여도")
        self.bars = self.ax3_bar.bar(SENSOR_NAMES, [0] * 7, color="skyblue")
        self.ax3_bar.tick_params(axis="x", rotation=45)
        self.ax3_bar.set_ylim(0, 0.1)

        # 우측: 히트맵 (상관관계) - 단순화하여 에러값 변화 추이로 대체
        self.ax3_heat = self.fig3.add_subplot(122)
        self.ax3_heat.set_title("최근 50 step 센서 패턴 히트맵")
        # 초기 빈 이미지
        self.im_heat = self.ax3_heat.imshow(np.zeros((7, 50)), aspect="auto", cmap="hot", vmin=0, vmax=0.1)
        self.ax3_heat.set_yticks(range(7))
        self.ax3_heat.set_yticklabels(SENSOR_NAMES)

        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=top_frame)
        self.canvas3.draw()
        self.canvas3.get_tk_widget().pack(fill="both", expand=True)

        # 하단: 비교 테이블
        bottom_frame = ttk.LabelFrame(content, text="정상 vs 현재 상태 비교", padding=10)
        bottom_frame.pack(fill="x", side="bottom", pady=10)

        cols = ("항목", "정상 평균", "현재 값", "상태")
        self.tree = ttk.Treeview(bottom_frame, columns=cols, show="headings", height=7)
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", width=150)
        self.tree.pack(fill="x")

    # --- [Tab 4] 운영 최적화 (기획서 4번 내용) ---
    def setup_tab4(self) -> None:
        frame = ttk.Frame(self.tab4, padding=20)
        frame.pack(fill="both", expand=True)

        # 비용 입력 패널
        input_frame = ttk.LabelFrame(frame, text="비용 파라미터 설정 (단위: 만원)", padding=10)
        input_frame.pack(fill="x", pady=10)

        ttk.Label(input_frame, text="고장 시 손실 비용:").pack(side="left", padx=5)
        ttk.Entry(input_frame, textvariable=self.cost_failure, width=10).pack(side="left", padx=5)

        ttk.Label(input_frame, text="공구 교체 비용:").pack(side="left", padx=20)
        ttk.Entry(input_frame, textvariable=self.cost_replace, width=10).pack(side="left", padx=5)

        # ROI 결과 표시
        res_frame = ttk.Frame(frame, padding=20)
        res_frame.pack(fill="both", expand=True)

        ttk.Label(res_frame, text="예상 절감 비용 (ROI)", font=("Helvetica", 14)).pack()
        self.lbl_roi = ttk.Label(res_frame, text="0 만원", font=("Helvetica", 36, "bold"), foreground="#27ae60")
        self.lbl_roi.pack(pady=10)

        ttk.Label(res_frame, text="AI 교체 제안", font=("Helvetica", 14)).pack(pady=(20, 0))
        self.lbl_advice = ttk.Label(res_frame, text="-", font=("Helvetica", 24, "bold"), foreground="#e67e22")
        self.lbl_advice.pack(pady=10)

    # --- 헬퍼 함수 ---
    def create_kpi_box(self, parent: ttk.Frame, title: str, col: int, is_status: bool = False) -> ttk.Label:
        f = ttk.Frame(parent, borderwidth=2, relief="groove")
        f.pack(side="left", expand=True, fill="both", padx=5)
        ttk.Label(f, text=title, style="KPI.TLabel").pack(pady=5)
        style = "Normal.TLabel" if is_status else "Value.TLabel"
        lbl = ttk.Label(f, text="대기" if is_status else "-", style=style)
        lbl.pack(pady=5)
        return lbl

    def update_speed(self, val: str) -> None:
        self.speed = int(float(val))
    
    def set_real_time_speed(self) -> None:
        """실제 데이터 샘플링 속도로 설정"""
        self.speed = max(1, int(REAL_TIME_STEP_MS))
        self.scale_speed.set(self.speed)
        self.status_var.set(f"재생 속도: 실제 속도 ({REAL_TIME_STEP_MS:.2f}ms/step, {self.update_interval} step마다 GUI 업데이트)")

    # [추가] 기준값 계산 함수
    def calc_baseline(self) -> None:
        """c1(정상 데이터)의 평균 에러값을 계산하여 정상 기준값으로 설정"""
        c1_path = Path(RESULT_DIR) / "c1_features.npy"
        if c1_path.exists():
            try:
                # c1 데이터 전체의 평균을 '정상 기준'으로 잡음
                c1_data = np.load(c1_path)
                self.normal_baseline = np.mean(c1_data, axis=0)  # 각 센서별 평균
                print(f"정상 기준값 계산 완료: {self.normal_baseline}")
            except Exception as e:
                print(f"Warning: 정상 기준값 계산 실패: {e}, 기본값(0.005) 사용")
                self.normal_baseline = np.full(7, 0.005)
        else:
            print(f"Warning: {c1_path} 파일 없음, 기본값(0.005) 사용")
            self.normal_baseline = np.full(7, 0.005)

    # --- 로직 함수 ---
    def on_scenario_change(self, _event) -> None:
        scenario = self.combo_scenario.get()
        score_path = Path(RESULT_DIR) / f"{scenario}.npy"
        feature_path = Path(RESULT_DIR) / f"{scenario}_features.npy"

        if not feature_path.exists():
            if score_path.exists():
                self.error_scores = np.load(score_path)
                self.feature_errors = np.tile(self.error_scores[:, np.newaxis], (1, 7))
            else:
                messagebox.showerror("Error", "결과 파일 없음")
                return
        else:
            self.error_scores = np.load(score_path) if score_path.exists() else None
            self.feature_errors = np.load(feature_path)
            if self.error_scores is None:
                self.error_scores = self.feature_errors.mean(axis=1)

        # Y축 고정 및 고장 시점 찾기
        max_val = np.max(self.feature_errors)
        self.y_max_limit = max(max_val * 1.2, 0.01)
        if hasattr(self, "ax3_bar"):
            self.ax3_bar.set_ylim(0, self.y_max_limit)

        # 고장 시점 탐색
        thresh = self.threshold.get()
        danger = np.where(self.error_scores > thresh)[0]
        self.failure_step = danger[0] if len(danger) > 0 else len(self.error_scores)

        # [추가] cut 경계 정보 로드
        boundaries_path = Path(RESULT_DIR) / f"{scenario}_cut_boundaries.json"
        if boundaries_path.exists():
            try:
                with open(boundaries_path, "r") as f:
                    data = json.load(f)
                    self.cut_boundaries = data.get("cut_boundaries", [])
            except Exception as e:
                print(f"Warning: cut 경계 정보 로드 실패: {e}, 기본값 사용")
                self.cut_boundaries = []
        else:
            # 경계 정보가 없으면 빈 리스트로 설정 (기존 방식으로 fallback)
            self.cut_boundaries = []

        # [추가] 마모량 데이터 로드 - 강제 우상향 적용
        wear_path = Path(RESULT_DIR) / f"{scenario}_wear.npy"
        if wear_path.exists():
            try:
                raw_preds = np.load(wear_path)
                
                # 1. 튀는 값 잡기 (초반 진입 충격 제거)
                # 시작하자마자 값이 확 튀는 경우가 있어서, 앞부분 20개를 0으로 눌러줍니다.
                if len(raw_preds) > 20:
                    raw_preds[:20] = 0

                # 2. 강력한 스무딩 (Smoothing)
                # 꼬불꼬불한 것을 다림질하듯 펴줍니다. (window를 100으로 늘림)
                smoothed = pd.Series(raw_preds).rolling(window=100, min_periods=1).mean().values

                # 3. [핵심] 누적 최대값 적용 (Cumulative Max)
                # "현재 값이 과거의 최대값보다 작으면, 과거 최대값으로 강제 고정"
                # 즉, 그래프가 절대 아래로 내려가지 않게 만듭니다.
                self.wear_preds = np.maximum.accumulate(smoothed)
                
                # 4. (선택사항) 최소값 보정
                # 마모량이 음수가 나오지 않게 0.0 이상으로 자름
                self.wear_preds = np.maximum(self.wear_preds, 0.0)
                
            except Exception:
                self.wear_preds = np.zeros_like(self.error_scores)  # 없으면 0으로 채움
        else:
            self.wear_preds = np.zeros_like(self.error_scores)  # 없으면 0으로 채움

        self.current_step = 0
        self.is_running = False
        self.update_gui_once()
        self.status_var.set(f"{scenario} 로드 완료.")

    def start_sim(self) -> None:
        if not self.is_running and self.error_scores is not None:
            self.is_running = True
            self.run_loop()

    def pause_sim(self) -> None:
        self.is_running = False
        self.status_var.set("일시정지")

    def reset_sim(self) -> None:
        self.is_running = False
        self.current_step = 0
        self.update_gui_once()
        self.status_var.set("초기화됨")

    def run_loop(self) -> None:
        if self.is_running and self.error_scores is not None:
            if self.current_step < len(self.error_scores):
                # GUI 업데이트 최적화: 매 update_interval step마다 한 번만 업데이트
                if self.current_step % self.update_interval == 0:
                    self.update_gui_once()
                else:
                    # GUI 업데이트 없이 step만 진행 (데이터는 계속 진행)
                    pass
                self.current_step += 1
                # step 간격을 update_interval로 나눠서 실제 속도 유지
                self.root.after(self.speed * self.update_interval, self.run_loop)
            else:
                self.is_running = False
                self.status_var.set("시뮬레이션 종료")

    def update_gui_once(self) -> None:
        if self.error_scores is None or len(self.error_scores) == 0:
            return
        idx = self.current_step
        if idx >= len(self.error_scores):
            return

        score = self.error_scores[idx]
        features = self.feature_errors[idx] if self.feature_errors is not None and len(self.feature_errors) > idx else np.zeros(7)
        thresh = self.threshold.get()

        # RUL 계산 (추세 기반 예측)
        history_window = 50
        if idx > history_window:
            recent_scores = self.error_scores[idx - history_window : idx]
            recent_x = np.arange(len(recent_scores))
            if np.max(recent_scores) < thresh:
                fit = np.polyfit(recent_x, recent_scores, 1)
                slope = fit[0]
                intercept = fit[1]
                if slope > 0.00001:
                    steps_to_failure = (thresh - intercept) / slope
                    remaining_steps_pred = steps_to_failure - history_window
                    remaining_steps_pred = min(remaining_steps_pred, 5000)
                    rul_cuts = int((remaining_steps_pred * STRIDE * DOWNSAMPLE_FACTOR) / SEQ_LEN)
                else:
                    rul_cuts = 9999
            else:
                rul_cuts = 0
        else:
            rul_cuts = 9999

        # 고장 시점 기반 RUL (백업)
        remaining = self.failure_step - idx
        if remaining < 0:
            remaining = 0
        rul_cuts_backup = int((remaining * STRIDE * DOWNSAMPLE_FACTOR) / SEQ_LEN)
        if rul_cuts == 9999:
            rul_cuts = rul_cuts_backup

        # 1. Tab 1 업데이트 (모니터링)
        # 실제 cut 번호 계산 (경계 정보 사용)
        if self.cut_boundaries and len(self.cut_boundaries) > 0:
            # 경계 정보가 있으면 현재 step 인덱스가 어느 cut에 속하는지 찾기
            # cut_boundaries[i]는 i+1번째 cut의 시작 step 인덱스
            cut_num = len(self.cut_boundaries)  # 기본값: 마지막 cut
            for i in range(len(self.cut_boundaries) - 1, -1, -1):  # 역순으로 검색
                if idx >= self.cut_boundaries[i]:
                    cut_num = i + 1
                    break
        else:
            # 경계 정보가 없으면 기존 방식 사용 (하위 호환성)
            approx_cut = int((idx * STRIDE * DOWNSAMPLE_FACTOR) / SEQ_LEN) + 1
            cut_num = approx_cut
        
        self.lbl_cut.config(text=f"#{cut_num}")
        self.lbl_score.config(text=f"{score:.4f}")

        is_danger = score > thresh
        if is_danger:
            self.lbl_status.config(text="🚨 위험", style="Danger.TLabel")
        else:
            self.lbl_status.config(text="✅ 정상", style="Normal.TLabel")

        window = 200
        start = max(0, idx - window)
        self.line_score.set_data(np.arange(start, idx + 1), self.error_scores[start : idx + 1])
        self.line_thresh.set_ydata([thresh] * 2)
        self.ax1.set_xlim(start, max(start + window, idx + 10))
        current_max = np.max(self.error_scores[start : idx + 1]) if idx > start else thresh
        self.ax1.set_ylim(0, max(current_max, thresh) * 1.2)
        self.canvas1.draw_idle()

        # 2. Tab 2 업데이트 (예측)
        self.lbl_rul_big.config(text=f"{rul_cuts} 회" if rul_cuts < 9999 else "안정")

        # 마모량 값 및 그래프 업데이트
        if hasattr(self, "wear_preds") and idx < len(self.wear_preds):
            curr_wear = self.wear_preds[idx]

            # 수치 업데이트
            self.lbl_wear_val.config(text=f"{curr_wear:.3f} mm")

            # 그래프 업데이트 (최근 300개)
            window = 300
            start = max(0, idx - window)
            self.line_wear.set_data(np.arange(start, idx + 1), self.wear_preds[start : idx + 1])
            self.line_limit.set_ydata([0.2] * 2)  # 한계치 0.2mm 가정
            self.ax2.set_xlim(start, max(start + window, idx + 10))
            self.canvas2.draw_idle()

            # 예측 메시지 (마모량 기반)
            wear_limit = 0.2
            remaining_wear = wear_limit - curr_wear
            if curr_wear >= wear_limit:
                msg = "즉시 가동을 중단하고 교체하세요!"
                msg_color = "red"
            elif remaining_wear < 0.05:
                msg = f"마모가 심합니다. (남은 여유: {remaining_wear:.3f}mm)"
                msg_color = "red"
            elif remaining_wear < 0.1:
                msg = f"주의 필요. (남은 여유: {remaining_wear:.3f}mm)"
                msg_color = "#f39c12"
            else:
                msg = f"안정적입니다. (남은 여유: {remaining_wear:.3f}mm)"
                msg_color = "green"
        else:
            self.lbl_wear_val.config(text="- mm")
            msg = "마모량 데이터 없음"
            msg_color = "gray"

        self.lbl_pred_msg.config(text=msg, foreground=msg_color)

        # 3. Tab 3 업데이트 (분석)
        # 바 차트 - 표와 기준 통일 (정상 평균 * 3)
        for i, (rect, h) in enumerate(zip(self.bars, features)):
            rect.set_height(h)
            
            # 해당 센서의 정상 기준값 가져오기
            baseline_val = self.normal_baseline[i]
            
            # 표와 동일한 로직 적용 (3배 넘으면 위험)
            if h > baseline_val * 3:
                rect.set_color("#e74c3c")  # 빨강 (위험)
            else:
                rect.set_color("skyblue")  # 파랑 (정상)

        # 히트맵 (최근 50개 데이터의 전치행렬)
        heat_start = max(0, idx - 50)
        heat_data = self.feature_errors[heat_start:idx].T if self.feature_errors is not None else np.zeros((7, 0))
        if heat_data.shape[1] > 0:
            # 크기가 계속 변하면 안되므로 0으로 패딩
            padded = np.zeros((7, 50))
            padded[:, : heat_data.shape[1]] = heat_data
            self.im_heat.set_data(padded)
            self.im_heat.set_clim(0, self.y_max_limit)
        self.canvas3.draw_idle()

        # 테이블 업데이트
        self.tree.delete(*self.tree.get_children())
        
        for i, name in enumerate(SENSOR_NAMES):
            curr_val = features[i]  # 현재 값
            norm_val = self.normal_baseline[i]  # 진짜 정상 평균 값
            
            # 상태 판단: 정상이면 초록색, 높으면 경고
            # (보통 정상 평균의 2~3배를 넘어가면 주의 단계로 봅니다)
            if curr_val > norm_val * 3:
                status = "⚠️ 높음"
            else:
                status = "정상"
            
            # 테이블에 삽입 (소수점 4자리까지 예쁘게)
            self.tree.insert("", "end", values=(name, f"{norm_val:.4f}", f"{curr_val:.4f}", status))

        # 4. Tab 4 업데이트 (최적화)
        # 절감 비용 계산: (고장비용 - 교체비용) * (진행률) -> 단순히 시뮬레이션용 수식
        # 실제로는: 고장을 막았을 때의 기회비용
        
        # 안전장치: 입력값이 비어있거나 잘못된 경우 기본값 사용
        try:
            c_fail = self.cost_failure.get()
            c_repl = self.cost_replace.get()
        except tk.TclError:
            # 입력값이 잘못되었거나 비어있으면 기본값 사용
            c_fail = 5000
            c_repl = 200
        
        if is_danger:
            saved = 0  # 이미 고장남
            advice = "교체 시기 놓침 (손실 발생)"
            color = "red"
        else:
            saved = c_fail - c_repl
            if rul_cuts < 30:
                advice = "🔥 지금 교체하세요 (최적)"
                color = "red"
            elif rul_cuts < 100:
                advice = "교체 준비 (예비품 확인)"
                color = "#f39c12"  # orange
            else:
                advice = "계속 사용 가능"
                color = "green"

        self.lbl_roi.config(text=f"{saved:,} 만원")
        self.lbl_advice.config(text=advice, foreground=color)


if __name__ == "__main__":
    root = tk.Tk()
    app = CNCViewerApp(root)
    root.mainloop()
