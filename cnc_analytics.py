import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import font_manager, rcParams
import numpy as np
import os
from pathlib import Path
import pandas as pd


def configure_font() -> None:
    """한글 폰트 설정"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    preferred = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    for font in preferred:
        if font in available:
            rcParams["font.family"] = font
            break
    rcParams["axes.unicode_minus"] = False


configure_font()

# --- 설정 ---
RESULT_DIR = "artifacts/results"
CONDITIONS = ["c1", "c4", "c6", "c2", "c3", "c5"]
LABELS = ["Train", "Train", "Train", "Test", "Test", "Test"]  # 구분용
COLORS = ["#3498db", "#3498db", "#3498db", "#e67e22", "#e67e22", "#e67e22"]  # 파랑(Train), 주황(Test)
THRESHOLD = 0.0674
COST_FAILURE = 200  # 만원
COST_REPLACE = 5  # 만원


class CNCAnalyticsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CNC 전체 데이터셋 종합 분석 리포트")
        self.root.geometry("1400x900")

        self.data_summary = []
        self.load_all_data()
        self.setup_ui()

    def load_all_data(self):
        """모든 npy 파일을 읽어서 통계 요약"""
        for cond, label in zip(CONDITIONS, LABELS):
            path = Path(RESULT_DIR) / f"{cond}.npy"
            if path.exists():
                try:
                    scores = np.load(path)
                    
                    # 데이터 검증
                    if len(scores) == 0:
                        print(f"Warning: {cond} 데이터가 비어있음, 건너뜀")
                        continue
                    
                    # 1. 수명 (Life): 임계값 넘는 시점
                    danger_zone = np.where(scores > THRESHOLD)[0]
                    life_step = danger_zone[0] if len(danger_zone) > 0 else len(scores)
                    
                    # 2. 최대 위험도
                    max_score = np.max(scores)
                    
                    # 3. 평균 위험도 (추가 통계)
                    avg_score = np.mean(scores)
                    
                    # 4. 비용 절감 (개선된 로직)
                    # 수명에 따라 절감액 차등 적용
                    # - 조기 감지(수명의 80% 이전): 최대 절감
                    # - 중간 감지(80-95%): 중간 절감
                    # - 늦은 감지(95% 이후): 최소 절감
                    # - 미감지: 손실
                    if len(danger_zone) > 0:
                        life_ratio = life_step / len(scores)
                        if life_ratio < 0.8:
                            saved = COST_FAILURE - COST_REPLACE  # 최대 절감
                            status = "성공 (조기 감지)"
                        elif life_ratio < 0.95:
                            saved = int((COST_FAILURE - COST_REPLACE) * 0.7)  # 70% 절감
                            status = "성공 (중간 감지)"
                        else:
                            saved = int((COST_FAILURE - COST_REPLACE) * 0.3)  # 30% 절감
                            status = "주의 (늦은 감지)"
                    else:
                        life_ratio = 1.0  # 미감지 시 수명 비율 100%
                        saved = 0
                        status = "실패 (미감지)"

                    self.data_summary.append(
                        {
                            "Condition": cond,
                            "Type": label,
                            "Total Steps": len(scores),
                            "Predicted Life": life_step,
                            "Life Ratio": life_ratio if len(danger_zone) > 0 else 1.0,
                            "Max Score": max_score,
                            "Avg Score": avg_score,
                            "ROI": saved,
                            "Status": status,
                        }
                    )
                except Exception as e:
                    print(f"Error loading {cond}: {e}")
                    continue
        
        # 데이터 검증
        if len(self.data_summary) == 0:
            messagebox.showerror("Error", "분석할 데이터가 없습니다.\n먼저 generate_results.py를 실행하세요.")
            return

    def setup_ui(self):
        # 타이틀
        ttk.Label(
            self.root, text="📊 CNC 데이터셋 종합 분석 리포트", font=("Helvetica", 20, "bold")
        ).pack(pady=10)

        # 탭 구성
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        tab1 = ttk.Frame(notebook)
        tab2 = ttk.Frame(notebook)

        notebook.add(tab1, text="  📈 성능 및 수명 비교  ")
        notebook.add(tab2, text="  💰 경제적 가치 (ROI)  ")

        self.setup_tab1(tab1)
        self.setup_tab2(tab2)

    def setup_tab1(self, parent):
        """성능 및 수명 비교 탭"""
        # 데이터 검증
        if len(self.data_summary) == 0:
            ttk.Label(parent, text="데이터가 없습니다.", font=("Helvetica", 16)).pack(pady=50)
            return
        
        # 2x2 그리드 그래프
        fig = Figure(figsize=(10, 8), dpi=100)

        # 데이터 준비
        df = pd.DataFrame(self.data_summary)
        x = df["Condition"]

        # 1. 공구 수명 비교 (Bar Chart)
        ax1 = fig.add_subplot(221)
        bars = ax1.bar(x, df["Predicted Life"], color=COLORS)
        ax1.set_title("공구별 예측 수명 (Life Duration)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Time Steps")
        ax1.grid(axis="y", alpha=0.3)
        # 평균선
        avg_life = df["Predicted Life"].mean()
        ax1.axhline(avg_life, color="red", linestyle="--", label=f"Avg: {int(avg_life)}")
        ax1.legend()
        # 값 표시
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

        # 2. 최대 이상 점수 비교 (Scatter)
        ax2 = fig.add_subplot(222)
        scatter = ax2.scatter(x, df["Max Score"], s=100, c=COLORS, alpha=0.7)
        ax2.axhline(THRESHOLD, color="r", linestyle="--", label="Threshold")
        ax2.set_title("공구별 최대 위험도 (Max Anomaly Score)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Score")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        # 값 표시
        for i, (xi, yi) in enumerate(zip(x, df["Max Score"])):
            ax2.text(xi, yi, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)

        # 3. 데이터 길이 vs 수명 (Scatter) - 상관관계
        ax3 = fig.add_subplot(223)
        ax3.scatter(df["Total Steps"], df["Predicted Life"], c=COLORS, s=100, alpha=0.7)
        ax3.set_title("전체 데이터 길이 vs 예측 수명", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Actual Data Length")
        ax3.set_ylabel("Predicted Life")
        ax3.grid(True)
        # 대각선 (y=x, 즉 완벽하게 끝까지 쓴 경우)
        lims = [0, max(df["Total Steps"].max(), df["Predicted Life"].max())]
        ax3.plot(lims, lims, "k--", alpha=0.5, label="Ideal")
        ax3.legend()
        # 레이블 표시
        for i, row in df.iterrows():
            ax3.text(
                row["Total Steps"],
                row["Predicted Life"],
                row["Condition"],
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # 4. 요약 텍스트
        ax4 = fig.add_subplot(224)
        ax4.axis("off")
        
        train_df = df[df["Type"] == "Train"]
        test_df = df[df["Type"] == "Test"]
        
        summary_text = (
            f"총 분석 대상: {len(df)}개 (Train {len(train_df)} / Test {len(test_df)})\n\n"
            f"평균 공구 수명: {int(avg_life)} steps\n"
            f"평균 위험도: {df['Max Score'].mean():.4f}\n"
            f"최장 수명: {df['Predicted Life'].max()} steps ({df.loc[df['Predicted Life'].idxmax(), 'Condition']})\n"
            f"최단 수명: {df['Predicted Life'].min()} steps ({df.loc[df['Predicted Life'].idxmin(), 'Condition']})\n\n"
            f"Train 평균 수명: {int(train_df['Predicted Life'].mean()) if len(train_df) > 0 else 0} steps\n"
            f"Test 평균 수명: {int(test_df['Predicted Life'].mean()) if len(test_df) > 0 else 0} steps\n\n"
            f"* 파란색: 학습 데이터 (Normal)\n"
            f"* 주황색: 테스트 데이터 (Test)"
        )
        ax4.text(0.1, 0.5, summary_text, fontsize=12, va="center")

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def setup_tab2(self, parent):
        """경제적 가치 (ROI) 탭"""
        # 데이터 검증
        if len(self.data_summary) == 0:
            ttk.Label(parent, text="데이터가 없습니다.", font=("Helvetica", 16)).pack(pady=50)
            return
        
        # 상단: 전체 ROI 요약 박스
        top_frame = ttk.Frame(parent, padding=20)
        top_frame.pack(fill="x")

        df = pd.DataFrame(self.data_summary)
        total_saved = df["ROI"].sum()
        success_count = len(df[df["Status"].str.contains("성공", na=False)])
        early_count = len(df[df["Status"] == "성공 (조기 감지)"])

        lbl_roi = ttk.Label(
            top_frame,
            text=f"총 예상 절감 비용: {total_saved:,} 만원",
            font=("Helvetica", 32, "bold"),
            foreground="green",
        )
        lbl_roi.pack()
        ttk.Label(
            top_frame,
            text=f"(6개 공구 전체 적용 시 시뮬레이션 결과 | 성공: {success_count}개 (조기: {early_count}개))",
            font=("Helvetica", 12),
        ).pack(pady=5)

        # ROI 상세 정보
        info_frame = ttk.Frame(parent, padding=10)
        info_frame.pack(fill="x", padx=20, pady=10)

        ttk.Label(
            info_frame,
            text=f"• 고장 시 손실 비용: {COST_FAILURE:,} 만원/건",
            font=("Helvetica", 11),
        ).pack(anchor="w")
        ttk.Label(
            info_frame,
            text=f"• 조기 교체 비용: {COST_REPLACE:,} 만원/건",
            font=("Helvetica", 11),
        ).pack(anchor="w")
        ttk.Label(
            info_frame,
            text=f"• 공구당 평균 절감액: {total_saved / len(df):.0f} 만원",
            font=("Helvetica", 11),
        ).pack(anchor="w")
        ttk.Label(
            info_frame,
            text=f"• 성공률: {success_count / len(df) * 100:.1f}%",
            font=("Helvetica", 11),
        ).pack(anchor="w")

        # 하단: 상세 테이블
        cols = ("Condition", "Type", "Status", "Max Score", "예측 수명", "수명 비율", "절감액(만원)")
        tree = ttk.Treeview(parent, columns=cols, show="headings", height=10)

        for col in cols:
            tree.heading(col, text=col)
            if col == "Status":
                tree.column(col, anchor="center", width=150)
            else:
                tree.column(col, anchor="center", width=120)

        for item in self.data_summary:
            life_ratio = item.get("Life Ratio", 1.0)
            tree.insert(
                "",
                "end",
                values=(
                    item["Condition"],
                    item["Type"],
                    item["Status"],
                    f"{item['Max Score']:.4f}",
                    item["Predicted Life"],
                    f"{life_ratio:.1%}",
                    item["ROI"],
                ),
            )

        tree.pack(fill="both", expand=True, padx=20, pady=20)


if __name__ == "__main__":
    root = tk.Tk()
    app = CNCAnalyticsApp(root)
    root.mainloop()

