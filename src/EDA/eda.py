"""
Phase 3: Module khám phá dữ liệu (EDA) và trực quan hóa phân phối.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_eda_reports(df: pd.DataFrame, output_dir: Path) -> dict:
    """Tạo báo cáo thống kê mô tả và lưu biểu đồ phân phối."""
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    desc = df[numeric_cols].describe().to_dict()

    try:
        # 1. Biểu đồ phân phối Histogram & Boxplot
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for idx, col in enumerate(numeric_cols):
            sns.histplot(df[col], kde=True, ax=axes[idx], color="#3b82f6")
            axes[idx].set_title(f"Phân phối {col}", fontsize=11, fontweight="bold")
            axes[idx].grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "distributions.png", dpi=150)
        plt.close()

        # 2. Scatter plot khám phá phân cụm tự nhiên
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=df,
            x="Annual Income (k$)",
            y="Spending Score (1-100)",
            hue="Gender",
            palette={"Male": "#2563eb", "Female": "#ec4899"},
            s=60,
            ax=ax
        )
        ax.set_title("Thu Nhập Hàng Năm vs Điểm Chi Tiêu (Raw)", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "raw_scatter.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"[!] Bỏ qua lưu biểu đồ EDA: {e}")

    return desc

