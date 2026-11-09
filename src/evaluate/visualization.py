"""
Phase 22: Module trực quan hóa kết quả phân cụm và lưu file ảnh ra outputs/figures/.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def export_clustering_figures(
    df_selection: pd.DataFrame,
    X_raw: np.ndarray,
    labels: np.ndarray,
    centroids_raw: np.ndarray,
    df_profile: pd.DataFrame,
    output_dir: Path
):
    """Xuất 4 biểu đồ trực quan hóa cao cấp theo yêu cầu README."""
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]

    # 1. Elbow Curve
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df_selection["K"], df_selection["WCSS"], marker="o", color="#2563eb", linewidth=2)
    ax.axvline(x=5, color="#ef4444", linestyle="--", label="Điểm khuỷu tay K=5")
    ax.set_title("Phương Pháp Khuỷu Tay (Elbow Method)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Số lượng cụm (K)")
    ax.set_ylabel("Inertia / WCSS")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "elbow_curve.png", dpi=150)
    plt.close()

    # 2. Silhouette Curve
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df_selection["K"], df_selection["Silhouette_Score"], marker="s", color="#10b981", linewidth=2)
    ax.axvline(x=5, color="#ef4444", linestyle="--", label="Đỉnh cao nhất K=5")
    ax.set_title("Đường Cong Hệ Số Silhouette Theo K", fontsize=12, fontweight="bold")
    ax.set_xlabel("Số lượng cụm (K)")
    ax.set_ylabel("Silhouette Score trung bình")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "silhouette_curve.png", dpi=150)
    plt.close()

    # 3. Clusters 2D Scatter Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    for k in range(len(centroids_raw)):
        pts = X_raw[labels == k]
        ax.scatter(pts[:, 0], pts[:, 1], s=55, color=palette[k % len(palette)], label=f"Cụm {k}", alpha=0.8)
    ax.scatter(
        centroids_raw[:, 0], centroids_raw[:, 1],
        s=250, c="#000000", marker="X", edgecolors="#ffffff", linewidth=2, label="Centroids", zorder=10
    )
    ax.set_title("Phân Cụm Khách Hàng 2D (K-Means K=5)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Thu nhập hàng năm (Annual Income k$)")
    ax.set_ylabel("Điểm chi tiêu (Spending Score 1-100)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "clusters_2d.png", dpi=150)
    plt.close()

    # 4. Cluster Profile Bar Chart
    fig, ax = plt.subplots(figsize=(10, 4.5))
    df_plot = df_profile[["Cluster", "Annual Income (k$)", "Spending Score (1-100)"]].set_index("Cluster")
    df_plot.plot(kind="bar", ax=ax, colormap="viridis", width=0.7)
    ax.set_title("So Sánh Đặc Trưng Trung Bình Giữa Các Phân Khúc Khách Hàng", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mã phân cụm (Cluster)")
    ax.set_ylabel("Giá trị trung bình")
    ax.set_xticklabels([f"Cụm {c}" for c in df_plot.index], rotation=0)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_profile.png", dpi=150)
    plt.close()

