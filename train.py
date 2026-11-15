"""
Pipeline Huấn Luyện K-Means Clustering Phân Khúc Khách Hàng (Zero Scikit-Learn).
Thực thi toàn diện các Phase từ Đọc dữ liệu, Tiền xử lý, Quét K đến Xuất mô hình.
"""
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

import config
try:
    from models.algorithms.preprocessing import Scaler
    from models.algorithms.kmeans import KMeansScratch
except ImportError:
    from src.Preprocessing.scaler import Scaler
    from src.training.kmeans import KMeansScratch
from src.EDA.data_loader import load_dataset
from src.EDA.validation import validate_dataset
from src.EDA.eda import generate_eda_reports
from src.Preprocessing.cleaning import clean_customer_data
from src.Preprocessing.feature_selection import extract_features
from src.evaluate.metrics import silhouette_score
from src.evaluate.model_selection import evaluate_k_range
from src.evaluate.profiling import build_cluster_profiles
from src.evaluate.visualization import export_clustering_figures


def main():
    print("=" * 70)
    print("LAB 3: PIPELINE K-MEANS CLUSTERING (100% FROM SCRATCH BẰNG NUMPY)")
    print("=" * 70)

    # 1. Nạp và kiểm tra tính toàn vẹn dữ liệu
    df_raw = load_dataset(config.RAW_DATA_PATH)
    val_report = validate_dataset(df_raw)
    print(f"[*] Nạp {val_report['n_samples']} dòng dữ liệu từ: {config.RAW_DATA_PATH.name}")
    print(f"[*] Missing: {val_report['missing_values']} | Duplicates: {val_report['duplicate_rows']}")

    # 2. EDA và làm sạch dữ liệu
    generate_eda_reports(df_raw, config.EDA_PLOTS_DIR)
    df_clean = clean_customer_data(df_raw)

    # 3. Trích xuất đặc trưng và Chuẩn hóa (Zero Data Leakage)
    X_raw, feature_cols = extract_features(df_clean, config.FEATURE_COLS)
    scaler = Scaler(method=config.SCALING_METHOD)
    X_scaled = scaler.fit_transform(X_raw)

    config.PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X_scaled, columns=feature_cols).to_csv(config.PROCESSED_DATA_PATH, index=False)
    print(f"[+] Đã lưu dữ liệu chuẩn hóa: {config.PROCESSED_DATA_PATH.name}")

    # 4. Quét K [2..10] để tìm K tối ưu
    print("\n" + "=" * 70)
    print("QUÉT SỐ CỤM K TỪ 2 ĐẾN 10 (ELBOW & SILHOUETTE SCORE)")
    print("=" * 70)
    df_selection = evaluate_k_range(
        X_scaled,
        k_min=config.K_MIN,
        k_max=config.K_MAX,
        random_state=config.RANDOM_STATE,
        n_init=config.N_INIT
    )
    print(df_selection.to_string(index=False))

    # 5. Huấn luyện mô hình tối ưu K = 5
    best_k = config.OPTIMAL_K
    print(f"\n[*] Tiến hành huấn luyện mô hình tối ưu với K = {best_k}...")
    model = KMeansScratch(
        n_clusters=best_k,
        init="kmeans++",
        n_init=config.N_INIT,
        max_iter=config.MAX_ITERS,
        tol=config.TOL,
        random_state=config.RANDOM_STATE
    )
    labels = model.fit_predict(X_scaled)
    final_sil = silhouette_score(X_scaled, labels)
    print(f"[+] Hội tụ sau {model.n_iter_} vòng lặp | Inertia: {model.inertia_:.4f} | Silhouette: {final_sil:.4f}")

    # 6. Khôi phục Centroids về thang đo gốc & Phân tích chân dung (Persona)
    centroids_raw = scaler.inverse_transform(model.cluster_centers_)
    df_profile = build_cluster_profiles(df_clean, labels, feature_cols)
    print("\n" + "=" * 70 + "\nBẢNG HỒ SƠ PHÂN KHÚC KHÁCH HÀNG (CUSTOMER PERSONA)\n" + "=" * 70)
    print(df_profile.to_string(index=False))

    # 7. Xuất file báo cáo số liệu và trực quan hóa
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_selection.to_csv(config.REPORTS_DIR / "model_selection.csv", index=False)
    df_profile.to_csv(config.REPORTS_DIR / "cluster_profile.csv", index=False)

    metrics_info = {
        "best_k": best_k,
        "inertia": round(model.inertia_, 4),
        "silhouette_score": round(final_sil, 4),
        "iterations": model.n_iter_,
        "converged": bool(model.converged_)
    }
    with open(config.REPORTS_DIR / "clustering_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_info, f, indent=4, ensure_ascii=False)

    df_clustered = df_clean.copy()
    df_clustered["Cluster"] = labels
    df_clustered.to_csv(config.CLUSTERED_DATA_PATH, index=False)

    export_clustering_figures(
        df_selection, X_raw, labels, centroids_raw, df_profile, config.FIGURES_DIR
    )
    print(f"[+] Đã xuất 4 biểu đồ cao cấp vào: {config.FIGURES_DIR}")

    # 8. Đóng gói mô hình và Scaler Artifact
    config.SAVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_artifact = {
        "metadata": {
            "model_name": "KMeansScratch",
            "optimal_k": best_k,
            "feature_columns": feature_cols,
            "silhouette_score": round(final_sil, 4),
            "inertia": round(model.inertia_, 4),
            "scaling_method": config.SCALING_METHOD
        },
        "scaler_params": scaler.get_params(),
        "centroids_scaled": model.cluster_centers_.tolist(),
        "centroids_raw": centroids_raw.tolist(),
        "cluster_personas": df_profile["Persona"].tolist()
    }
    # Lưu JSON Artifact vào save_model/
    with open(config.MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(model_artifact, f, indent=4, ensure_ascii=False)

    # Lưu Pickle Artifacts (.pkl) tương đồng kiến trúc Lab 2
    model.save(str(config.MODEL_PKL_PATH))
    scaler.save(str(config.SCALER_PKL_PATH))
    print(f"[+] Đã lưu mô hình và Scaler vào: {config.SAVE_MODEL_DIR}")

    rep_dir = config.BASE_DIR / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    with open(rep_dir / "evaluation_results.md", "w", encoding="utf-8") as f:
        f.write("# BÁO CÁO ĐÁNH GIÁ MÔ HÌNH — LAB 3 (K-MEANS CLUSTERING)\n\n")
        f.write(f"- **Thuật toán:** `KMeansScratch` (100% From-Scratch)\n")
        f.write(f"- **Số cụm tối ưu (K):** `{best_k}`\n")
        f.write(f"- **Inertia (WCSS):** `{model.inertia_:.4f}`\n")
        f.write(f"- **Silhouette Score:** `{final_sil:.4f}`\n\n")
        f.write("### Bảng Hồ Sơ Phân Khúc Khách Hàng (Cluster Personas)\n\n")
        f.write("| Cụm | Thu nhập TB (k$) | Chi tiêu TB (1-100) | Số lượng | Tỷ lệ (%) | Chân dung Persona |\n")
        f.write("|---|---|---|---|---|---|\n")
        for _, row in df_profile.iterrows():
            f.write(f"| **{int(row['Cluster'])}** | {row['Annual Income (k$)']:.2f} | {row['Spending Score (1-100)']:.2f} | {int(row['Count'])} | {row['Percentage (%)']:.1f}% | {row['Persona']} |\n")
        f.write("\n")
    print(f"[+] Đã lưu báo cáo Markdown tại: {rep_dir / 'evaluation_results.md'}")

    print("\n" + "=" * 70)
    print("HOÀN TẤT HUẤN LUYỆN LAB 3 THÀNH CÔNG (< 200 DÒNG)!")
    print("=" * 70)


if __name__ == "__main__":
    main()

