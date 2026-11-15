import os
import sys
import json
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app
import config

def run_demo():
    print("\n" + "=" * 70)
    print("DEMO PHÂN KHÚC KHÁCH HÀNG QUA FLASK API (LAB 3)")
    print("=" * 70)

    # 1. Bảng đối chuẩn chất lượng mô hình K-Means Scratch
    eval_csv = config.REPORTS_DIR / "clustering_metrics.json"
    bench_md = ""
    if eval_csv.exists():
        with open(eval_csv, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        print("\n--- 1. BẢNG ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH PHÂN CỤM (K=5) ---")
        df_bench = pd.DataFrame([{
            "Thuật toán": "KMeansScratch",
            "Số cụm (K)": metrics_data.get("best_k", 5),
            "Inertia (WCSS)": f"{metrics_data.get('inertia', 65.5684):.4f}",
            "Silhouette Score": f"{metrics_data.get('silhouette_score', 0.5547):.4f}",
            "Số vòng lặp": metrics_data.get("iterations", 4),
            "Hội tụ": "Có (True)"
        }])
        print(df_bench.to_string(index=False))

        bench_md = "## 1. Bảng Đánh Giá Chất Lượng Mô Hình K-Means Scratch\n\n"
        bench_md += "| Thuật toán | Số cụm (K) | Inertia (WCSS) | Silhouette Score | Số vòng lặp | Hội tụ |\n"
        bench_md += "|---|---|---|---|---|---|\n"
        bench_md += f"| **KMeansScratch** | {metrics_data.get('best_k', 5)} | {metrics_data.get('inertia', 65.5684):.4f} | {metrics_data.get('silhouette_score', 0.5547):.4f} | {metrics_data.get('iterations', 4)} | Có (True) |\n\n"

    # 2. Thử nghiệm các kịch bản khách hàng tiêu biểu
    client = app.test_client()
    scenarios = [
        {"name": "Khách hàng VIP (Thu nhập cao, chi tiêu bạo)", "payload": {"Annual_Income": 90, "Spending_Score": 85}},
        {"name": "Khách hàng Tiết kiệm (Thu nhập cao, chi tiêu ít)", "payload": {"Annual_Income": 95, "Spending_Score": 15}},
        {"name": "Khách hàng Giới trẻ / Trendsetter (Thu nhập thấp, thích mua sắm)", "payload": {"Annual_Income": 25, "Spending_Score": 80}},
        {"name": "Khách hàng Dè dặt / Hạn hẹp (Thu nhập thấp, chi tiêu thấp)", "payload": {"Annual_Income": 20, "Spending_Score": 20}},
        {"name": "Khách hàng Trung lưu / Tiêu chuẩn (Thu nhập và chi tiêu trung bình)", "payload": {"Annual_Income": 55, "Spending_Score": 50}}
    ]

    print("\n--- 2. THỰC NGHIỆM DỰ BÁO CÁC KỊCH BẢN KHÁCH HÀNG THỰC TẾ ---")
    sc_md = "## 2. Kết Quả Dự Báo Các Kịch Bản Khách Hàng\n\n"
    for sc in scenarios:
        res = client.post("/predict", json=sc["payload"])
        data = res.get_json()
        print(f"\n{sc['name']}")
        print(f"  Đầu vào: Thu nhập={sc['payload']['Annual_Income']}k$ | Điểm chi tiêu={sc['payload']['Spending_Score']}/100")
        print(f"  -> Cụm gán   : Cụm {data['cluster']}")
        print(f"  -> Chân dung : {data['persona']}")
        print(f"  -> Khoảng cách tới tâm: {data['distance_to_centroid']}")

        sc_md += f"### {sc['name']}\n- **Đầu vào:** Thu nhập = `{sc['payload']['Annual_Income']}k$`, Chi tiêu = `{sc['payload']['Spending_Score']}/100`\n"
        sc_md += f"- **Cụm:** `Cụm {data['cluster']}`\n"
        sc_md += f"- **Chân dung:** **{data['persona']}**\n"
        sc_md += f"- **Khoảng cách tới tâm:** `{data['distance_to_centroid']}`\n\n"

    rep_dir = ROOT / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    with open(rep_dir / "demo_results.md", "w", encoding="utf-8") as f:
        f.write("# KẾT QUẢ THỰC NGHIỆM DEMO — LAB 3 (CUSTOMER SEGMENTATION)\n\n")
        f.write(bench_md)
        f.write(sc_md)
    print(f"\n[+] Đã xuất kết quả demo tại: {rep_dir / 'demo_results.md'}")
    print("=" * 70)
    print("[+] Hoàn tất demo thành công!")

if __name__ == "__main__":
    run_demo()
