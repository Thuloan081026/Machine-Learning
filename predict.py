"""
Script Suy Luận Phân Khúc Khách Hàng Từ Mô Hình K-Means Đã Huấn Luyện.
"""
import argparse
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

import numpy as np

import config
from src.Preprocessing.scaler import Scaler
from src.training.distance import euclidean_distance


def load_model_artifact(model_path: Path) -> dict:
    """Nạp file JSON chứa tham số mô hình K-Means và Scaler."""
    if not model_path.exists():
        raise FileNotFoundError(f"Chưa tìm thấy mô hình tại {model_path}. Hãy chạy python train.py trước!")
    with open(model_path, "r", encoding="utf-8") as f:
        artifact = json.load(f)
    return artifact


def predict_customer_segment(income: float, spending: float, artifact: dict) -> dict:
    """Dự đoán cụm và hồ sơ khách hàng mới."""
    scaler = Scaler().set_params(artifact["scaler_params"])
    centroids_scaled = np.array(artifact["centroids_scaled"], dtype=np.float64)
    personas = artifact["cluster_personas"]

    # Chuẩn hóa mẫu mới theo đúng tham số scaler đã fit
    X_new = np.array([[income, spending]], dtype=np.float64)
    X_new_scaled = scaler.transform(X_new)

    # Tính khoảng cách tới các tâm cụm
    dists = euclidean_distance(X_new_scaled, centroids_scaled)[0]
    best_cluster = int(np.argmin(dists))
    persona = personas[best_cluster] if best_cluster < len(personas) else "Chưa xác định"

    return {
        "cluster": best_cluster,
        "persona": persona,
        "distance_to_centroid": float(dists[best_cluster])
    }


def run_demo(artifact: dict):
    """Chạy kịch bản kiểm thử mẫu trên 5 khách hàng tiêu biểu."""
    test_cases = [
        ("Khách hàng A (VIP)", 90.0, 85.0),
        ("Khách hàng B (Tiết kiệm)", 95.0, 15.0),
        ("Khách hàng C (Trendsetter)", 25.0, 80.0),
        ("Khách hàng D (Dè dặt)", 20.0, 20.0),
        ("Khách hàng E (Trung lưu)", 55.0, 50.0),
    ]
    print("\n" + "=" * 70)
    print("CHẠY THỬ NGHIỆM DỰ BÁO PHÂN KHÚC KHÁCH HÀNG (DEMO CLI)")
    print("=" * 70)
    for name, inc, sp in test_cases:
        res = predict_customer_segment(inc, sp, artifact)
        print(f"\n[{name}]")
        print(f"  Thu nhập: {inc} k$ | Điểm chi tiêu: {sp}/100")
        print(f"  -> Cụm gán: Cụm {res['cluster']}")
        print(f"  -> Chân dung Persona: {res['persona']}")
        print(f"  -> Khoảng cách tới tâm cụm: {res['distance_to_centroid']:.4f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Dự đoán phân khúc khách hàng K-Means Scratch")
    parser.add_argument("--income", type=float, help="Thu nhập hàng năm của khách (Annual Income k$)")
    parser.add_argument("--spending", type=float, help="Điểm chi tiêu từ 1 đến 100 (Spending Score)")
    parser.add_argument("--demo", action="store_true", help="Chạy thử nghiệm trên các khách hàng mẫu")
    args = parser.parse_args()

    artifact = load_model_artifact(config.MODEL_PATH)

    if args.demo or (args.income is None and args.spending is None):
        run_demo(artifact)
    elif args.income is not None and args.spending is not None:
        res = predict_customer_segment(args.income, args.spending, artifact)
        print(f"\n[+] Kết quả dự đoán cho khách hàng ({args.income}k$, chi tiêu {args.spending}):")
        print(f"    - Phân cụm: Cụm {res['cluster']}")
        print(f"    - Chân dung: {res['persona']}")
    else:
        print("[!] Vui lòng cung cấp cả hai tham số --income và --spending, hoặc dùng --demo!")


if __name__ == "__main__":
    main()

