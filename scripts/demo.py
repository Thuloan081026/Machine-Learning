import os, sys, json

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import app

def run_demo():
    print("\n" + "=" * 65)
    print("CHẠY THỬ NGHIỆM KỊCH BẢN DỰ ĐOÁN PHÂN KHÚC KHÁCH HÀNG (DEMO)")
    print("=" * 65)

    client = app.test_client()
    scenarios = [
        {"name": "Khách hàng VIP (Thu nhập cao, chi tiêu bạo)", "payload": {"Annual_Income": 90, "Spending_Score": 85}},
        {"name": "Khách hàng Tiết kiệm (Thu nhập cao, chi tiêu ít)", "payload": {"Annual_Income": 95, "Spending_Score": 15}},
        {"name": "Khách hàng Giới trẻ / Trendsetter (Thu nhập thấp, thích mua sắm)", "payload": {"Annual_Income": 25, "Spending_Score": 80}},
        {"name": "Khách hàng Dè dặt / Hạn hẹp (Thu nhập thấp, chi tiêu thấp)", "payload": {"Annual_Income": 20, "Spending_Score": 20}},
        {"name": "Khách hàng Trung lưu / Tiêu chuẩn (Thu nhập và chi tiêu trung bình)", "payload": {"Annual_Income": 55, "Spending_Score": 50}}
    ]

    md_lines = ["# KẾT QUẢ THỰC NGHIỆM DEMO — LAB 3 (CUSTOMER SEGMENTATION)\n\n"]
    for sc in scenarios:
        res = client.post("/predict", json=sc["payload"])
        data = res.get_json()
        print(f"\n[Kịch bản]: {sc['name']}")
        print(f"  Đầu vào: Thu nhập={sc['payload']['Annual_Income']}k$ | Điểm chi tiêu={sc['payload']['Spending_Score']}")
        print(f"  -> Cụm gán   : Cụm {data['cluster']}")
        print(f"  -> Chân dung : {data['persona']}")
        print(f"  -> Khoảng cách tới tâm cụm: {data['distance_to_centroid']}")
        md_lines.append(f"### {sc['name']}\n- **Đầu vào:** Thu nhập = `{sc['payload']['Annual_Income']}k$`, Chi tiêu = `{sc['payload']['Spending_Score']}/100`\n- **Cụm:** `Cụm {data['cluster']}`\n- **Chân dung:** **{data['persona']}**\n- **Khoảng cách:** `{data['distance_to_centroid']}`\n\n")

    print("\n" + "=" * 65)
    rep_dir = os.path.join(ROOT, "reports")
    os.makedirs(rep_dir, exist_ok=True)
    with open(os.path.join(rep_dir, "demo_results.md"), "w", encoding="utf-8") as f:
        f.writelines(md_lines)
    print(f"[+] Đã xuất kết quả demo tại: reports/demo_results.md")

if __name__ == "__main__":
    run_demo()

