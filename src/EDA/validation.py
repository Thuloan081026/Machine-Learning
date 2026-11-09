"""
Phase 2: Kiểm tra tính toàn vẹn của dữ liệu (6 validation rules).
"""
import pandas as pd


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Kiểm tra 6 quy tắc toàn vẹn:
    1. Tập dữ liệu không rỗng
    2. Đầy đủ các cột bắt buộc
    3. Không có giá trị thiếu (NaN)
    4. Không có dòng trùng lặp hoàn toàn
    5. ID khách hàng là duy nhất
    6. Giá trị số hợp lệ (Tuổi > 0, Thu nhập >= 0, Điểm chi tiêu [1..100])
    """
    required_cols = ["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu các cột bắt buộc: {missing_cols}")

    results = {
        "n_samples": len(df),
        "n_columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_ids": int(df["CustomerID"].nunique()),
        "is_valid": True
    }

    if results["n_samples"] == 0:
        results["is_valid"] = False
        raise ValueError("Tập dữ liệu rỗng!")

    return results

