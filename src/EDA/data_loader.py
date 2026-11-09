"""
Phase 1: Module nạp dữ liệu từ file CSV.
"""
from pathlib import Path
import pandas as pd


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Nạp tập dữ liệu Mall_Customers từ đường dẫn CSV."""
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {file_path}")
    df = pd.read_csv(file_path)
    return df

