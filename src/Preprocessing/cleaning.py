"""
Phase 4: Làm sạch dữ liệu và xử lý ngoại lai/trùng lặp nếu có.
"""
import pandas as pd


def clean_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    """Loại bỏ trùng lặp và làm sạch định dạng dữ liệu."""
    df_clean = df.drop_duplicates().copy()
    return df_clean

