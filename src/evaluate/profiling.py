"""
Phase 21: Phân tích hồ sơ khách hàng (Customer Persona) theo từng cụm.
"""
from typing import Dict
import pandas as pd
import numpy as np


def get_persona_name(avg_income: float, avg_spending: float) -> str:
    """Gán tên Persona định tính dựa trên thu nhập và điểm chi tiêu trung bình."""
    if avg_income > 70 and avg_spending > 70:
        return "VIP / Big Spenders (Thu nhập cao - Chi tiêu mạnh)"
    elif avg_income > 70 and avg_spending < 40:
        return "Careful / Savers (Thu nhập cao - Tiết kiệm/Thận trọng)"
    elif avg_income < 40 and avg_spending > 60:
        return "Target / Trendsetters (Thu nhập thấp - Chi tiêu phóng khoáng)"
    elif avg_income < 40 and avg_spending < 40:
        return "Sensible / Budget (Thu nhập thấp - Chi tiêu dè dặt)"
    else:
        return "Standard / Moderate (Thu nhập & Chi tiêu trung bình)"


def build_cluster_profiles(
    df_raw: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list
) -> pd.DataFrame:
    """Tạo bảng thống kê trung bình theo từng cụm và gán nhãn Persona."""
    df_with_cluster = df_raw.copy()
    df_with_cluster["Cluster"] = labels

    profile = df_with_cluster.groupby("Cluster")[feature_cols].mean().reset_index()
    counts = df_with_cluster["Cluster"].value_counts().reset_index()
    counts.columns = ["Cluster", "Count"]

    profile = pd.merge(profile, counts, on="Cluster")
    profile["Percentage (%)"] = (profile["Count"] / len(df_raw) * 100).round(1)

    personas = []
    for _, row in profile.iterrows():
        name = get_persona_name(row[feature_cols[0]], row[feature_cols[1]])
        personas.append(name)
    profile["Persona"] = personas

    return profile.sort_values(by="Cluster").reset_index(drop=True)

