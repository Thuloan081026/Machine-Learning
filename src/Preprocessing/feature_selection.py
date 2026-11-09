"""
Phase 5: Trích xuất ma trận đặc trưng số học cho K-Means.
"""
from typing import List, Tuple
import numpy as np
import pandas as pd


def extract_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Trích xuất ma trận đặc trưng 2 chiều (Annual Income, Spending Score)."""
    X = df[feature_cols].values.astype(np.float64)
    return X, feature_cols

