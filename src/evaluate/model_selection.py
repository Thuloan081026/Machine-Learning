"""
Phase 15, 18: Lựa chọn số cụm K tối ưu qua phương pháp Elbow và Silhouette Score.
"""
from typing import List, Dict
import pandas as pd
import numpy as np
from src.training.kmeans import KMeansScratch
from src.evaluate.metrics import silhouette_score


def evaluate_k_range(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
    n_init: int = 10
) -> pd.DataFrame:
    """
    Thực nghiệm quét K trong đoạn [k_min, k_max].
    Tính WCSS (Inertia) và Silhouette Score cho từng K.
    """
    results: List[Dict] = []

    for k in range(k_min, k_max + 1):
        model = KMeansScratch(
            n_clusters=k,
            init="kmeans++",
            n_init=n_init,
            random_state=random_state
        )
        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels)

        results.append({
            "K": k,
            "WCSS": round(model.inertia_, 4),
            "Silhouette_Score": round(sil, 4),
            "Iterations": model.n_iter_,
            "Converged": model.converged_
        })

    df_results = pd.DataFrame(results)
    return df_results

