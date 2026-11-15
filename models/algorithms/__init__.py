"""
Package chứa toàn bộ các thuật toán phân cụm From-Scratch của Lab 3:
- KMeansScratch: Thuật toán K-Means thuần NumPy
- euclidean_distance, squared_euclidean_distance, manhattan_distance
- random_init, kmeans_plus_plus_init
- compute_inertia, silhouette_score, davies_bouldin_score
- Scaler: Bộ chuẩn hóa dữ liệu
"""
from models.algorithms.kmeans import KMeansScratch
from models.algorithms.distance import (
    euclidean_distance,
    squared_euclidean_distance,
    manhattan_distance,
)
from models.algorithms.initialization import random_init, kmeans_plus_plus_init
from models.algorithms.metrics import (
    compute_inertia,
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
)
from models.algorithms.preprocessing import Scaler

__all__ = [
    "KMeansScratch",
    "euclidean_distance",
    "squared_euclidean_distance",
    "manhattan_distance",
    "random_init",
    "kmeans_plus_plus_init",
    "compute_inertia",
    "silhouette_score",
    "silhouette_samples",
    "davies_bouldin_score",
    "Scaler",
]
