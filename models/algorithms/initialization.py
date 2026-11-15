"""
Module khởi tạo tâm cụm ban đầu (Random & K-Means++).
100% From-Scratch (Zero Scikit-Learn).
"""
import numpy as np

try:
    from models.algorithms.distance import euclidean_distance
except ImportError:
    from .distance import euclidean_distance


def random_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Khởi tạo ngẫu nhiên K điểm dữ liệu làm K tâm cụm."""
    n_samples = X.shape[0]
    indices = rng.choice(n_samples, size=k, replace=False)
    return X[indices].copy()


def kmeans_plus_plus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Khởi tạo K tâm cụm theo K-Means++:
    Tâm tiếp theo được chọn theo phân phối xác suất tỉ lệ với D(x)^2.
    """
    n_samples, n_features = X.shape
    centroids = np.empty((k, n_features), dtype=np.float64)

    # Tâm thứ nhất chọn ngẫu nhiên
    first_idx = rng.integers(0, n_samples)
    centroids[0] = X[first_idx]

    # Các tâm còn lại chọn dựa trên xác suất D(x)^2
    for c_idx in range(1, k):
        dists = euclidean_distance(X, centroids[:c_idx])
        min_sq_dists = np.min(dists ** 2, axis=1)
        total_sq = np.sum(min_sq_dists)
        if total_sq == 0.0:
            prob = np.ones(n_samples) / n_samples
        else:
            prob = min_sq_dists / total_sq
        next_idx = rng.choice(n_samples, p=prob)
        centroids[c_idx] = X[next_idx]

    return centroids
