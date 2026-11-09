"""
Module đo lường chất lượng phân cụm: Inertia (WCSS) & Silhouette Score.
"""
import numpy as np


def compute_inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Tính WCSS (Within-Cluster Sum of Squares) / Inertia.
    Inertia = sum_{i=1}^N ||x_i - mu_{c_i}||^2
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    centroids = np.asarray(centroids, dtype=np.float64)
    return float(np.sum((X - centroids[labels]) ** 2))


def silhouette_samples(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Tính hệ số Silhouette cho từng mẫu i:
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    Nếu mẫu thuộc cụm singleton (1 phần tử) -> s(i) = 0.0
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    n_samples = len(X)
    unique_labels = np.unique(labels)

    if len(unique_labels) <= 1:
        return np.zeros(n_samples, dtype=np.float64)

    # Ma trận khoảng cách giữa mọi cặp điểm pairwise: (N, N)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    pairwise_dist = np.sqrt(np.sum(diff ** 2, axis=2))

    scores = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        own_label = labels[i]
        same_cluster_mask = (labels == own_label)
        n_same = np.sum(same_cluster_mask) - 1

        if n_same == 0:
            # Singleton cluster
            scores[i] = 0.0
            continue

        # a(i): Khoảng cách trung bình nội cụm
        a_i = np.sum(pairwise_dist[i, same_cluster_mask]) / n_same

        # b(i): Khoảng cách trung bình tới cụm gần nhất khác cụm
        b_i = np.inf
        for other_label in unique_labels:
            if other_label == own_label:
                continue
            other_cluster_mask = (labels == other_label)
            if np.sum(other_cluster_mask) > 0:
                avg_dist_other = np.mean(pairwise_dist[i, other_cluster_mask])
                if avg_dist_other < b_i:
                    b_i = avg_dist_other

        denom = max(a_i, b_i)
        scores[i] = (b_i - a_i) / denom if denom > 0 else 0.0

    return scores


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Tính Silhouette Score trung bình trên toàn bộ tập dữ liệu."""
    scores = silhouette_samples(X, labels)
    return float(np.mean(scores))

