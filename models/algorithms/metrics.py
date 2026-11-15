"""
Module đo lường chất lượng phân cụm From-Scratch:
- Inertia (WCSS)
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
Zero Scikit-Learn.
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

    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    pairwise_dist = np.sqrt(np.sum(diff ** 2, axis=2))

    scores = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        own_label = labels[i]
        same_cluster_mask = (labels == own_label)
        n_same = np.sum(same_cluster_mask) - 1

        if n_same == 0:
            scores[i] = 0.0
            continue

        a_i = np.sum(pairwise_dist[i, same_cluster_mask]) / n_same

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


def davies_bouldin_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Tính Davies-Bouldin Index From-Scratch:
    DB = (1 / k) * sum_{i} max_{j != i} ((s_i + s_j) / d(mu_i, mu_j))
    Giá trị càng nhỏ phân cụm càng tách biệt và gọn gàng.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    if k <= 1:
        return 0.0

    centroids = np.zeros((k, X.shape[1]))
    dispersions = np.zeros(k)

    for idx, c in enumerate(unique_labels):
        cluster_points = X[labels == c]
        centroids[idx] = np.mean(cluster_points, axis=0)
        dispersions[idx] = np.mean(np.linalg.norm(cluster_points - centroids[idx], axis=1))

    # Tính tỷ số tương đồng R_ij
    centroid_dists = np.linalg.norm(centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    np.fill_diagonal(centroid_dists, np.inf)

    db_terms = np.zeros(k)
    for i in range(k):
        r_ij = (dispersions[i] + dispersions) / centroid_dists[i]
        db_terms[i] = np.max(r_ij)

    return float(np.mean(db_terms))
