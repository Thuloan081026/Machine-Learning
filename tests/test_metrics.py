"""
Unit tests cho Inertia và Silhouette Score.
"""

import numpy as np
from src.evaluate.metrics import compute_inertia, silhouette_score, silhouette_samples


def test_inertia_calculation():
    """Kiểm tra tính toán inertia."""
    X = np.array([
        [1.0, 1.0],
        [1.0, 3.0],
        [9.0, 9.0],
        [9.0, 11.0],
    ])
    labels = np.array([0, 0, 1, 1])
    centroids = np.array([
        [1.0, 2.0],
        [9.0, 10.0],
    ])
    # Cluster 0: dist_sq = (0^2 + (-1)^2) + (0^2 + 1^2) = 1 + 1 = 2
    # Cluster 1: dist_sq = (0^2 + (-1)^2) + (0^2 + 1^2) = 1 + 1 = 2
    # Total = 4.0
    inertia = compute_inertia(X, labels, centroids)
    np.testing.assert_allclose(inertia, 4.0, atol=1e-9)


def test_silhouette_score_well_separated():
    """Tập dữ liệu phân tách cực tốt phải có silhouette > 0.8."""
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [100.0, 100.0],
        [100.1, 100.0],
    ])
    labels = np.array([0, 0, 1, 1])
    score = silhouette_score(X, labels)
    assert score > 0.8


def test_silhouette_singleton_cluster():
    """Cluster có 1 điểm duy nhất (singleton) phải được gán s(i) = 0 theo quy định."""
    X = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [50.0, 50.0],  # singleton
    ])
    labels = np.array([0, 0, 1])
    sample_scores = silhouette_samples(X, labels)

    # Điểm thứ 3 là singleton trong cluster 1 -> s = 0.0
    np.testing.assert_allclose(sample_scores[2], 0.0, atol=1e-9)
