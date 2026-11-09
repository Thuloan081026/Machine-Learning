"""
Unit tests cho Euclidean Distance.
"""

import numpy as np
from src.training.distance import euclidean_distance, squared_euclidean_distance


def test_euclidean_distance_known_value():
    """Kiểm tra distance([0,0], [3,4]) == 5.0 (Pythagorean 3-4-5)."""
    X = np.array([[3.0, 4.0]])
    centroids = np.array([[0.0, 0.0]])
    dist = euclidean_distance(X, centroids)
    assert dist.shape == (1, 1)
    np.testing.assert_allclose(dist[0, 0], 5.0, atol=1e-9)


def test_squared_euclidean_distance():
    """Kiểm tra squared distance([0,0], [3,4]) == 25.0."""
    X = np.array([[3.0, 4.0]])
    centroids = np.array([[0.0, 0.0]])
    sq_dist = squared_euclidean_distance(X, centroids)
    np.testing.assert_allclose(sq_dist[0, 0], 25.0, atol=1e-9)


def test_euclidean_distance_broadcasting_shapes():
    """Kiểm tra ma trận distance cho N samples và K centroids."""
    N, K, D = 10, 4, 3
    np.random.seed(42)
    X = np.random.randn(N, D)
    centroids = np.random.randn(K, D)

    dist = euclidean_distance(X, centroids)
    assert dist.shape == (N, K)

    # Kiểm tra tính toán thủ công từng cặp
    for i in range(N):
        for j in range(K):
            manual = np.linalg.norm(X[i] - centroids[j])
            np.testing.assert_allclose(dist[i, j], manual, atol=1e-9)
