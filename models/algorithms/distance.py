"""
Module tính khoảng cách ma trận hóa (Broadcasting) bằng NumPy thuần.
100% From-Scratch (Zero Scikit-Learn).
"""
import numpy as np


def squared_euclidean_distance(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Tính ma trận bình phương khoảng cách Euclid giữa N điểm và K tâm cụm.
    Shape: (N, 1, D) - (1, K, D) -> (N, K)
    """
    diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    return np.sum(diff ** 2, axis=2)


def euclidean_distance(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Tính ma trận khoảng cách Euclid giữa N điểm và K tâm cụm.
    Shape: (N, 1, D) - (1, K, D) -> (N, K)
    """
    return np.sqrt(squared_euclidean_distance(X, centroids))


def manhattan_distance(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Tính ma trận khoảng cách Manhattan (L1 norm).
    Shape: (N, K)
    """
    diff = np.abs(X[:, np.newaxis, :] - centroids[np.newaxis, :, :])
    return np.sum(diff, axis=2)
