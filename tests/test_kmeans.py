"""
Unit tests cho KMeansScratch.
"""

import numpy as np
import pytest
from src.training.kmeans import KMeansScratch


def test_kmeans_toy_dataset():
    """Kiểm tra với tập nhỏ 4 điểm tách biệt rõ: [1,1], [1,2] vs [9,9], [10,9]."""
    X = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [9.0, 9.0],
        [10.0, 9.0],
    ])
    model = KMeansScratch(n_clusters=2, init="kmeans++", n_init=5, random_state=42)
    labels = model.fit_predict(X)

    # 2 điểm đầu cùng 1 cluster, 2 điểm sau cùng 1 cluster khác
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert model.converged_ is True


def test_kmeans_synthetic_gaussian_clusters():
    """Kiểm tra với 3 cụm Gaussian nhân tạo tách rời."""
    np.random.seed(42)
    c1 = np.random.randn(30, 2) + np.array([-10.0, -10.0])
    c2 = np.random.randn(30, 2) + np.array([0.0, 10.0])
    c3 = np.random.randn(30, 2) + np.array([10.0, -10.0])
    X = np.vstack([c1, c2, c3])

    model = KMeansScratch(n_clusters=3, init="kmeans++", n_init=10, random_state=42)
    labels = model.fit_predict(X)

    assert len(np.unique(labels)) == 3
    # Độ thuần khiết từng cụm
    for start, end in [(0, 30), (30, 60), (60, 90)]:
        cluster_labels = labels[start:end]
        most_common_count = np.max(np.bincount(cluster_labels))
        assert most_common_count >= 28  # độ chính xác >= 93%


def test_kmeans_predict_new_samples():
    """Kiểm tra tính năng predict cho sample mới."""
    X = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [9.0, 9.0],
        [10.0, 9.0],
    ])
    model = KMeansScratch(n_clusters=2, init="kmeans++", n_init=5, random_state=42)
    model.fit(X)

    X_new = np.array([[1.5, 1.5], [9.5, 9.5]])
    preds = model.predict(X_new)
    assert preds[0] == model.labels_[0]
    assert preds[1] == model.labels_[2]


def test_kmeans_k_greater_than_samples_raises():
    """K > n_samples phải raise ValueError."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    model = KMeansScratch(n_clusters=3)
    with pytest.raises(ValueError):
        model.fit(X)
