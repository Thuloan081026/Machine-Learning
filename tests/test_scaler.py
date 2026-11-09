"""
Unit tests cho Scaler (Standardization & Min-Max).
"""

import numpy as np
import pytest
from src.Preprocessing.scaler import Scaler


def test_standardization_properties():
    """Kiểm tra mean ≈ 0 và std ≈ 1 sau standardization."""
    np.random.seed(42)
    X = np.random.normal(loc=50.0, scale=15.0, size=(100, 3))
    scaler = Scaler(method="standardization")
    X_scaled = scaler.fit_transform(X)

    np.testing.assert_allclose(np.mean(X_scaled, axis=0), np.zeros(3), atol=1e-7)
    np.testing.assert_allclose(np.std(X_scaled, axis=0), np.ones(3), atol=1e-7)


def test_minmax_properties():
    """Kiểm tra min = 0 và max = 1 sau Min-Max scaling."""
    X = np.array([[10.0, 200.0], [20.0, 100.0], [30.0, 300.0]])
    scaler = Scaler(method="minmax")
    X_scaled = scaler.fit_transform(X)

    np.testing.assert_allclose(np.min(X_scaled, axis=0), [0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(np.max(X_scaled, axis=0), [1.0, 1.0], atol=1e-7)


def test_inverse_transform():
    """Kiểm tra tính bảo toàn khi inverse transform."""
    X = np.array([[15.0, 39.0], [35.0, 81.0], [80.0, 15.0]])
    for method in ["standardization", "minmax"]:
        scaler = Scaler(method=method)
        X_scaled = scaler.fit_transform(X)
        X_recovered = scaler.inverse_transform(X_scaled)
        np.testing.assert_allclose(X, X_recovered, atol=1e-7)


def test_scaler_params_serialization():
    """Kiểm tra get_params và set_params để lưu/load model."""
    X = np.array([[10.0, 20.0], [30.0, 40.0]])
    scaler1 = Scaler(method="standardization").fit(X)
    params = scaler1.get_params()

    scaler2 = Scaler().set_params(params)
    assert scaler2.is_fitted
    assert scaler2.method == "standardization"
    np.testing.assert_allclose(scaler1.mean_, scaler2.mean_)
    np.testing.assert_allclose(scaler1.std_, scaler2.std_)


def test_unfitted_scaler_raises():
    """Gọi transform trước fit phải raise RuntimeError."""
    scaler = Scaler(method="standardization")
    with pytest.raises(RuntimeError):
        scaler.transform(np.array([[1.0, 2.0]]))
