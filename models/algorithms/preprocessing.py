"""
Module chuẩn hóa dữ liệu Scaler From-Scratch (Standardization & Min-Max).
Hỗ trợ save / load qua pickle.
Zero Scikit-Learn.
"""
import pickle
import numpy as np


class Scaler:
    """
    Bộ chuẩn hóa dữ liệu tự cài đặt bằng NumPy thuần:
    - method="standardization": z = (x - mean) / std
    - method="minmax": x_norm = (x - min) / (max - min)
    """
    def __init__(self, method: str = "standardization"):
        if method not in ("standardization", "minmax"):
            raise ValueError(f"Phương thức không hợp lệ: {method}. Chọn 'standardization' hoặc 'minmax'.")
        self.method = method
        self.is_fitted = False
        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None

    def fit(self, X: np.ndarray):
        """Tính toán các tham số thống kê trên dữ liệu X."""
        X = np.asarray(X, dtype=np.float64)
        if self.method == "standardization":
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
            self.std_ = np.where(self.std_ == 0.0, 1.0, self.std_)
        elif self.method == "minmax":
            self.min_ = np.min(X, axis=0)
            self.max_ = np.max(X, axis=0)
            denom = self.max_ - self.min_
            denom = np.where(denom == 0.0, 1.0, denom)
            self._denom = denom
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Áp dụng phép chuẩn hóa lên dữ liệu X."""
        if not self.is_fitted:
            raise RuntimeError("Scaler chưa được fit trước khi gọi transform!")
        X = np.asarray(X, dtype=np.float64)
        if self.method == "standardization":
            return (X - self.mean_) / self.std_
        else:
            denom = getattr(self, "_denom", self.max_ - self.min_)
            denom = np.where(denom == 0.0, 1.0, denom)
            return (X - self.min_) / denom

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Vừa fit vừa transform trong 1 bước."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Khôi phục dữ liệu đã scale về thang đo gốc ban đầu."""
        if not self.is_fitted:
            raise RuntimeError("Scaler chưa được fit!")
        X_scaled = np.asarray(X_scaled, dtype=np.float64)
        if self.method == "standardization":
            return (X_scaled * self.std_) + self.mean_
        else:
            denom = getattr(self, "_denom", self.max_ - self.min_)
            return (X_scaled * denom) + self.min_

    def get_params(self) -> dict:
        """Lấy tham số để serialize sang JSON."""
        params = {"method": self.method}
        if self.method == "standardization":
            params["mean"] = self.mean_.tolist() if self.mean_ is not None else None
            params["std"] = self.std_.tolist() if self.std_ is not None else None
        else:
            params["min"] = self.min_.tolist() if self.min_ is not None else None
            params["max"] = self.max_.tolist() if self.max_ is not None else None
        return params

    def set_params(self, params: dict):
        """Khôi phục tham số từ dictionary JSON."""
        self.method = params.get("method", "standardization")
        if self.method == "standardization":
            self.mean_ = np.array(params["mean"], dtype=np.float64)
            self.std_ = np.array(params["std"], dtype=np.float64)
        else:
            self.min_ = np.array(params["min"], dtype=np.float64)
            self.max_ = np.array(params["max"], dtype=np.float64)
            denom = self.max_ - self.min_
            self._denom = np.where(denom == 0.0, 1.0, denom)
        self.is_fitted = True
        return self

    def save(self, path: str):
        """Lưu scaler ra file pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        """Nạp scaler từ file pickle."""
        with open(path, "rb") as f:
            return pickle.load(f)
