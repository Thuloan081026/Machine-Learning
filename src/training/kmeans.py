"""
Thuật toán K-Means Clustering tự cài đặt 100% From-Scratch bằng NumPy.
"""
import numpy as np
from src.training.distance import euclidean_distance
from src.training.initialization import kmeans_plus_plus_init, random_init


class KMeansScratch:
    """
    Thuật toán phân cụm K-Means Scratch:
    - Hỗ trợ khởi tạo: 'kmeans++' và 'random'
    - Multiple Restarts: n_init
    - Tối ưu hóa Broadcasting NumPy
    - Xử lý cụm rỗng (Empty Cluster Handling)
    """
    def __init__(
        self,
        n_clusters: int = 5,
        init: str = "kmeans++",
        n_init: int = 10,
        max_iter: int = 300,
        max_iters: int = None,
        tol: float = 1e-4,
        random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iters if max_iters is not None else max_iter
        self.max_iters = self.max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.converged_ = False

    @property
    def cluster_centers_(self):
        """Alias cho centroids_ để tương thích với scikit-learn API chuẩn."""
        return self.centroids_

    def fit(self, X: np.ndarray):
        """Huấn luyện mô hình K-Means trên ma trận X."""
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        if self.n_clusters > n_samples:
            raise ValueError(f"n_clusters={self.n_clusters} không thể lớn hơn số mẫu={n_samples}!")

        base_rng = np.random.default_rng(self.random_state)
        best_inertia = np.inf

        for _ in range(self.n_init):
            run_seed = int(base_rng.integers(0, 2**31))
            run_rng = np.random.default_rng(run_seed)

            # Khởi tạo tâm cụm
            if self.init == "kmeans++":
                centroids = kmeans_plus_plus_init(X, self.n_clusters, run_rng)
            else:
                centroids = random_init(X, self.n_clusters, run_rng)

            converged = False
            n_iter = 0

            for it in range(self.max_iter):
                n_iter = it + 1
                # Gán nhãn cho từng điểm vào tâm gần nhất
                dists = euclidean_distance(X, centroids)
                labels = np.argmin(dists, axis=1)

                # Cập nhật tọa độ tâm cụm
                new_centroids = np.empty_like(centroids)
                for k in range(self.n_clusters):
                    points_in_k = X[labels == k]
                    if len(points_in_k) > 0:
                        new_centroids[k] = points_in_k.mean(axis=0)
                    else:
                        # Cụm rỗng: Chọn điểm xa tâm hiện tại nhất làm tâm mới
                        all_dists = np.min(euclidean_distance(X, centroids), axis=1)
                        farthest_idx = np.argmax(all_dists)
                        new_centroids[k] = X[farthest_idx]

                shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))
                centroids = new_centroids
                if shift < self.tol:
                    converged = True
                    break

            # Tính Inertia (WCSS)
            dists = euclidean_distance(X, centroids)
            labels = np.argmin(dists, axis=1)
            inertia = float(np.sum((X - centroids[labels]) ** 2))

            if inertia < best_inertia:
                best_inertia = inertia
                self.centroids_ = centroids
                self.labels_ = labels
                self.inertia_ = inertia
                self.n_iter_ = n_iter
                self.converged_ = converged

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán cụm cho các mẫu mới."""
        if self.centroids_ is None:
            raise RuntimeError("Mô hình chưa được huấn luyện trước khi gọi predict!")
        X = np.asarray(X, dtype=np.float64)
        dists = euclidean_distance(X, self.centroids_)
        return np.argmin(dists, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Vừa fit vừa trả về nhãn."""
        self.fit(X)
        return self.labels_

