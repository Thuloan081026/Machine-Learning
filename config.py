"""
Cấu hình tập trung cho Lab 3: K-Means Clustering Customer Segmentation.
"""
from pathlib import Path

# Thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent

# Đường dẫn dữ liệu
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Mall_Customers.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "X_scaled.csv"
EDA_PLOTS_DIR = DATA_DIR / "EDA" / "eda_plots"

# Đường dẫn báo cáo, biểu đồ và mô hình
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CLUSTERED_DATA_PATH = DATA_DIR / "processed" / "Mall_Customers_clustered.csv"
MODEL_DIR = BASE_DIR / "models"
SAVE_MODEL_DIR = MODEL_DIR / "save_model"
MODEL_PATH = SAVE_MODEL_DIR / "kmeans_model.json"
MODEL_PKL_PATH = SAVE_MODEL_DIR / "kmeans_model.pkl"
SCALER_PKL_PATH = SAVE_MODEL_DIR / "scaler.pkl"

# Cấu hình đặc trưng
FEATURE_COLS = ["Annual Income (k$)", "Spending Score (1-100)"]
ID_COL = "CustomerID"

# Cấu hình thuật toán K-Means
OPTIMAL_K = 5
K_MIN = 2
K_MAX = 10
RANDOM_STATE = 42
N_INIT = 10
MAX_ITERS = 300
TOL = 1e-4
SCALING_METHOD = "standardization"

