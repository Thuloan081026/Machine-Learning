"""
Package models cho Lab 3: Customer Segmentation & K-Means Clustering.
Cấu trúc tương đồng Lab 2:
- models/algorithms/: Mã nguồn thuật toán From-Scratch
- models/save_model/: Mô hình và Scaler đã lưu (.pkl / .json)
"""
from models.algorithms.kmeans import KMeansScratch
from models.algorithms.preprocessing import Scaler

__all__ = ["KMeansScratch", "Scaler"]
