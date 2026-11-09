# Model Card — KMeansScratch

- **Tên mô hình:** KMeansScratch (Customer Segmentation Champion Model)
- **Phiên bản:** 1.0.0
- **Tác giả:** Triển khai From-Scratch thuần NumPy/Pandas
- **Loại mô hình:** Phân cụm không giám sát (Clustering)
- **Đặc trưng đầu vào:**
  1. `Annual Income (k$)` (Thu nhập năm)
  2. `Spending Score (1-100)` (Điểm chi tiêu)
- **Số lượng cụm (K):** 5 cụm
- **Chỉ số đánh giá:**
  - Inertia (WCSS): ~44.42 (trên không gian chuẩn hóa)
  - Silhouette Score: ~0.5547
- **Định dạng lưu trữ:** `models/kmeans_model.json`

