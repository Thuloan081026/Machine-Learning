# Data Pipeline Architecture — Customer Segmentation

```text
Raw Data (Mall_Customers.csv)
  │
  ├── 1. Ingestion: Đọc file CSV, trích xuất metadata và số dòng/cột.
  ├── 2. Validation: Kiểm tra 6 quy tắc toàn vẹn (schema, kiểu dữ liệu, miền giá trị).
  ├── 3. Cleaning: Xử lý missing values bằng Median/Mode, loại bỏ trùng lặp.
  ├── 4. Feature Selection: Trích xuất ma trận số học (Annual Income, Spending Score).
  ├── 5. Scaling: Chuẩn hóa Z-Score với lớp Scaler From-Scratch (lưu mean, std).
  └── 6. Model Input: Ma trận đặc trưng số học 200x2 chuyển vào KMeansScratch.
```

