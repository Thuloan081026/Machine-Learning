<div align="center">

# Customer Segmentation with K-Means Clustering

### From-scratch Unsupervised Learning Pipeline with Vectorized Lloyd's Algorithm, K-Means++, and Customer Persona Profiling

<table>
<tr>
<td><b>Notebook chính</b></td>
<td><code>notebooks/01_customer_segmentation_kmeans.ipynb</code></td>
</tr>
<tr>
<td><b>Dữ liệu</b></td>
<td><code>data/raw/Mall_Customers.csv</code></td>
</tr>
<tr>
<td><b>Bài toán</b></td>
<td>Phân cụm dữ liệu Khách hàng không giám sát (Customer Segmentation)</td>
</tr>
</table>
<p>
Project này xây dựng toàn bộ quy trình Phân khúc khách hàng, đi từ dữ liệu thô,
kiểm tra tính toàn vẹn, tiền xử lý, chuẩn hóa dữ liệu, quét K tối ưu (Elbow & Silhouette),
huấn luyện thuật toán K-Means From-Scratch, xây dựng hồ sơ Persona, đến triển khai suy luận qua Web API Flask.
</p>
</div>

## 1. Mục tiêu dự án

Trong kinh doanh bán lẻ, việc tiếp cận tất cả khách hàng bằng cùng một chiến lược tiếp thị là kém hiệu quả và gây lãng phí ngân sách. Mỗi nhóm khách hàng có mức thu nhập và thói quen mua sắm hoàn toàn khác nhau.

Bài toán đặt ra: **Phân nhóm tự động các khách hàng của trung tâm thương mại thành các phân khúc đặc trưng (Customer Personas) dựa trên Thu nhập hàng năm (Annual Income) và Điểm số chi tiêu (Spending Score).**

Mục tiêu chính:

- Triển khai **100% From-Scratch (Zero Scikit-Learn)** bằng NumPy và Pandas: tự viết khoảng cách Euclid (Broadcasting), lớp `Scaler` (Z-Score & Min-Max), giải thuật `KMeansScratch`, và công thức `Silhouette Score`.

- Cài đặt cơ chế khởi tạo thông minh **K-Means++** và đa khởi tạo $n_{\text{init}} = 10$ để khắc phục triệt để bẫy cực trị địa phương (Local Optima).

- Lựa chọn số cụm $K$ tối ưu dựa trên sự đồng thuận của hai phương pháp: **Khuỷu tay (Elbow/WCSS)** và **Hệ số Silhouette**.

- Phân tích và xây dựng hồ sơ chân dung khách hàng (**Customer Personas**) phục vụ các chiến dịch tiếp thị thực tế.

- Đóng gói toàn diện thành dịch vụ **Web REST API (Flask)** và các công cụ dòng lệnh CLI phục vụ dự đoán thời gian thực.

- Đảm bảo chất lượng mã nguồn bằng bộ kiểm thử tự động **pytest (15/15 tests passed)**.

---

## 2. Dữ liệu sử dụng & Xử lý (Data Overview)

Nguồn dữ liệu gốc: `data/raw/Mall_Customers.csv` gồm 200 dòng đại diện cho 200 khách hàng.

### 2.1. Dữ liệu gốc ban đầu

| CustomerID | Gender | Age | Annual Income (k$) | Spending Score (1-100) |
|---|---|---|---|---|
| 1 | Male | 19 | 15 | 39 |
| 2 | Male | 21 | 15 | 81 |
| 3 | Female | 20 | 16 | 6 |
| 4 | Female | 23 | 16 | 77 |

### 2.2. Tiền xử lý & Chuẩn hóa dữ liệu (Data Preprocessing & Scaling)

1. **Kiểm tra tính toàn vẹn (Validation):** Thực hiện 6 phép kiểm tra toàn vẹn (Schema, Missing Values, Duplicates, Boundary Range, Datatypes).
2. **Làm sạch dữ liệu (Cleaning):** Xử lý giá trị thiếu bằng Median/Mode, loại bỏ bản ghi trùng lặp.
3. **Trích xuất đặc trưng (Feature Selection):** Lựa chọn 2 đặc trưng quan trọng nhất để phân khúc là `Annual Income (k$)` và `Spending Score (1-100)`.
4. **Chuẩn hóa Z-Score (Scaling):** Sử dụng lớp `Scaler` tự lập trình để đưa hai đặc trưng về cùng thang đo (Mean = 0, Std = 1), lưu trữ tham số vào JSON để suy luận mà không gây rò rỉ dữ liệu (Data Leakage).

### 2.3. Lựa chọn số cụm tối ưu K (Elbow & Silhouette Analysis)

Pipeline thực hiện quét $K$ từ 2 đến 10 và đo lường sự thay đổi của Inertia (WCSS) cùng Silhouette Score:
- **Phương pháp Elbow:** Đồ thị WCSS giảm mạnh và tạo điểm gập (khuỷu tay) rõ nét tại $K = 5$.
- **Hệ số Silhouette:** Đạt đỉnh cực đại tại $K = 5$ (Silhouette Score $\approx 0.5547$).
- **5 Nhóm Persona định hình:**
  1. **Cụm 0 — VIP / Big Spenders:** Thu nhập cao - Chi tiêu mạnh.
  2. **Cụm 1 — Sensible / Budget:** Thu nhập thấp - Chi tiêu dè dặt.
  3. **Cụm 2 — Target / Trendsetters:** Thu nhập thấp - Chi tiêu phóng khoáng.
  4. **Cụm 3 — Careful / Savers:** Thu nhập cao - Tiết kiệm / Thận trọng.
  5. **Cụm 4 — Standard / Moderate:** Thu nhập & Chi tiêu trung bình.

---

## 3. Cấu trúc thư mục hiện tại

```text
Lab3_KMeans_Clustering/
├── README.md                             # Hướng dẫn chi tiết dự án phân cụm khách hàng Lab 3
├── train.py                              # Pipeline huấn luyện chính toàn bộ mô hình (Phases 1-23)
├── predict.py                            # Script CLI suy luận dự đoán phân khúc cho khách hàng mới
├── app.py                                # Web API Flask phục vụ phân loại khách hàng thời gian thực
├── config.py                             # Cấu hình tập trung tham số K-Means, đường dẫn, features
├── requirements.txt                      # Danh sách thư viện phụ thuộc (NumPy, Pandas, Flask...)
├── data/                                 # Quản lý dữ liệu thô và dữ liệu sau xử lý
│   ├── raw/                              # Dữ liệu khách hàng gốc
│   │   └── Mall_Customers.csv            # 200 bản ghi khách hàng (Tuổi, Thu nhập, Chi tiêu)
│   ├── processed/                        # Dữ liệu sau tiền xử lý và chuẩn hóa
│   │   └── X_scaled.csv                  # Ma trận đặc trưng số học đã chuẩn hóa Z-Score
│   └── EDA/                              # Thư mục chứa biểu đồ phân tích khám phá dữ liệu
│       └── eda_plots/                    # Biểu đồ phân phối histogram và phân tán ban đầu
├── models/                               # Lưu trữ mã nguồn thuật toán và mô hình đã lưu
│   ├── algorithms/                       # Mã nguồn thuật toán From-Scratch thuần NumPy
│   │   ├── kmeans.py                     # Thuật toán KMeansScratch (Lloyd + K-Means++)
│   │   ├── distance.py                   # Ma trận khoảng cách Euclid (Broadcasting)
│   │   ├── initialization.py             # Khởi tạo tâm K-Means++ và Random
│   │   ├── metrics.py                    # Đo lường WCSS, Silhouette, Davies-Bouldin
│   │   └── preprocessing.py              # Bộ chuẩn hóa dữ liệu Scaler
│   └── save_model/                       # Artifacts xuất xưởng đã lưu (.pkl & .json)
│       ├── kmeans_model.json             # Model artifact JSON đầy đủ metadata
│       ├── kmeans_model.pkl              # Mô hình K-Means pickle
│       └── scaler.pkl                    # Scaler artifact pickle
├── templates/                            # Giao diện Web tương tác trực quan
│   └── index.html                        # Giao diện Web Demo Bootstrap / FontAwesome
├── notebooks/                            # Sổ tay nghiên cứu trực quan
│   └── 01_customer_segmentation_kmeans.ipynb # Toàn bộ quy trình EDA, Elbow, Silhouette, Profiling
├── outputs/                              # Kết quả đầu ra của pipeline huấn luyện
│   ├── Mall_Customers_clustered.csv      # Dữ liệu khách hàng kèm cột nhãn 'Cluster'
│   ├── figures/                          # Bộ biểu đồ trực quan hóa cao cấp (Elbow, Silhouette, 2D)
│   └── reports/                          # Báo cáo số liệu chi tiết (Inertia, Silhouette, Persona)
├── scripts/                              # Kịch bản thực thi chức năng độc lập
│   ├── data_preparation.py              # Đọc dữ liệu, kiểm tra toàn vẹn, tiền xử lý và lưu X_scaled
│   ├── eval.py                           # Đánh giá độc lập mô hình K-Means trên tập dữ liệu
│   └── demo.py                           # Chạy thử nghiệm phân loại 5 kịch bản khách hàng tiêu biểu
├── reports/                              # Báo cáo kết quả và tài liệu trực quan
│   ├── BAO_CAO_LAB3.md                   # Báo cáo chi tiết toàn bộ kết quả phân cụm nộp môn học
│   └── figures/                          # Biểu đồ kết quả phân cụm và hồ sơ khách hàng
├── tests/                                # Bộ kiểm thử tự động toàn diện (15 unit tests)
│   ├── conftest.py                       # Cấu hình đường dẫn cho pytest
│   ├── test_distance.py                  # Kiểm thử ma trận khoảng cách Euclid & broadcasting
│   ├── test_kmeans.py                    # Kiểm thử thuật toán KMeansScratch trên synthetic data
│   ├── test_metrics.py                   # Kiểm thử Inertia, Silhouette Score & Singleton handling
│   └── test_scaler.py                    # Kiểm thử chuẩn hóa Z-score, MinMax & Inverse Transform
└── docs/                                 # Tài liệu kiến trúc và thiết kế
    ├── DESIGN.md                         # Thiết kế dự án theo cấu trúc 4W1H
    ├── PLAN.md                           # Kế hoạch phát triển chi tiết qua 25 phases
    ├── DATA_PIPELINE.md                  # Kiến trúc Pipeline xử lý và chuẩn hóa dữ liệu
    ├── ALGORITHMS.md                     # Chi tiết công thức toán học các thuật toán From-Scratch
    └── MODEL_CARD.md                     # Thẻ thông số kỹ thuật mô hình phân cụm
```

| Thành phần | Vai trò |
|---|---|
| `data/raw/Mall_Customers.csv` | File dữ liệu khách hàng gốc. |
| `notebooks/` | Chứa sổ tay thực nghiệm trực quan hóa EDA và phân cụm. |
| `src/` & `train.py` | Mã nguồn giải thuật From-Scratch và pipeline huấn luyện tự động. |
| `models/` | Mô hình K-Means đã huấn luyện lưu dưới định dạng JSON. |
| `app.py` | Web API Deployment bằng Flask để phục vụ phân loại thời gian thực. |
| `scripts/` | Kiểm tra dữ liệu, đánh giá độc lập và chạy demo 5 kịch bản. |
| `tests/` | Bộ 15 unit tests kiểm tra tính đúng đắn toán học (100% Pass). |
| `reports/` | Báo cáo, biểu đồ phân tích và kết quả phân cụm. |
| `docs/` | Thiết kế 4W1H, kế hoạch, pipeline, thuật toán và model card. |

---

## 4. Điểm nhấn Kỹ thuật (From-scratch compliance)

Các thành phần cốt lõi được triển khai 100% bằng NumPy và Pandas (tuyệt đối không dùng Scikit-Learn):

| Thành phần | Class / logic chính | Ghi chú |
|---|---|---|
| Chuẩn hóa Dữ liệu | `Scaler` | Class chuẩn hóa (Standardization & MinMax), hỗ trợ `fit`, `transform`, `inverse_transform`. |
| Khoảng cách Euclid | `euclidean_distance` | Tối ưu hóa bằng NumPy Broadcasting, không dùng vòng lặp `for`. |
| Khởi tạo Tâm thông minh | `kmeans_plus_plus_init` | Phân phối xác suất tỉ lệ với bình phương khoảng cách $D(x)^2$. |
| Đa Khởi tạo | `random_init` | Khởi tạo ngẫu nhiên kết hợp $n_{\text{init}} = 10$ chống cực trị địa phương. |
| Thuật toán Phân cụm | `KMeansScratch` | Triển khai vòng lặp Lloyd, xử lý cụm rỗng (Empty Cluster Re-seeding). |
| Đánh giá Cụm | `compute_inertia` | Tính tổng bình phương khoảng cách nội cụm (WCSS). |
| Hệ số Tách biệt | `silhouette_score` | Đánh giá độ gắn kết và phân tách của các cụm (xử lý singleton score = 0.0). |

---

## 5. Hướng dẫn Chạy Dự án (How to Run)

### A. Huấn luyện và Đánh giá tự động qua CLI

1. Cài đặt các thư viện cơ bản:
```bash
pip install -r requirements.txt
```

2. Chạy toàn bộ pipeline huấn luyện (Data Prep -> Validation -> K Selection -> Train -> Persona -> Save Model):
```bash
python train.py
```

3. Đánh giá mô hình đã lưu trên tập dữ liệu:
```bash
python scripts/eval.py
```

4. Chạy kịch bản demo phân loại khách hàng:
```bash
python scripts/demo.py
```

5. Chạy bộ kiểm thử tự động (Unit Tests):
```bash
pytest tests/
```

### B. Chạy Khối lượng Nghiên cứu (Notebooks)

1. Cài đặt thư viện:
```bash
pip install -r requirements.txt
```

2. Khởi chạy Jupyter Notebook hoặc mở trên VS Code:
```bash
jupyter notebook notebooks/01_customer_segmentation_kmeans.ipynb
```

### C. Chạy Triển khai API (Model Deployment)

1. Tại thư mục gốc của dự án, mở Terminal và gõ:
```bash
python app.py
```

2. Máy chủ Flask sẽ khởi chạy tại `http://127.0.0.1:5000/`. Các Endpoints hỗ trợ bao gồm:
- **`GET /`**: Trang chủ với giao diện Web Demo đồ họa trực quan (hỗ trợ nhập thu nhập/chi tiêu, thử nhanh kịch bản và xem bảng đối chuẩn phân cụm trực tiếp trên trình duyệt).
- **`GET /health`**: Kiểm tra trạng thái hoạt động của mô hình (Health Check).
- **`POST /predict`**: Nhận dữ liệu JSON (Annual_Income, Spending_Score) và trả về Cụm cùng Chân dung Persona.

### D. Kết quả Đánh giá Benchmark Mô hình Phân Cụm:

| Thuật toán | Số cụm (K) | Inertia (WCSS) | Silhouette Score | Davies-Bouldin | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **KMeansScratch 🏆** | **5** | **65.5684** | **0.5547** | **0.5722** | **Hội tụ (4 vòng lặp)** |

3. Mở một Terminal khác (hoặc dùng Postman), gửi dữ liệu khách hàng vào API để lấy kết quả phân khúc:

*(Ví dụ dùng PowerShell trên Windows)*
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -ContentType "application/json" -Body '{"Annual_Income": 90, "Spending_Score": 85}'
```

4. Kết quả phản hồi từ API:
```json
{
  "status": "success",
  "input": {"Annual_Income": 90.0, "Spending_Score": 85.0},
  "cluster": 0,
  "persona": "VIP / Big Spenders (Thu nhập cao - Chi tiêu mạnh)",
  "distance_to_centroid": 0.1729
}
```
