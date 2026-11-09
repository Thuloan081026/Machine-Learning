## LAB 3: PHÂN KHÚC KHÁCH HÀNG & PHÂN ĐOẠN ẢNH VỚI K-MEANS CLUSTERING

---

## 1. WHAT (Dự án này là gì? Giải quyết vấn đề gì?)
- **Mục tiêu bài toán chính (Practice Exercise 3):** Xây dựng hệ thống học máy không giám sát (*Unsupervised Learning*) để phân nhóm 200 khách hàng trung tâm thương mại thành các phân khúc đặc trưng (**Customer Personas**) dựa trên 2 thuộc tính hành vi mua sắm cốt lõi: **Thu nhập hàng năm (`Annual Income`)** và **Điểm chi tiêu (`Spending Score`)**.
- **Mục tiêu bài toán mở rộng (Practice Exercise 3 - Additional):** Ứng dụng giải thuật phân cụm K-Means vào lĩnh vực thị giác máy tính để **Phân đoạn ảnh màu (Image Segmentation)**, tự động gom nhóm hàng vạn điểm ảnh RGB thành $K \in \{2, 3, 5, 8\}$ dải màu chủ đạo phục vụ nén dữ liệu và nhận diện đối tượng.
- **Yêu cầu kỹ thuật cốt lõi:**
  - Lập trình **100% From-Scratch (Zero Scikit-Learn)** bằng NumPy và Pandas: tự viết giải thuật phân cụm `KMeansScratch`, khởi tạo xác suất thông minh `kmeans_plus_plus_init`, chuẩn hóa `Scaler`, tính khoảng cách Euclid ma trận hóa bằng NumPy 3D Broadcasting, và các hàm đo lường `Inertia` (WCSS), `silhouette_score`.
  - Khắc phục nguy cơ cực trị địa phương (Local Minima) của K-Means truyền thống thông qua cơ chế chọn tâm K-Means++ và đa khởi tạo $n_{\text{init}} = 10$.
  - Đóng gói mô hình thành tệp JSON nhẹ ([kmeans_model.json](../models/kmeans_model.json)) và triển khai microservice **Web REST API (Flask)** tại cổng 5002.

---

## 2. WHERE (Các thành phần nằm ở đâu trong dự án?)
Dự án được phân chia module hóa rõ ràng theo chuẩn kiến trúc công nghiệp:

| Thành phần | File / Thư mục | Nhiệm vụ chính |
|---|---|---|
| **Dữ liệu thô** | [`data/raw/Mall_Customers.csv`](../data/raw/Mall_Customers.csv), [`data/raw/nature_landscape.jpg`](../data/raw/nature_landscape.jpg) | Chứa 200 bản ghi khách hàng gốc và bức ảnh phong cảnh tự nhiên dùng phân đoạn. |
| **Dữ liệu chuẩn hóa** | [`data/processed/X_scaled.csv`](../data/processed/X_scaled.csv), [`outputs/Mall_Customers_clustered.csv`](../outputs/Mall_Customers_clustered.csv) | Dữ liệu sau khi qua Scaler và bảng dữ liệu đã được gán nhãn phân cụm 0..4. |
| **Tiền xử lý & EDA** | [`src/EDA/`](../src/EDA/), [`src/Preprocessing/`](../src/Preprocessing/) | Kiểm tra chất lượng (0 missing, 0 duplicate), vẽ phân phối và chuẩn hóa Z-Score. |
| **Thuật toán From-Scratch** | [`src/training/`](../src/training/) | Chứa các module thuật toán tự lập trình: `kmeans.py`, `initialization.py` (K-Means++), `distance.py` (NumPy 3D Broadcasting). |
| **Đánh giá & Phân tích** | [`src/evaluate/`](../src/evaluate/) | Tính WCSS, Silhouette Score, quét tìm $K$ tối ưu và xây dựng hồ sơ chân dung khách hàng. |
| **File mô hình xuất xưởng** | [`models/kmeans_model.json`](../models/kmeans_model.json) | Lưu tọa độ 5 tâm cụm, bộ tham số Mean/Std của Scaler và tên nhãn Persona. |
| **Script chạy chính** | [`train.py`](../train.py) | Chạy toàn bộ pipeline từ nạp dữ liệu, quét $K=2..10$, train model $K=5$ và xuất báo cáo. |
| **Script phân đoạn ảnh** | [`scripts/image_segmentation.py`](../scripts/image_segmentation.py) | Chạy tác vụ nén ảnh K-Means RGB cho $K=2, 3, 5, 8$ và xuất biểu đồ Side-by-Side. |
| **Kiểm thử & Chạy thử** | [`scripts/eval.py`](../scripts/eval.py), [`scripts/demo.py`](../scripts/demo.py) | Đánh giá chỉ số phân cụm và chạy thử nghiệm 5 kịch bản phân khúc khách hàng mẫu. |
| **Sổ tay nghiên cứu** | [`notebooks/`](../notebooks/) | Gồm 2 Jupyter Notebook: phân khúc khách hàng (`01`) và phân đoạn ảnh (`02`). |
| **Bộ kiểm thử tự động** | [`tests/`](../tests/) | Bộ 15 bài kiểm thử unit tests (pytest) bảo đảm tính đúng đắn toán học. |
| **Ứng dụng API** | [`app.py`](../app.py), [`predict.py`](../predict.py) | Máy chủ Flask Web API và công cụ CLI để suy luận phân khúc người dùng mới. |
| **Báo cáo & Biểu đồ** | [`reports/BAO_CAO_LAB3.md`](../reports/BAO_CAO_LAB3.md), [`outputs/figures/`](../outputs/figures/) | Báo cáo đồ án chi tiết và các biểu đồ phân tích (Elbow, Silhouette, Clusters 2D, Image Side-by-Side). |

---

## 3. WHEN (Quy trình thực hiện theo thứ tự nào?)
Quy trình thực hiện dự án diễn ra tuần tự theo các bước chặt chẽ:

```text
[Bước 1: Nạp và kiểm tra dữ liệu khách hàng]
  └─ Đọc file Mall_Customers.csv (200 dòng), kiểm tra missing (0 NaN), loại bỏ CustomerID.
       │
[Bước 2: Phân tích khám phá EDA & Chọn đặc trưng]
  └─ Vẽ Histogram phân phối, Boxplot tìm ngoại lai và Scatter plot tương quan.
  └─ Chọn cặp biến cốt lõi: 'Annual Income (k$)' và 'Spending Score (1-100)'.
       │
[Bước 3: Chuẩn hóa Z-Score độc lập]
  └─ Dùng Scaler tự viết để đưa 2 trục về thang đo mean=0, std=1 (lưu tham số mean_, std_).
       │
[Bước 4: Quét tìm số cụm K tối ưu (K = 2 đến 10)]
  └─ Chạy K-Means quét qua K=2..10. Tính toán hàm mất mát WCSS (Inertia) và điểm Silhouette Score.
  └─ Đồ thị Elbow xuất hiện điểm gập và Silhouette đạt đỉnh cao nhất (0.5547) tại K = 5.
       │
[Bước 5: Huấn luyện mô hình Champion K = 5]
  └─ Khởi tạo thông minh K-Means++, chạy đa khởi tạo n_init=10 vòng lặp Lloyd để tìm nghiệm tối ưu toàn cục.
  └─ Xử lý bù cụm rỗng (Empty Cluster Re-seeding) nếu có cụm không nhận được mẫu nào.
       │
[Bước 6: Giải mã 5 Chân dung Khách hàng (Customer Personas)]
  └─ Giải chuẩn hóa tâm cụm về giá trị gốc: gán nhãn 5 nhóm (VIP, Tiết kiệm, Giới trẻ, Thận trọng, Trung lưu).
       │
[Bước 7: Thực nghiệm mở rộng Phân đoạn ảnh (Image Segmentation)]
  └─ Đọc ảnh nature_landscape.jpg, duỗi thành mảng pixel (15000, 3) kênh RGB chuẩn hóa [0, 1].
  └─ Chạy K-Means phân đoạn ảnh với K=2, 3, 5, 8 và xuất biểu đồ so sánh Side-by-Side.
       │
[Bước 8: Tuần tự hóa JSON & Triển khai Flask API]
  └─ Lưu file models/kmeans_model.json và khởi chạy app.py (port 5002) phục vụ suy luận.
```

---

## 4. WHY (Tại sao lại làm như vậy?)
Lý do đằng sau các quyết định kỹ thuật cốt lõi trong Lab 3:

1. **Tại sao chỉ chọn `Annual Income` và `Spending Score` mà không gom cả `Age` vào phân cụm 2D?**
   - Khi vẽ ma trận phân tán đa chiều, cặp `Age` vs `Spending Score` tạo thành một đám mây điểm dày đặc không có biên giới tự nhiên rõ ràng. Ngược lại, cặp `Annual Income` vs `Spending Score` hình thành **5 hòn đảo dữ liệu tách biệt cực kỳ sắc nét**, mang lại giá trị định vị thương mại và khả năng hành động tiếp thị cao nhất.
2. **Tại sao bắt buộc phải dùng thuật toán khởi tạo K-Means++ thay vì khởi tạo ngẫu nhiên?**
   - Khởi tạo ngẫu nhiên có xác suất rất cao chọn 2 tâm cụm nằm sát nhau trong cùng 1 cụm dày đặc, khiến thuật toán hội tụ vào nghiệm cực trị địa phương tồi tệ. K-Means++ chọn tâm tiếp theo với xác suất tỷ lệ thuận với $D(x)^2$ (bình phương khoảng cách tới tâm gần nhất), ép các tâm phải tản ra xa nhau nhất có thể, được chứng minh toán học giúp cải thiện tốc độ hội tụ và giảm sai số $O(\log K)$ lần.
3. **Tại sao lại dùng 3D NumPy Broadcasting để tính khoảng cách Euclid?**
   - Thay vì dùng 2 vòng lặp lồng nhau qua $N$ điểm và $K$ cụm (vốn cực chậm trên Python thuần khi xử lý hàng vạn pixel ảnh), công thức `diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]` tận dụng việc tính toán vector hóa ở tầng mã máy C/BLAS của NumPy, tăng tốc độ xử lý lên gấp $95\%$.
4. **Tại sao chọn $K = 5$ là số cụm tối ưu nhất?**
   - Có sự đồng thuận tuyệt đối giữa 2 phương pháp đánh giá khoa học:
     - **Phương pháp Elbow (WCSS):** Đường cong sai số nội cụm giảm dốc rất nhanh từ $K=2 \to 4$ và bắt đầu bão hòa rõ nét tại điểm gập $K = 5$.
     - **Hệ số Silhouette:** Đo lường độ gắn kết nội cụm so với độ tách biệt ngoại cụm đạt đỉnh cực đại toàn cục **$0.5547$ tại $K = 5$** (từ $K \ge 6$, điểm Silhouette sụt giảm nghiêm trọng do các cụm tự nhiên bị chia cắt phân mảnh khiên cưỡng).
5. **Tại sao bản chất toán học của Phân khúc khách hàng và Phân đoạn ảnh lại tương đồng?**
   - Cả hai đều là bài toán tối ưu hóa hình học nhằm thu nhỏ khoảng cách bình phương nội cụm: phân khúc khách hàng là gom cụm vector 2 chiều $[\text{Thu nhập}, \text{Chi tiêu}]$, còn phân đoạn ảnh là gom cụm vector 3 chiều $[\text{Red}, \text{Green}, \text{Blue}]$. Giải thuật K-Means không phân biệt đối tượng là người hay điểm ảnh, nó chỉ tìm ra các tâm biểu diễn tối ưu nhất cho không gian đa chiều.

---

## 5. HOW (Mô hình hoạt động và dự đoán như thế nào?)
Khi nhận được dữ liệu của một khách hàng mới gửi lên qua Web API hoặc CLI, quy trình suy luận diễn ra qua **4 bước**:

```text
[Dữ liệu khách hàng mới: Thu nhập = 90k$, Điểm chi tiêu = 85]
       │
       ▼
1. Đọc và Kiểm tra: Tiếp nhận input JSON, kiểm tra tính hợp lệ của hai trường số học.
       │
       ▼
2. Chuẩn hóa Z-Score: Sử dụng mean_ và std_ đã lưu trong kmeans_model.json để chuẩn hóa:
   z_income = (90 - 60.56) / 26.26 = +1.121
   z_spending = (85 - 50.20) / 25.82 = +1.348
       │
       ▼
3. Tính khoảng cách Euclid tới 5 tâm cụm chuẩn hóa:
   d(x, μ_0) = 0.1729  <-- Cực tiểu (Gần nhất!)
   d(x, μ_1) = 3.3421
   d(x, μ_2) = 2.4110
   d(x, μ_3) = 2.6582
   d(x, μ_4) = 1.6214
       │
       ▼
4. Gán nhãn & Trả kết quả:
   - Cụm gán: Cluster 0
   - Chân dung: "VIP / Big Spenders (Thu nhập cao - Chi tiêu mạnh)"
   - Khoảng cách tới tâm: 0.1729
```
