# BÁO CÁO DỰ ÁN MÔN HỌC MÁY HỌC (MACHINE LEARNING) — LAB 3

## PHÂN KHÚC KHÁCH HÀNG & PHÂN ĐOẠN ẢNH VỚI K-MEANS CLUSTERING (100% FROM SCRATCH)
**Tập dữ liệu chính:** `data/raw/Mall_Customers.csv` (Kaggle Dataset)  
**Tập dữ liệu mở rộng:** `data/raw/nature_landscape.jpg` (K-Means Image Segmentation)  
**Mã nguồn triển khai:** 100% Thuần NumPy / Pandas (Tuyệt đối không dùng Scikit-Learn)

---

## 1. GIỚI THIỆU (INTRODUCTION)

### 1.1. Bối cảnh và Định nghĩa bài toán
Trong thời đại kinh tế số và bán lẻ hiện đại, sự cạnh tranh giữa các doanh nghiệp không chỉ nằm ở chất lượng sản phẩm mà phụ thuộc sống còn vào việc thấu hiểu khách hàng. Nếu tiếp cận hàng triệu người tiêu dùng bằng một chiến dịch tiếp thị đồng nhất, doanh nghiệp sẽ lãng phí nguồn lực khổng lồ cho những đối tượng không có nhu cầu hoặc không đủ khả năng tài chính.

**Bài toán đặt ra:** Xây dựng hệ thống học máy tự động phân nhóm tập dữ liệu khách hàng thành các phân khúc đặc trưng (**Customer Personas**) dựa trên các chỉ số hành vi: **Thu nhập hàng năm (`Annual Income`)** và **Điểm số chi tiêu (`Spending Score`)**.

Bên cạnh bài toán bảng, dự án mở rộng sang bài toán thị giác máy tính: **Phân đoạn ảnh màu (Image Segmentation)**, tự động gom nhóm hàng chục nghìn pixel màu thành $K$ cụm màu chủ đạo để nén ảnh và tách biệt vật thể.

### 1.2. Mục tiêu của bài thực hành
1. Xây dựng quy trình học máy phân cụm hoàn chỉnh gồm 25 phases từ dữ liệu thô đến triển khai.
2. Triển khai **100% From-Scratch (Zero Scikit-Learn)** bằng NumPy:
   - Khoảng cách Euclid ma trận hóa với NumPy Broadcasting.
   - Lớp chuẩn hóa dữ liệu `Scaler` (hỗ trợ Standardization & Min-Max).
   - Giải thuật phân cụm `KMeansScratch` với khởi tạo thông minh K-Means++ và đa khởi tạo $n_{\text{init}} = 10$.
   - Các hàm đo lường chất lượng cụm: `compute_inertia` (WCSS) và `silhouette_score`.
3. Khảo sát và lựa chọn số cụm tối ưu $K$ dựa trên sự đồng thuận của phương pháp Khuỷu tay (Elbow) và hệ số Silhouette.
4. Xây dựng hồ sơ chân dung khách hàng (**Customer Personas**) chi tiết, đưa ra khuyến nghị kinh doanh thực tế.
5. Ứng dụng giải thuật vào bài toán **Phân đoạn ảnh (Image Segmentation)** theo yêu cầu mở rộng, hiển thị so sánh side-by-side.
6. Đóng gói mô hình thành dịch vụ **Web REST API (Flask)** và script CLI sẵn sàng cho Production.
7. Đảm bảo chất lượng bằng bộ 15 kiểm thử tự động **pytest (15/15 tests passed)**.

### 1.3. Loại bài toán học máy
Bài toán thuộc lớp **Học không giám sát (Unsupervised Learning)**, cụ thể là **Phân cụm dữ liệu (Data Clustering)**. Dữ liệu đầu vào hoàn toàn không có nhãn giám sát ($y$). Mô hình phải tự khám phá cấu trúc hình học ẩn sâu và ranh giới tự nhiên giữa các cụm điểm dữ liệu trong không gian đặc trưng.

---

## 2. MÔ TẢ TẬP DỮ LIỆU & QUY TRÌNH THỰC HIỆN

### 2.1. Nguồn và Quy mô dữ liệu

#### A. Dữ liệu phân khúc khách hàng (`Mall_Customers.csv`)
Dữ liệu được thu thập từ thẻ thành viên của một trung tâm thương mại lớn, gồm **200 bản ghi** khách hàng:

| Tên Cột | Kiểu Dữ Liệu | Thống Kê Mô Tả | Ý Nghĩa Thực Tế | Vai Trò Trong Mô Hình |
| :--- | :---: | :---: | :--- | :--- |
| `CustomerID` | Số nguyên | 1 đến 200 | Mã định danh khách hàng | Định danh (Loại bỏ khi huấn luyện) |
| `Gender` | Chuỗi | 112 Nữ, 88 Nam | Giới tính khách hàng | Phân tích nhân khẩu học |
| `Age` | Số nguyên | $18 \sim 70$ tuổi ($\mu=38.85$) | Độ tuổi của khách hàng | Phân tích mở rộng |
| `Annual Income (k$)` | Số thực | $15 \sim 137$ k$ ($\mu=60.56$) | Thu nhập hàng năm tính theo ngàn USD | **Đặc trưng số học cốt lõi 1** |
| `Spending Score (1-100)`| Số nguyên | $1 \sim 99$ ($\mu=50.20$) | Điểm đánh giá mức độ chi tiêu của siêu thị | **Đặc trưng số học cốt lõi 2** |

#### B. Dữ liệu ảnh phân đoạn (`nature_landscape.jpg`)
Ảnh phong cảnh tự nhiên kích thước $300 \times 200$ pixels chứa các đối tượng tự nhiên đa dạng màu sắc: bầu trời, mặt trời, rặng núi, thảm thực vật và mặt hồ. Sau khi resize về $150 \times 100$, tập dữ liệu pixel có quy mô $15,000$ điểm ảnh trong không gian màu 3 chiều $[R, G, B]$.

### 2.2. Ràng buộc học thuật cốt lõi (Academic Constraints)
- **Zero Scikit-Learn Rule:** Tuyệt đối không import `sklearn.cluster.KMeans`, `sklearn.preprocessing.StandardScaler`, hay `sklearn.metrics.silhouette_score`. Mọi phép toán ma trận, hàm chuẩn hóa, khoảng cách Euclid, và cập nhật trọng tâm đều lập trình thuần túy bằng NumPy và Pandas.
- **Chống rò rỉ thông số (No Information Leakage):** Scaler tính toán Mean và Std, lưu trữ tham số vào file cấu hình JSON. Khi suy luận cho khách hàng mới, chỉ áp dụng công thức chuyển đổi mà không được fit lại trên dữ liệu mới.

### 2.3. Sơ đồ quy trình tổng thể (End-to-End Pipeline)
```text
[Raw Mall Data] ──► [Kiểm tra 6 quy tắc Validation] ──► [Làm sạch Missing/Duplicates]
                              │                                  │
                              ▼                                  ▼
[Triển khai API / CLI] ◄── [Đóng gói JSON Artifact] ◄── [Scaler Standardization Scratch]
        │                                                        │
        ▼                                                        ▼
[Phân tích 5 Personas] ◄── [K-Means Scratch K=5] ◄── [Quét K: Elbow + Silhouette]
        │
        ▼
[Mở rộng: Phân đoạn ảnh K-Means Side-by-Side] ──► [Xuất Báo cáo & Đồ thị]
```

---

## 3. TIỀN XỬ LÝ & CHUẨN HÓA DỮ LIỆU

### 3.1. Kiểm tra 6 quy tắc toàn vẹn dữ liệu (Data Validation)
Pipeline tự động chạy 6 bước kiểm tra tính toàn vẹn:
1. **Kiểm tra Schema:** Xác nhận đầy đủ 5 cột dữ liệu quy chuẩn.
2. **Kiểm tra Missing Values:** Không phát hiện giá trị khuyết thiếu ($0$ missing).
3. **Kiểm tra Bản ghi trùng lặp (Duplicates):** $0$ bản ghi trùng lặp trên cả 5 cột.
4. **Kiểm tra Kiểu dữ liệu:** Đúng định dạng số nguyên và số thực cho các cột định lượng.
5. **Kiểm tra Miền giá trị (Boundary Check):**
   - Tuổi: nằm nghiêm ngặt trong khoảng $[18, 100]$.
   - Điểm chi tiêu: nằm nghiêm ngặt trong khoảng $[1, 100]$.
   - Thu nhập: $> 0$.
6. **Kiểm tra Phân bố nhị phân của Giới tính:** Chỉ chứa hai giá trị `Male` và `Female`.

### 3.2. Trích xuất đặc trưng & Bộ chuẩn hóa Scaler tự xây dựng
Do Thu nhập hàng năm dao động từ $15$ đến $137$ k\$, trong khi Điểm chi tiêu từ $1$ đến $100$, khoảng cách Euclid sẽ bị chi phối bởi biến có phương sai lớn hơn nếu không được chuẩn hóa.

Lớp `Scaler` được thiết kế thủ công gồm 2 phương thức:
1. **Standardization (Z-Score):**
   $$z = \frac{x - \mu}{\sigma}, \quad \mu = \frac{1}{N}\sum_{i=1}^N x_i, \quad \sigma = \sqrt{\frac{1}{N}\sum_{i=1}^N (x_i - \mu)^2}$$
2. **Min-Max Scaling:**
   $$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
3. **Hỗ trợ biến đổi ngược (`inverse_transform`):** Phục hồi tọa độ tâm cụm từ không gian chuẩn hóa về thang đo thực tế (k\$ và điểm) phục vụ diễn giải kinh doanh.

---

## 4. THUẬT TOÁN K-MEANS TỰ LẬP TRÌNH 100% BẰNG NUMPY

### 4.1. Ma trận hóa khoảng cách Euclid bằng Broadcasting
Thay vì dùng 2 vòng lặp `for` lồng nhau gây chậm trễ, thuật toán ứng dụng cơ chế **NumPy Array Broadcasting**:
- Ma trận dữ liệu: $X \in \mathbb{R}^{N \times D} \rightarrow X_{\text{expanded}} \in \mathbb{R}^{N \times 1 \times D}$
- Ma trận tâm cụm: $C \in \mathbb{R}^{K \times D} \rightarrow C_{\text{expanded}} \in \mathbb{R}^{1 \times K \times D}$
- Hiệu ma trận 3 chiều: $\Delta = X_{\text{expanded}} - C_{\text{expanded}} \in \mathbb{R}^{N \times K \times D}$
- Bình phương khoảng cách:
  $$D_{ij}^2 = \sum_{d=1}^D (X_{id} - C_{jd})^2 = \text{np.sum}(\Delta^2, \text{axis}=2) \in \mathbb{R}^{N \times K}$$
- Gán nhãn cực nhanh: $\text{labels} = \text{np.argmin}(D^2, \text{axis}=1)$.

### 4.2. Khởi tạo thông minh K-Means++
Khởi tạo ngẫu nhiên truyền thống rất dễ rơi vào cực trị địa phương (Local Optima). Thuật toán K-Means++ được tự lập trình như sau:
1. Chọn tâm thứ nhất $\mu_1$ ngẫu nhiên từ tập dữ liệu: $\mu_1 \sim \mathcal{U}(X)$.
2. Với mỗi điểm $x_i$, tính khoảng cách ngắn nhất đến các tâm đã chọn:
   $$D(x_i) = \min_{j=1..m} \|x_i - \mu_j\|_2$$
3. Chọn tâm tiếp theo với phân phối xác suất tỉ lệ với bình phương khoảng cách:
   $$P(x_i) = \frac{D(x_i)^2}{\sum_{k=1}^N D(x_k)^2}$$
4. Lặp lại bước 2 và 3 cho đến khi đủ $K$ tâm.

### 4.3. Vòng lặp Lloyd & Xử lý cụm rỗng (Empty Cluster Handling)
- **Assignment Step:** Gán mỗi điểm vào tâm gần nhất: $z_i = \arg\min_k \|x_i - \mu_k\|^2$.
- **Update Step:** Tính trung bình cộng các điểm trong cụm: $\mu_k = \frac{1}{|S_k|}\sum_{x \in S_k} x$.
- **Xử lý cụm rỗng (Re-seeding):** Nếu trong quá trình lặp một cụm không nhận được điểm dữ liệu nào ($|S_k| = 0$), thuật toán tự động tái khởi tạo tâm cụm đó tại điểm dữ liệu có khoảng cách xa nhất so với tâm cụm hiện thời, đảm bảo số cụm luôn ổn định.
- **Tiêu chí dừng hội tụ:** Dừng khi tâm cụm dịch chuyển nhỏ hơn ngưỡng $\text{tol} = 10^{-4}$ hoặc đạt tối đa 300 vòng lặp.

### 4.4. Đo lường Inertia (WCSS) và Hệ số Silhouette
- **Inertia (Within-Cluster Sum of Squares):**
  $$\text{WCSS} = \sum_{k=1}^K \sum_{x \in S_k} \|x - \mu_k\|^2$$
- **Silhouette Coefficient cho điểm $i$:**
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
  - $a(i)$: Khoảng cách trung bình từ điểm $i$ đến các điểm khác trong cùng cụm.
  - $b(i)$: Khoảng cách trung bình nhỏ nhất từ điểm $i$ đến các điểm thuộc cụm khác gần nhất.
  - Xử lý biên: Nếu một cụm chỉ có 1 điểm duy nhất (*singleton*), gán $s(i) = 0.0$ để tránh lỗi phân chia.

---

## 5. THỰC NGHIỆM PHÂN KHÚC KHÁCH HÀNG (CUSTOMER SEGMENTATION)

### 5.1. Quét chọn số cụm tối ưu K (Elbow Method & Silhouette Analysis)
Hệ thống quét $K$ từ 2 đến 10 với $n_{\text{init}} = 10$ lần khởi tạo độc lập:

| Số cụm $K$ | Inertia (WCSS) | Silhouette Score | Số vòng lặp hội tụ | Trạng thái hội tụ |
| :---: | :---: | :---: | :---: | :---: |
| 2 | 269.0168 | 0.2951 | 13 | True |
| 3 | 157.7040 | 0.4666 | 17 | True |
| 4 | 108.9213 | 0.4939 | 14 | True |
| **5 (Tối ưu)** | **65.5684** | **0.5547** | **4** | **True** |
| 6 | 55.0573 | 0.5399 | 5 | True |
| 7 | 44.8648 | 0.5281 | 7 | True |
| 8 | 37.1995 | 0.4541 | 10 | True |
| 9 | 32.8280 | 0.4576 | 8 | True |
| 10 | 29.6724 | 0.4451 | 7 | True |

**Đánh giá:**
- **Đồ thị Elbow:** Tốc độ suy giảm của WCSS chậm lại rõ rệt bắt đầu từ $K = 5$, tạo thành điểm uốn khuỷu tay kinh điển.
- **Silhouette Score:** Đạt giá trị cực đại **0.5547** tại $K = 5$ (các giá trị $K \ge 6$ đều giảm dần do cụm bị chia cắt phân mảnh).
- **Kết luận:** Chọn **$K = 5$** là số lượng phân khúc tối ưu nhất.

### 5.2. Chân dung 5 phân khúc khách hàng (Customer Personas)
Dữ liệu sau khi giải chuẩn hóa (Inverse Transform) về đơn vị thực tế:

| Cụm | Thu nhập TB | Chi tiêu TB | Số lượng | Tỷ lệ (%) | Tên Chân Dung (Persona) | Chiến Lược Tiếp Thị Đề Xuất |
| :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **0** | **86.54 k$** | **82.13/100** | 39 | 19.5% | **VIP / Big Spenders** | Khách hàng chiến lược: Cung cấp dịch vụ chăm sóc cao cấp, thẻ thành viên VIP, thông báo sớm hàng hiệu limited. |
| **1** | **26.30 k$** | **20.91/100** | 23 | 11.5% | **Sensible / Budget** | Nhạy cảm về giá: Đẩy mạnh các chương trình trợ giá, giảm giá nhu yếu phẩm, tặng coupon điểm thưởng. |
| **2** | **25.73 k$** | **79.36/100** | 22 | 11.0% | **Target / Trendsetters** | Giới trẻ phóng khoáng: Tiếp thị sản phẩm thời trang xu hướng, công nghệ mới, chương trình trả góp linh hoạt. |
| **3** | **88.20 k$** | **17.11/100** | 35 | 17.5% | **Careful / Savers** | Thu nhập cao nhưng thận trọng: Đẩy mạnh truyền thông về chất lượng bền bỉ, sản phẩm đầu tư, bảo hiểm tài chính. |
| **4** | **55.30 k$** | **49.52/100** | 81 | 40.5% | **Standard / Moderate** | Tầng lớp trung lưu (Đông nhất): Giữ chân bằng các chương trình khách hàng thân thiết, tích điểm đổi quà thường xuyên. |

---

## 6. THỰC NGHIỆM MỞ RỘNG: PHÂN ĐOẠN ẢNH (IMAGE SEGMENTATION)

Theo yêu cầu bổ sung trong đề bài thực hành (**Practice Exercise 3 - additional**), thuật toán K-Means From-Scratch được ứng dụng trực tiếp để phân đoạn ảnh phong cảnh tự nhiên:

### 6.1. Quy trình xử lý ảnh:
1. Đọc ảnh tự nhiên `nature_landscape.jpg` bằng `PIL` và chuyển đổi sang hệ màu RGB.
2. Resize ảnh về kích thước $150 \times 100$ để xử lý tốc độ cao.
3. Duỗi (flatten) ảnh thành ma trận đặc trưng pixel $(15000, 3)$, với 3 kênh $[R, G, B]$ chuẩn hóa về $[0, 1]$.
4. Huấn luyện `KMeansScratch` với các mức màu $K \in [2, 3, 5, 8]$.
5. Thay thế màu mỗi pixel bằng màu của tâm cụm tương ứng, tái tạo ảnh $(100, 150, 3)$.

### 6.2. Kết quả nén màu & trực quan hóa Side-by-Side:
- **$K = 2$:** Phân đoạn nhị phân tách biệt vùng sáng (Bầu trời, mặt trời) và vùng tối (Núi, hồ nước, thảm cỏ).
- **$K = 3$:** Bắt đầu phân tách rõ 3 mảng: Bầu trời xanh, núi & cây cối xanh đậm, và vùng nước phản chiếu.
- **$K = 5$:** Tách biệt đầy đủ 5 vùng cảnh quan tự nhiên: Bầu trời, Mặt trời vàng, Nét núi non, Cây cỏ ven hồ, và Mặt hồ sâu.
- **$K = 8$:** Tái tạo ảnh mượt mà, giữ trọn vẹn dải gradient màu sắc nhưng dung lượng màu giảm hơn 99% so với ảnh 24-bit gốc.

*Biểu đồ so sánh chi tiết được lưu tại:* `reports/figures/image_segmentation_comparison.png` và báo cáo tại `reports/image_segmentation_report.md`.

---

## 7. ĐÓNG GÓI MÔ HÌNH & TRIỂN KHAI HỆ THỐNG

### 7.1. Tuần tự hóa mô hình (Model Artifact Serialization)
Mô hình sau khi huấn luyện được xuất sang định dạng JSON gọn nhẹ (`models/kmeans_model.json`), bao gồm:
- Metadata huấn luyện (Optimal $K$, Inertia, Silhouette Score).
- Bộ tham số chuẩn hóa `mean_` và `std_` của Scaler.
- Ma trận tọa độ tâm cụm chuẩn hóa và tọa độ tâm cụm gốc.
- Danh sách nhãn định danh Persona tương ứng.

### 7.2. Triển khai Web REST API (Flask)
Dịch vụ Flask microservice trong [`app.py`](file:///d:/Năm%203/Máy%20học/ML_Lab/Lab3_KMeans_Clustering/app.py) cung cấp các endpoint chuẩn:
- `GET /`: Thông tin dịch vụ.
- `GET /health`: Health check trạng thái nạp mô hình.
- `POST /predict`: Nhận JSON đầu vào `{"Annual_Income": 90, "Spending_Score": 85}`, trả về:
  ```json
  {
    "status": "success",
    "cluster": 0,
    "persona": "VIP / Big Spenders (Thu nhập cao - Chi tiêu mạnh)",
    "distance_to_centroid": 0.1729
  }
  ```

### 7.3. Bộ kiểm thử tự động (Unit Tests)
Đạt **15/15 unit tests Passed (100% Green)** kiểm tra tính đúng đắn toán học:
- `test_distance.py`: Kiểm tra tính đúng đắn ma trận Euclid và Broadcasting.
- `test_kmeans.py`: Kiểm tra thuật toán phân cụm trên dữ liệu synthetic.
- `test_metrics.py`: Kiểm tra công thức Inertia và Silhouette (xử lý singleton).
- `test_scaler.py`: Kiểm tra tính chất chuẩn hóa trung bình 0, phương sai 1 và `inverse_transform`.

---

## 8. KẾT LUẬN & HƯỚNG PHÁT TRIỂN

### 8.1. Đóng góp nổi bật của dự án
1. **Tuân thủ học thuật tuyệt đối:** Xây dựng thành công toàn bộ pipeline K-Means Clustering và Image Segmentation từ đầu bằng NumPy thuần túy.
2. **Khắc phục triệt để nhược điểm của K-Means:** Tự cài đặt phân phối xác suất K-Means++ và đa khởi tạo giúp loại bỏ bẫy cực trị địa phương.
3. **Giá trị ứng dụng kép:** Vừa giải quyết bài toán phân khúc khách hàng thương mại điện tử, vừa giải quyết bài toán thị giác máy tính phân đoạn ảnh.
4. **Kiến trúc MLOps chuẩn mực:** Đáp ứng giới hạn $\le 200$ dòng lệnh trên từng file, phân tách module chặt chẽ, kiểm thử tự động và triển khai Web API.

### 8.2. Hướng phát triển tiếp theo
- Thử nghiệm các thuật toán phân cụm dựa trên mật độ (DBSCAN) để phát hiện cụm có hình dạng phi cầu.
- Tích hợp thêm các đặc trưng hành vi số: Tần suất ghé thăm cửa hàng, tỷ lệ mua hàng hoàn trả, giá trị đơn hàng trung bình (AOV).
