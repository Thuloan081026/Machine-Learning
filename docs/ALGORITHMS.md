# Thuật toán — Lab 3: K-Means Clustering & Customer Segmentation

- **KMeansScratch:** Thuật toán phân cụm K-Means thuần NumPy với K-Means++ và xử lý cụm rỗng.
- **Khoảng cách ma trận hóa:** Tối ưu hóa 3D Broadcasting cho khoảng cách Euclid và Manhattan.
- **Scaler:** Chuẩn hóa dữ liệu theo Standardization và MinMax.
- **Đo lường chất lượng phân cụm:** WCSS (Inertia), Silhouette Score, Davies-Bouldin Index.
- **Lượng tử hóa màu ảnh:** Phân vùng ảnh (Color Quantization) trong không gian màu 3D RGB.

Các thuật toán cốt lõi được viết 100% bằng NumPy và Pandas, không dùng scikit-learn.

Mã nguồn thuật toán nằm trong `models/algorithms/`; các model đã huấn luyện được lưu trong `models/save_model/`.

---

## 1. TÍNH TOÁN KHOẢNG CÁCH EUCLID MA TRẬN HÓA (VECTORIZED DISTANCE)

### 1.1. Công thức toán học
Khoảng cách Euclid giữa điểm dữ liệu $x_i \in \mathbb{R}^D$ và tâm cụm $\mu_k \in \mathbb{R}^D$:
$$d(x_i, \mu_k) = \|x_i - \mu_k\|_2 = \sqrt{\sum_{j=1}^D (x_{ij} - \mu_{kj})^2}$$
Bình phương khoảng cách Euclid (Squared Euclidean Distance):
$$d^2(x_i, \mu_k) = \|x_i - \mu_k\|_2^2 = \sum_{j=1}^D (x_{ij} - \mu_{kj})^2$$

### 1.2. Tối ưu hóa Siêu tốc bằng 3D NumPy Broadcasting
Thay vì sử dụng 2 vòng lặp lồng nhau qua $N$ điểm dữ liệu và $K$ cụm (vốn cực chậm trên Python thuần với độ phức tạp $O(N \cdot K \cdot D)$ khi xử lý hàng trăm nghìn pixel ảnh):
```python
# X có kích thước (N, D) -> mở rộng chiều thành (N, 1, D)
# centroids có kích thước (K, D) -> mở rộng chiều thành (1, K, D)
diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # Shape: (N, K, D)
dist_sq = np.sum(diff ** 2, axis=-1)                     # Shape: (N, K)
distances = np.sqrt(dist_sq)                             # Shape: (N, K)
```
- **Nguyên lý hoạt động:** NumPy tự động nhân bản (broadcast) dữ liệu trên bộ nhớ RAM ở tầng C/Fortran thông qua các thư viện đại số tuyến tính BLAS/LAPACK.
- **Tốc độ:** Thực hiện toàn bộ phép tính khoảng cách cho hàng vạn điểm chỉ trong vài chục mili-giây, giảm tải $95\%$ thời gian so với vòng lặp truyền thống.

---

## 2. THUẬT TOÁN KHỞI TẠO TÂM CỤM K-MEANS++ (`initialization.py`)

### 2.1. Hạn chế của Khởi tạo ngẫu nhiên (Random Initialization)
Khi chọn ngẫu nhiên $K$ điểm bất kỳ làm tâm cụm ban đầu, có xác suất cao các tâm này nằm sát nhau trong cùng một cụm tự nhiên. Khi đó, thuật toán lặp Lloyd sẽ bị mắc kẹt tại **Cực tiểu địa phương (Local Minima)**, dẫn tới việc chia sai ranh giới các cụm.

### 2.2. Quy trình thuật toán K-Means++
K-Means++ đưa ra cơ chế chọn tâm ban đầu sao cho các tâm **tản ra xa nhau nhất có thể** theo phân phối xác suất:
1. **Bước 1:** Chọn tâm cụm đầu tiên $c_1$ ngẫu nhiên đều từ tập dữ liệu $X$:
   $$c_1 \sim \mathcal{U}(X)$$
2. **Bước 2:** Với mỗi điểm $x \in X$, tính khoảng cách ngắn nhất từ nó tới các tâm đã được chọn trước đó:
   $$D(x) = \min_{j \in \{1, \dots, m\}} \|x - c_j\|_2$$
3. **Bước 3:** Chọn tâm tiếp theo $c_{m+1} = x'$ từ tập dữ liệu với xác suất tỷ lệ thuận với bình phương khoảng cách:
   $$P(x) = \frac{D(x)^2}{\sum_{x' \in X} D(x')^2}$$
   *(Điểm nào càng ở xa các tâm đã có thì xác suất được chọn làm tâm mới càng cao)*.
4. **Bước 4:** Lặp lại Bước 2 và Bước 3 cho đến khi chọn đủ $K$ tâm cụm.
- **Bảo đảm toán học:** Arthur & Vassilvitskii (2007) đã chứng minh K-Means++ đảm bảo kỳ vọng sai số WCSS đạt giới hạn xấp xỉ $O(\log K)$ lần so với phân cụm tối ưu toàn cục.

---

## 3. THUẬT TOÁN LẶP LLOYD & CƠ CHẾ XỬ LÝ CỤM RỖNG (`kmeans.py`)

### 3.1. Hàm mục tiêu WCSS (Within-Cluster Sum of Squares / Inertia)
Thuật toán K-Means giải bài toán tối ưu hóa không lồi nhằm tìm tập hợp $K$ cụm $C = \{C_1, C_2, \dots, C_K\}$ và các tâm $\mu = \{\mu_1, \dots, \mu_K\}$ tối thiểu hóa tổng bình phương khoảng cách nội cụm:
$$\mathcal{J}(C, \mu) = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|_2^2$$

### 3.2. Chu trình Expectation-Maximization (EM)
1. **Bước Gán cụm (Assignment Step / Expectation):**
   Gán mỗi điểm dữ liệu $x_i$ vào cụm có tâm gần nó nhất:
   $$z_i = \arg\min_{k \in \{1, \dots, K\}} \|x_i - \mu_k\|_2^2$$
2. **Bước Cập nhật tâm (Update Step / Maximization):**
   Tính toán lại tọa độ tâm cụm bằng trung bình cộng tọa độ của tất cả các điểm được gán vào cụm đó:
   $$\mu_k = \frac{1}{|C_k|} \sum_{i: z_i = k} x_i, \quad \text{với } C_k = \{x_i \mid z_i = k\}$$
3. **Điều kiện Hội tụ (Convergence Criteria):**
   Thuật toán dừng lặp khi chuẩn Frobenius của sự thay đổi tọa độ tâm cụm nhỏ hơn ngưỡng dung sai `tol`:
   $$\|\mu^{(t)} - \mu^{(t-1)}\|_F = \sqrt{\sum_{k=1}^K \sum_{j=1}^D (\mu_{kj}^{(t)} - \mu_{kj}^{(t-1)})^2} < \text{tol} \quad (\text{hoặc khi đạt } \text{max\_iter} = 300)$$

### 3.3. Cơ chế Khắc phục Cụm Rỗng (Empty Cluster Re-seeding)
Trong quá trình lặp, có thể xảy ra trường hợp một cụm không được gán bất kỳ điểm dữ liệu nào ($|C_k| = 0$). Phép tính $\mu_k = \frac{1}{0}\sum x_i$ sẽ sinh lỗi `NaN` (Zero Division Error).
- **Thuật toán xử lý From-Scratch:**
  1. Kiểm tra mảng nhãn: nếu cụm $k$ có số lượng phần tử bằng 0.
  2. Tìm điểm dữ liệu $x^*$ trong toàn bộ tập dữ liệu đang có khoảng cách tới tâm cụm hiện tại của nó là lớn nhất (điểm dị biệt nhất).
  3. Tái khởi tạo tâm $\mu_k \leftarrow x^*$ để cụm rỗng này thu nạp các điểm ngoại lai.

---

## 4. ĐO LƯỜNG CHẤT LƯỢNG PHÂN CỤM & XÁC ĐỊNH SỐ CỤM TỐI ƯU

### 4.1. Phương pháp Điểm gập Khuỷu tay (Elbow Method with WCSS)
Quét số lượng cụm $K \in [2, 10]$ và vẽ đồ thị WCSS theo $K$. Đường cong sẽ dốc đứng khi $K$ nhỏ và thoải dần khi $K$ lớn. Điểm gập khuỷu tay (Elbow point) là giá trị $K$ mà tại đó tốc độ giảm WCSS bắt đầu bão hòa.

### 4.2. Hệ số Silhouette Phân tích Chi tiết (`metrics.py`)
Đo lường mức độ tương đồng của một điểm đối với cụm của chính nó (độ gắn kết) so với các cụm khác (độ tách biệt).
Với mỗi điểm dữ liệu $i$:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
Trong đó:
- $a(i)$ (Intra-cluster distance): Khoảng cách trung bình từ điểm $i$ tới tất cả các điểm khác trong cùng cụm $C_A$:
  $$a(i) = \frac{1}{|C_A| - 1} \sum_{j \in C_A, j \ne i} \|x_i - x_j\|_2$$
- $b(i)$ (Nearest-cluster distance): Khoảng cách trung bình từ điểm $i$ tới các điểm thuộc cụm gần nhất tiếp theo $C_B \ne C_A$:
  $$b(i) = \min_{C_B \ne C_A} \frac{1}{|C_B|} \sum_{j \in C_B} \|x_i - x_j\|_2$$
- **Xử lý ngoại lệ Singleton:** Nếu một cụm chỉ có duy nhất 1 điểm ($|C_A| = 1$), quy ước $s(i) = 0.0$ để tránh lỗi chia cho 0.
- **Silhouette trung bình toàn thể:** $\bar{S} = \frac{1}{N} \sum_{i=1}^N s(i) \in [-1, +1]$. Điểm đạt đỉnh cao nhất tại **$K = 5$** ($\bar{S} \approx 0.55$), khẳng định 5 cụm là cấu trúc tự nhiên hoàn hảo nhất của tập Mall Customers.

---

## 5. THUẬT TOÁN LƯỢNG TỬ HÓA MÀU SẮC & PHÂN VÙNG ẢNH (`image_segmentation.py`)

### 5.1. Bản chất Phân cụm Không gian Màu 3D
Mỗi bức ảnh màu kỹ thuật số có kích thước không gian $(H, W, 3)$, trong đó mỗi điểm ảnh (pixel) là một vector 3 chiều:
$$p_i = [R_i, G_i, B_i] \in [0.0, 1.0]^3$$
Bức ảnh chứa $N = H \times W$ điểm ảnh tương ứng với một tập dữ liệu gồm $N$ mẫu trong không gian Euclid 3 chiều $\mathbb{R}^3$.

### 5.2. Quy trình Lượng tử hóa (Color Quantization Pipeline)
1. **Tiền xử lý:** Đọc ảnh bằng PIL/OpenCV, chuẩn hóa giá trị kênh màu từ $[0, 255]$ về đoạn thực $[0.0, 1.0]$ bằng phép chia cho $255.0$.
2. **Trải phẳng (Flattening):** Biến đổi mảng 3D $(H, W, 3)$ thành ma trận 2D $(N, 3)$.
3. **Phân cụm K-Means Scratch:** Chạy thuật toán K-Means++ với số lượng màu chủ đạo $K \in \{2, 3, 5, 8\}$. Sau khi hội tụ, thu được $K$ vector tâm cụm màu:
   $$\mu_k = [R_k^*, G_k^*, B_k^*], \quad k \in \{1, \dots, K\}$$
4. **Ánh xạ màu (Pixel Replacement):** Mỗi điểm ảnh thứ $i$ được gán lại bằng giá trị màu của tâm cụm tương ứng:
   $$I_{\text{quantized}}[i] = \mu_{z_i}$$
5. **Tái tạo không gian (Spatial Reshaping):** Reshape ma trận $(N, 3)$ trở lại định dạng ảnh $(H, W, 3)$ và nhân với $255.0$ để xuất ảnh phân vùng.
- **Ý nghĩa thực tiễn:** Nén giảm dung lượng dữ liệu ảnh, trích xuất dải màu đại diện (Color Palette Extraction) và hỗ trợ thị giác máy tính tách biệt vật thể nền/tiền cảnh.
