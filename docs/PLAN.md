# Plan.md — End-to-End K-Means Clustering Pipeline

## 1. Tổng quan

**Dataset:** `Mall_Customers.csv`  
**Bài toán:** Customer Segmentation bằng K-Means Clustering  
**Ràng buộc:** Không sử dụng `scikit-learn`; các thành phần ML cốt lõi phải tự triển khai bằng Python/NumPy/Pandas.

Pipeline:

```text
Raw Data → Ingestion → Validation → EDA → Cleaning → Feature Selection
→ Scaling → K Selection → K-Means Training → Evaluation
→ Cluster Profiling → Visualization → Model Serialization → Inference
```

Mục tiêu là xây dựng pipeline vừa đáp ứng yêu cầu học thuật (hiểu thuật toán từ đầu), vừa có cấu trúc ML Engineering hoàn chỉnh (training, lưu model, inference).

---

## 2. Mục tiêu

### 2.1. Mục tiêu chính

- Tự cài đặt K-Means bằng NumPy.
- Không dùng `sklearn.cluster.KMeans`, `sklearn.preprocessing`, `sklearn.metrics`, `sklearn.decomposition` hoặc các module sklearn khác.
- Tự triển khai Euclidean Distance.
- Tự triển khai Random Initialization và K-Means++.
- Tự triển khai Assignment, Centroid Update, Convergence và Empty Cluster Handling.
- Hỗ trợ Multiple Initialization (`n_init`).
- Tự triển khai Inertia/WCSS và Silhouette Score.
- Dùng Elbow + Silhouette + stability + interpretability để chọn K.
- Lưu model và scaler để inference dữ liệu mới.
- Trực quan hóa và diễn giải các cluster.

### 2.2. Mục tiêu kỹ thuật

```bash
python train.py
```

phải thực hiện được:

```text
Load Data → Validate → EDA/Preprocess → Scale → Select K
→ Train → Evaluate → Save Model → Generate Reports
```

và:

```bash
python predict.py
```

phải thực hiện được:

```text
Load Model → Load Scaler → New Customer → Transform → Predict Cluster
```

---

## 3. Mục đích và ý nghĩa

### 3.1. Kỹ thuật

Hiểu rõ cơ chế K-Means, distance, centroid initialization, local optimum, scaling, lựa chọn K và cách xây dựng pipeline từ raw data đến inference.

### 3.2. Ứng dụng

Customer segmentation có thể hỗ trợ marketing/CRM, ví dụ các nhóm có thu nhập và mức chi tiêu tương đồng. Tên và ý nghĩa cuối cùng của cluster **chỉ được xác định sau khi chạy model và phân tích**, không giả định trước.

---

## 4. Dataset

`Mall_Customers.csv` gồm 200 dòng và 5 cột:

| Column | Type | Ý nghĩa |
|---|---|---|
| `CustomerID` | int | ID khách hàng |
| `Gender` | category | Giới tính |
| `Age` | int | Tuổi |
| `Annual Income (k$)` | int | Thu nhập hàng năm |
| `Spending Score (1-100)` | int | Điểm chi tiêu |

Dataset hiện tại không có missing/duplicate; tuy vậy pipeline vẫn phải có validation tổng quát.

---

## 5. Feature Strategy

### Baseline

```text
Annual Income (k$)
Spending Score (1-100)
```

Hai feature này được chọn làm feature chính vì dễ trực quan hóa và phù hợp với mục tiêu phân khúc theo thu nhập/hành vi chi tiêu.

### Experiment mở rộng

```text
Age + Annual Income + Spending Score
```

`Gender` không đưa trực tiếp vào baseline K-Means vì encoding categorical thành số có thể tạo khoảng cách Euclidean không phù hợp. Có thể dùng `Gender` để **profile cluster sau clustering** hoặc làm experiment riêng.

`CustomerID` luôn bị loại vì chỉ là identifier.

---

## 6. Input / Output

### Input

```text
data/raw/Mall_Customers.csv
```

Configuration tập trung trong `config.py`:

> **[ĐÃ SỬA — Điểm 7: Thêm config.py]**
> Trước đây plan liệt kê các tham số cấu hình nhưng không có nơi tập trung.
> Nếu mỗi module tự đặt giá trị riêng (ví dụ `max_iters = 300` ở file này,
> `max_iters = 500` ở file khác), sẽ rất khó quản lý và dễ mâu thuẫn.
> Giải pháp: tạo một file `config.py` chứa tất cả tham số ở một chỗ duy nhất,
> các module khác chỉ cần `import` từ file này.

```python
# config.py
RANDOM_SEED = 42
K_MIN = 2
K_MAX = 10
MAX_ITERATIONS = 300
TOLERANCE = 1e-4
N_INIT = 10
INITIALIZATION = "kmeans++"
DISTANCE_METRIC = "euclidean"
SCALING_METHOD = "standardization"

# Feature configuration
BASELINE_FEATURES = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]
EXTENDED_FEATURES = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

# Paths
RAW_DATA_PATH = "data/raw/Mall_Customers.csv"
MODEL_PATH = "models/kmeans_model.json"
OUTPUT_DIR = "outputs"
```

### Output

```text
outputs/Mall_Customers_clustered.csv
models/kmeans_model.json
outputs/reports/model_selection.csv
outputs/reports/clustering_metrics.json
outputs/reports/cluster_profile.csv
outputs/figures/elbow_curve.png
outputs/figures/silhouette_curve.png
outputs/figures/clusters_2d.png
outputs/figures/cluster_profile.png
```

Model tối thiểu phải lưu:

```text
algorithm
n_clusters
feature_names
centroids
scaler parameters
scaling method
distance metric
random seed
initialization
n_init
```

---

## 7. Kiến trúc End-to-End

```text
                    RAW DATA
                       │
                       ▼
                 DATA INGESTION
                       │
                       ▼
                DATA VALIDATION
                       │
                       ▼
                       EDA
                       │
                       ▼
                DATA CLEANING
                       │
                       ▼
               FEATURE SELECTION
                       │
                       ▼
                FEATURE SCALING
                       │
                       ▼
                 K SELECTION
                 /           \
             Elbow        Silhouette
                 \           /
                  SELECT K
                     │
                     ▼
              K-MEANS SCRATCH
                     │
          ┌──────────┼──────────┐
          │          │          │
      K-Means++  Assignment   Update
          │          │          │
          └──────────┼──────────┘
                     │
                 Convergence
                     │
                     ▼
              MODEL EVALUATION
                     │
             ┌───────┴────────┐
             ▼                ▼
         Metrics       Visualization
             │                │
             └───────┬────────┘
                     ▼
              CLUSTER PROFILE
                     │
                     ▼
               SAVE MODEL
                     │
                     ▼
              NEW CUSTOMER
                     │
                     ▼
               LOAD SCALER
                     │
                     ▼
              TRANSFORM DATA
                     │
                     ▼
              LOAD CENTROIDS
                     │
                     ▼
             PREDICT CLUSTER
```

---

## 8. Chi tiết từng Phase

> **[ĐÃ SỬA — Điểm 1: Đánh số mục]**
> Trước đây mục này dùng header cấp 1 (`# 8.`) thay vì cấp 2 (`## 8.`),
> khiến cấu trúc phân cấp bị vỡ — các Phase bên trong trông như ngang hàng
> với mục chính. Ngoài ra, đánh số nhảy từ 7 → 10 (thiếu mục 8 và 9).
> Giải pháp: sửa header về đúng cấp 2, và đánh số lại toàn bộ liên tục
> từ 1 đến 19.

### Phase 1 — Data Ingestion

**Input:** `Mall_Customers.csv`  
**Output:** Raw DataFrame.

Công việc:

- Đọc CSV bằng Pandas.
- Kiểm tra file, shape, columns, dtypes.
- Hiển thị một số dòng đầu.

---

### Phase 2 — Data Validation

Kiểm tra:

- Missing values.
- Duplicate rows.
- Data types.
- `Age > 0`.
- `Income > 0`.
- `0 <= Spending <= 100`.
- `CustomerID` unique.

Output là validation report.

---

### Phase 3 — EDA

Thực hiện:

- `describe()`.
- Histogram cho Age, Income, Spending.
- Boxplot để kiểm tra outlier.
- Scatter Income vs Spending.
- Correlation giữa numerical features.

EDA phải trả lời: feature nào có ý nghĩa, có outlier hay không, phân phối thế nào và feature nào có khả năng tạo cluster.

---

### Phase 4 — Data Cleaning

- Missing numerical → median nếu cần.
- Missing categorical → mode nếu cần.
- Duplicate → loại bỏ khi xác định là bản ghi trùng.
- Invalid values → xử lý theo rule đã định nghĩa.

Dataset hiện tại sạch, nên phase này chủ yếu bảo đảm khả năng tái sử dụng pipeline.

---

### Phase 5 — Feature Selection

Baseline:

```python
features = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]
```

Experiment mở rộng có thể thêm `Age`. `CustomerID` không được dùng.

---

### Phase 6 — Feature Scaling

Không dùng `sklearn.preprocessing`.

#### Standardization

```text
z = (x - mean) / std
```

#### Min-Max

```text
x' = (x - min) / (max - min)
```

Không coi scaling là kết luận có sẵn. Thực hiện experiment:

```text
No Scaling vs Standardization vs Min-Max
```

so sánh Inertia, Silhouette, stability và interpretability.

Scaler phải lưu parameters để inference dùng đúng preprocessing.

---

### Phase 7 — Euclidean Distance

Công thức:

```text
d(x,c) = sqrt(Σ(x_i - c_i)^2)
```

Trong assignment có thể dùng squared distance:

```text
d²(x,c) = Σ(x_i - c_i)^2
```

vì không cần căn bậc hai khi chỉ tìm khoảng cách nhỏ nhất.

---

### Phase 8 — Centroid Initialization

#### Random

Chọn K samples ngẫu nhiên.

#### K-Means++

1. Chọn centroid đầu tiên ngẫu nhiên.
2. Tính khoảng cách D(x) tới centroid gần nhất.
3. Xác suất chọn centroid tiếp theo tỉ lệ với D(x)^2.
4. Lặp đến khi đủ K centroid.

K-Means++ là initialization chính; Random được giữ để experiment.

---

### Phase 9 — K-Means Scratch

Đề xuất class:

```python
class KMeansScratch:
    def __init__(
        self,
        n_clusters,
        max_iters=300,
        tol=1e-4,
        init="kmeans++",
        n_init=10,
        random_state=None
    ):
        ...

    def fit(self, X): ...
    def predict(self, X): ...
    def fit_predict(self, X): ...
```

Private methods:

```text
_init_centroids()
_euclidean_distance()
_assign_clusters()
_update_centroids()
_compute_inertia()
```

---

### Phase 10 — Assignment

Với mỗi sample, tính distance tới toàn bộ centroid và chọn centroid gần nhất:

```text
label = argmin(distance)
```

Nên vectorize bằng NumPy broadcasting để code rõ ràng và hiệu quả.

---

### Phase 11 — Centroid Update

Với cluster k:

```text
centroid_k = mean(points assigned to k)
```

---

### Phase 12 — Empty Cluster

Nếu cluster rỗng, không thể tính mean. Strategy đề xuất:

```text
Tìm sample có khoảng cách lớn tới centroid hiện tại
→ dùng sample đó làm centroid mới
```

Strategy phải được cố định và ghi trong documentation.

---

### Phase 13 — Convergence

Sau mỗi iteration:

```text
movement = ||new_centroids - old_centroids||
```

Nếu:

```text
movement < tolerance
```

→ converged.

Nếu đạt `max_iterations` → dừng.

Output:

```text
centroids
labels
iterations
converged
inertia
```

---

### Phase 14 — Multiple Initialization

Chạy model nhiều lần, ví dụ:

```text
n_init = 10
```

Chọn run có:

```text
minimum inertia
```

Mục đích là giảm ảnh hưởng của local optimum.

---

### Phase 15 — Chọn K

Đánh giá:

```text
K = 2 ... 10
```

Mỗi K phải train K-Means và tính Inertia + Silhouette.

Không giả định trước K tối ưu.

---

### Phase 16 — Elbow Method

Inertia/WCSS:

```text
WCSS = Σ Σ ||x_i - centroid_k||²
```

Vẽ Inertia theo K và tìm vùng elbow. Elbow chỉ là một tín hiệu, không phải tiêu chí duy nhất.

---

### Phase 17 — Silhouette Score

Với sample i:

```text
a(i) = average distance tới các điểm khác cùng cluster
b(i) = minimum average distance tới cluster khác
```

```text
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

```text
-1 ≤ s(i) ≤ 1
```

- Gần 1: cluster tốt.
- Gần 0: cluster chồng lấn.
- Nhỏ hơn 0: sample có khả năng thuộc cluster khác.

> **[ĐÃ SỬA — Điểm 5: Xử lý Silhouette cho cluster có 1 sample]**
> Trước đây plan chỉ nói "phải có rule xử lý" nhưng không nói rule đó là gì.
> Vấn đề: nếu cluster chỉ có 1 điểm thì `a(i)` (khoảng cách trung bình đến
> các điểm khác cùng cluster) không tính được vì không có điểm nào khác.
> Nếu không xử lý, code sẽ bị chia cho 0 hoặc trả kết quả vô nghĩa.
> Quy ước chuẩn: gán `s(i) = 0` — nghĩa là "không đủ thông tin để đánh giá
> cluster tốt hay xấu cho điểm này".

**Rule xử lý singleton cluster:** Nếu cluster chỉ có duy nhất 1 sample, `a(i)` không xác định. Trong trường hợp này, gán `s(i) = 0`. Đây là quy ước chuẩn của thuật toán Silhouette — giá trị 0 thể hiện "trung lập", không tốt không xấu, vì không có đủ dữ liệu để đánh giá.

---

### Phase 18 — Model Selection

K cuối cùng dựa trên:

```text
Elbow
+ Silhouette
+ Cluster Size
+ Stability
+ Interpretability
```

Không chọn K chỉ dựa trên một metric.

> **[ĐÃ SỬA — Điểm 6: Định nghĩa Stability Metric]**
> Trước đây plan dùng từ "Stability" như một tiêu chí chọn K nhưng không
> giải thích đo cái gì, đo bằng cách nào. Điều này gây mơ hồ khi triển khai.
> Vấn đề cốt lõi: K-Means phụ thuộc vào centroid khởi tạo ngẫu nhiên — nếu
> mỗi lần chạy cho kết quả rất khác nhau thì clustering đó không đáng tin.
> "Stability" chính là thước đo: "kết quả có ổn định giữa các lần chạy không?"

**Định nghĩa Stability:**

Stability đo mức độ nhất quán của kết quả clustering khi chạy K-Means nhiều lần với các random seed khác nhau. Nếu một giá trị K cho kết quả ổn định (luôn tạo ra các cluster tương tự nhau), nó đáng tin cậy hơn giá trị K mà mỗi lần chạy cho cluster hoàn toàn khác.

**Cách đo:**

```text
1. Với mỗi K, chạy K-Means M lần (ví dụ M = 5) với các seed khác nhau.
2. Thu thập Inertia và Silhouette Score của mỗi lần chạy.
3. Tính độ lệch chuẩn (std) của Inertia và Silhouette qua M lần chạy.
4. Std nhỏ → ổn định → K đáng tin cậy.
   Std lớn → không ổn định → K dễ bị ảnh hưởng bởi initialization.
```

Phương pháp này không cần triển khai thêm metric phức tạp (như ARI) vì chỉ sử dụng Inertia và Silhouette đã có sẵn.

---

### Phase 19 — Final Training

Sau khi chọn `K_optimal`, train final model với:

```text
init = kmeans++
n_init = 10
max_iters = 300
```

Output:

```text
Final centroids
Final labels
Final inertia
Final silhouette
```

---

### Phase 20 — Evaluation

Đánh giá:

- Inertia.
- Silhouette Score.
- Cluster size.
- Stability giữa các random seeds (xem định nghĩa ở Phase 18).
- Tính dễ diễn giải của cluster.

Phân tích cluster quá nhỏ hoặc mất cân bằng.

---

### Phase 21 — Cluster Profiling

Dùng dữ liệu **chưa scale** để diễn giải cluster.

Ví dụ:

```text
Cluster 0
Customers: 35
Avg Income: 40k
Avg Spending: 80
Avg Age: 25
```

`Gender` có thể dùng để profile:

```text
Female: 60%
Male: 40%
```

Không cần đưa Gender vào distance calculation của baseline.

---

### Phase 22 — Visualization

Tạo:

1. Elbow curve.
2. Silhouette curve.
3. Income vs Spending scatter với màu cluster và centroid.
4. Cluster profile bar chart.

> **[ĐÃ SỬA — Điểm 9: Chiến lược visualization cho 3 features]**
> Trước đây plan nói "PCA tự implement bằng SVD là phase mở rộng" nhưng
> không nói nếu không làm PCA thì visualize 3 features bằng cách nào.
> Giải pháp: dùng pairwise scatter — vẽ 3 cặp biểu đồ 2D, mỗi cặp hiển thị
> 2 trong 3 features. Cách này đơn giản, không cần PCA, và vẫn thấy được
> quan hệ giữa các feature.

**Visualization cho experiment 3 features (Age + Income + Spending):**

Nếu không triển khai PCA, dùng pairwise scatter plot — vẽ 3 biểu đồ 2D:

```text
1. Income vs Spending (có sẵn từ baseline)
2. Age vs Income
3. Age vs Spending
```

Mỗi biểu đồ tô màu theo cluster label, hiển thị centroid tương ứng. PCA tự implement bằng SVD/eigendecomposition vẫn là phase mở rộng; không bắt buộc cho baseline.

---

### Phase 23 — Model Serialization

Lưu:

```text
models/kmeans_model.json
```

Ví dụ schema:

```json
{
  "algorithm": "KMeansScratch",
  "n_clusters": 5,
  "feature_names": [
    "Annual Income (k$)",
    "Spending Score (1-100)"
  ],
  "centroids": [],
  "scaler": {
    "method": "standardization",
    "mean": [],
    "std": []
  },
  "distance_metric": "euclidean",
  "init": "kmeans++",
  "n_init": 10,
  "random_seed": 42,
  "version": "1.0"
}
```

---

### Phase 24 — Inference

Input customer mới:

```text
Age = 25
Income = 40
Spending = 75
```

Nếu final model chỉ dùng Income + Spending thì inference cũng chỉ sử dụng đúng hai feature đó.

Pipeline:

```text
New Customer
→ Validation
→ Feature Selection
→ Load Scaler
→ Transform
→ Load Centroids
→ Distance
→ Nearest Centroid
→ Cluster ID
```

---

### Phase 25 — Training / Inference Separation

#### Training

```text
Training Data
→ Fit Scaler
→ Transform
→ Train K-Means
→ Save Centroids + Scaler
```

#### Inference

```text
New Data
→ Load Scaler
→ Transform
→ Load Centroids
→ Predict
```

**Không được fit lại scaler trên dữ liệu inference.**

---

## 9. Project Structure

> **[ĐÃ SỬA — Điểm 3: Cấu trúc project không khớp]**
> Trước đây plan ghi cấu trúc phẳng (`src/kmeans.py`, `src/scaler.py`...)
> nhưng project thực tế dùng thư mục con (`src/EDA/`, `src/Preprocessing/`...).
> Sự mâu thuẫn này gây nhầm lẫn: đọc plan thì làm theo kiểu phẳng, nhìn
> project thì thấy kiểu khác. Giải pháp: cập nhật plan khớp với cấu trúc
> thư mục con đã chọn, đồng thời thêm `config.py` và sửa `evalute` → `evaluate`.

```text
PRAC3 - Clustering/
├── config.py                              ← [MỚI] Tham số tập trung
├── train.py                               ← Entry point training
├── predict.py                             ← Entry point inference
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   │   └── Mall_Customers.csv
│   ├── processed/
│   │   └── X_scaled.csv
│   ├── EDA/                               ← Output EDA (figures, reports)
│   └── training/                          ← Output training data
├── src/
│   ├── __init__.py
│   ├── EDA/
│   │   ├── __init__.py
│   │   ├── data_loader.py                 ← CSV → DataFrame
│   │   ├── validation.py                  ← Data validation
│   │   └── eda.py                         ← EDA analysis
│   ├── Preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaning.py                    ← Missing, duplicate, invalid
│   │   ├── feature_selection.py           ← Feature selection
│   │   └── scaler.py                      ← Manual scaler (fit/transform)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── distance.py                    ← Euclidean distance
│   │   ├── initialization.py             ← Random + K-Means++
│   │   └── kmeans.py                      ← KMeansScratch class
│   └── evaluate/
│       ├── __init__.py
│       ├── metrics.py                     ← Inertia + Silhouette
│       ├── model_selection.py             ← Evaluate K, select K
│       ├── visualization.py               ← Charts
│       └── profiling.py                   ← Cluster profiling
├── models/
│   └── kmeans_model.json
├── outputs/
│   ├── Mall_Customers_clustered.csv
│   ├── reports/
│   │   ├── model_selection.csv
│   │   ├── clustering_metrics.json
│   │   └── cluster_profile.csv
│   └── figures/
│       ├── elbow_curve.png
│       ├── silhouette_curve.png
│       ├── clusters_2d.png
│       └── cluster_profile.png
├── tests/
│   ├── test_scaler.py
│   ├── test_distance.py
│   ├── test_kmeans.py
│   └── test_metrics.py
├── notebooks/
│   └── 01_eda_and_clustering.ipynb
└── docs/
    └── PLAN.md
```

---

## 10. Module Responsibilities

> **[ĐÃ SỬA — cập nhật theo cấu trúc thư mục con]**

| Component | Module | Responsibility |
|---|---|---|
| **EDA** | `data_loader.py` | CSV → DataFrame |
| **EDA** | `validation.py` | Data validation |
| **EDA** | `eda.py` | EDA analysis và visualization |
| **Preprocessing** | `cleaning.py` | Missing, duplicate, invalid handling |
| **Preprocessing** | `feature_selection.py` | Feature selection |
| **Preprocessing** | `scaler.py` | Manual scaler (fit/transform/save) |
| **Training** | `distance.py` | Euclidean distance |
| **Training** | `initialization.py` | Random + K-Means++ |
| **Training** | `kmeans.py` | KMeansScratch class |
| **Evaluate** | `metrics.py` | Inertia + Silhouette |
| **Evaluate** | `model_selection.py` | Evaluate K, select K, stability |
| **Evaluate** | `visualization.py` | Charts |
| **Evaluate** | `profiling.py` | Cluster profiling |
| **Root** | `config.py` | Tham số tập trung |
| **Root** | `train.py` | Entry point training pipeline |
| **Root** | `predict.py` | Entry point inference |

---

## 11. Testing Plan

### Unit Tests

#### Scaler

Kiểm tra mean gần 0 và std gần 1 sau Standardization.

#### Distance

```text
distance([0,0], [3,4]) = 5
```

#### K-Means

Dataset nhỏ:

```text
[1,1]
[1,2]
[9,9]
[10,9]
```

Expected: 2 clusters.

### Synthetic Dataset

Tạo Gaussian clusters bằng NumPy với centroid biết trước để kiểm tra thuật toán.

### Edge Cases

```text
K = 1
Empty cluster
Duplicate points
Identical points
K > number of samples
```

---

## 12. Experiment Plan

### Experiment 1 — Feature Selection

```text
A: Income + Spending
B: Age + Income + Spending
```

So sánh chất lượng và interpretability.

### Experiment 2 — Scaling

```text
No Scaling
Standardization
Min-Max
```

### Experiment 3 — Initialization

```text
Random
K-Means++
```

### Experiment 4 — Multiple Initialization

```text
n_init = 1
n_init = 10
```

### Experiment 5 — Number of Clusters

```text
K = 2 ... 10
```

Metrics:

```text
Inertia
Silhouette
Cluster Size
Stability (std của Inertia/Silhouette qua M lần chạy)
```

---

## 13. Reproducibility

Tất cả experiment phải hỗ trợ `random_seed`, ví dụ:

```python
random_state = 42
```

Cùng dataset + parameters + seed phải cho kết quả có tính tái lập.

---

## 14. Risks & Mitigation

| Risk | Mitigation |
|---|---|
| Local optimum | K-Means++ + multiple initialization |
| Empty cluster | Reinitialize empty centroid |
| Wrong K | Elbow + Silhouette + stability + interpretability |
| Feature scale imbalance | Compare scaling strategies |
| Outlier | EDA + sensitivity analysis |
| Implementation bug | Unit test + synthetic data |
| Data leakage | Fit scaler only on training data |
| Inference mismatch | Save feature names + scaler parameters |
| Unstable clustering | Test multiple random seeds, đo stability (std) |

---

## 15. Definition of Done

### Data

- [ ] Dataset load thành công.
- [ ] Validation hoàn thành.
- [ ] Missing/duplicate được kiểm tra.
- [ ] EDA được thực hiện.
- [ ] Feature strategy được xác định.

### Preprocessing

- [ ] CustomerID bị loại.
- [ ] Scaler tự implement.
- [ ] Scaler parameters được lưu.
- [ ] Training và inference dùng cùng preprocessing.

### K-Means

- [ ] Euclidean distance tự implement.
- [ ] Random initialization tự implement.
- [ ] K-Means++ tự implement.
- [ ] Assignment tự implement.
- [ ] Centroid update tự implement.
- [ ] Empty cluster được xử lý.
- [ ] Convergence được xử lý.
- [ ] Multiple initialization được hỗ trợ.

### Evaluation

- [ ] Inertia tự implement.
- [ ] Elbow Method.
- [ ] Silhouette Score tự implement.
- [ ] Singleton cluster Silhouette = 0.
- [ ] K được chọn dựa trên nhiều tiêu chí.
- [ ] Stability metric hoạt động.

### Model

- [ ] Model được lưu.
- [ ] Centroids được lưu.
- [ ] Scaler được lưu.
- [ ] Feature names được lưu.
- [ ] Configuration/random seed được lưu.

### Inference

- [ ] Model load lại được.
- [ ] Customer mới predict được.
- [ ] Không cần train lại model khi inference.

### Testing & Reporting

- [ ] Unit tests pass.
- [ ] Synthetic tests pass.
- [ ] Edge cases pass.
- [ ] Elbow plot.
- [ ] Silhouette plot.
- [ ] Cluster visualization.
- [ ] Cluster profile.
- [ ] Business interpretation.
- [ ] README.
- [ ] Plan.md.

---

## 16. Thứ tự triển khai thực tế

```text
01. Data Loading                    → src/EDA/data_loader.py
02. Data Validation                 → src/EDA/validation.py
03. EDA                             → src/EDA/eda.py
04. Data Cleaning                   → src/Preprocessing/cleaning.py
05. Feature Selection               → src/Preprocessing/feature_selection.py
06. Manual Scaler                   → src/Preprocessing/scaler.py
07. Euclidean Distance              → src/training/distance.py
08. Random Initialization           → src/training/initialization.py
09. Basic K-Means                   → src/training/kmeans.py
10. Convergence                     → src/training/kmeans.py
11. Empty Cluster Handling          → src/training/kmeans.py
12. K-Means++                       → src/training/initialization.py
13. Multiple Initialization         → src/training/kmeans.py
14. Inertia                         → src/evaluate/metrics.py
15. Silhouette Score                → src/evaluate/metrics.py
16. K Selection + Stability         → src/evaluate/model_selection.py
17. Final Training                  → train.py
18. Cluster Profiling               → src/evaluate/profiling.py
19. Visualization                   → src/evaluate/visualization.py
20. Model Serialization             → train.py (save) + predict.py (load)
21. Inference                       → predict.py
22. Unit Testing                    → tests/
23. Experiments                     → notebooks/
24. Final Report                    → README.md
```

---

## 17. Final Success Criteria

Pipeline hoàn chỉnh phải chứng minh được 4 vấn đề:

### 1. Thuật toán đúng

```text
K-Means Scratch → Synthetic Test → Correct Clustering
```

### 2. K được lựa chọn có cơ sở

```text
Elbow + Silhouette + Stability (std) + Interpretability
```

### 3. Model có thể tái sử dụng

```text
Train → Save → Load → New Customer → Predict
```

### 4. Kết quả có ý nghĩa

```text
Cluster → Statistical Profile → Customer Persona → Business Interpretation
```

---

## 18. Quy tắc kỹ thuật bắt buộc

Không được sử dụng:

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
```

Các thành phần tương ứng phải được tự triển khai.

Được phép sử dụng:

```text
Python 3.x
NumPy
Pandas
Matplotlib
Seaborn (nếu cần visualization)
unittest/pytest
JSON
```

---

## 19. Tổng hợp các điểm đã sửa so với bản gốc

| # | Vấn đề | Giải pháp |
|---|---|---|
| 1 | Đánh số mục nhảy từ 7 → 10, header Phase sai cấp | Đánh số lại liên tục 1–19, Phase dùng `###` thay vì `##` |
| 2 | Thư mục `src/evalute/` bị lỗi chính tả | Đã đổi tên thành `src/evaluate/` |
| 3 | Cấu trúc project trong plan (phẳng) khác project thực tế (thư mục con) | Cập nhật plan theo cấu trúc thư mục con, thêm mapping file → module |
| 4 | Thiếu thư mục `outputs/` và `tests/` | Đã thêm vào project structure (tạo khi chạy lần đầu) |
| 5 | Silhouette singleton cluster chỉ nói "cần rule" nhưng không nói rule gì | Quy định rõ: `s(i) = 0` cho cluster chỉ có 1 sample |
| 6 | Stability metric được liệt kê nhưng không định nghĩa cách đo | Định nghĩa: std của Inertia/Silhouette qua M lần chạy với seed khác nhau |
| 7 | Tham số cấu hình phân tán, không có file tập trung | Thêm `config.py` vào project structure với toàn bộ tham số |
| 8 | `train.py` và `predict.py` chưa tồn tại | Xác nhận đúng — sẽ tạo khi bắt đầu triển khai |
| 9 | Experiment 3 features không có chiến lược visualization nếu không làm PCA | Thêm pairwise scatter plot (3 cặp 2D) làm giải pháp thay thế |

Mục tiêu cuối cùng là một **K-Means End-to-End Pipeline from Scratch**, trong đó phần thuật toán và các thành phần ML cốt lõi do project tự xây dựng, từ raw data → preprocessing → model selection → training → evaluation → model artifact → inference.
