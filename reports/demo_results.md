# KẾT QUẢ THỰC NGHIỆM DEMO — LAB 3 (CUSTOMER SEGMENTATION)

## 1. Bảng Đánh Giá Chất Lượng Mô Hình K-Means Scratch

| Thuật toán | Số cụm (K) | Inertia (WCSS) | Silhouette Score | Số vòng lặp | Hội tụ |
|---|---|---|---|---|---|
| **KMeansScratch** | 5 | 65.5684 | 0.5547 | 4 | Có (True) |

## 2. Kết Quả Dự Báo Các Kịch Bản Khách Hàng

### Khách hàng VIP (Thu nhập cao, chi tiêu bạo)
- **Đầu vào:** Thu nhập = `90k$`, Chi tiêu = `85/100`
- **Cụm:** `Cụm 0`
- **Chân dung:** **VIP / Big Spenders (Thu nhập cao - Chi tiêu mạnh)**
- **Khoảng cách tới tâm:** `0.1729`

### Khách hàng Tiết kiệm (Thu nhập cao, chi tiêu ít)
- **Đầu vào:** Thu nhập = `95k$`, Chi tiêu = `15/100`
- **Cụm:** `Cụm 3`
- **Chân dung:** **Careful / Savers (Thu nhập cao - Tiết kiệm/Thận trọng)**
- **Khoảng cách tới tâm:** `0.2722`

### Khách hàng Giới trẻ / Trendsetter (Thu nhập thấp, thích mua sắm)
- **Đầu vào:** Thu nhập = `25k$`, Chi tiêu = `80/100`
- **Cụm:** `Cụm 2`
- **Chân dung:** **Target / Trendsetters (Thu nhập thấp - Chi tiêu phóng khoáng)**
- **Khoảng cách tới tâm:** `0.0372`

### Khách hàng Dè dặt / Hạn hẹp (Thu nhập thấp, chi tiêu thấp)
- **Đầu vào:** Thu nhập = `20k$`, Chi tiêu = `20/100`
- **Cụm:** `Cụm 1`
- **Chân dung:** **Sensible / Budget (Thu nhập thấp - Chi tiêu dè dặt)**
- **Khoảng cách tới tâm:** `0.2432`

### Khách hàng Trung lưu / Tiêu chuẩn (Thu nhập và chi tiêu trung bình)
- **Đầu vào:** Thu nhập = `55k$`, Chi tiêu = `50/100`
- **Cụm:** `Cụm 4`
- **Chân dung:** **Standard / Moderate (Thu nhập & Chi tiêu trung bình)**
- **Khoảng cách tới tâm:** `0.0218`

