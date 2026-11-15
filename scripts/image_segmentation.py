import os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.kmeans import KMeansScratch

def load_and_preprocess_image(image_path: str, max_size=(150, 150)):
    """Task 1 & 2: Load Image, convert color space, resize, flatten to 2D array."""
    img = Image.open(image_path).convert("RGB")
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float64) / 255.0
    h, w, c = img_array.shape
    pixel_features = img_array.reshape(-1, c)
    return img_array, pixel_features, (h, w, c)

def segment_image_with_kmeans(pixel_features: np.ndarray, shape: tuple, k: int, random_state=42):
    """Task 3 & 4: K-Means Clustering & Reconstruct segmented image."""
    h, w, c = shape
    kmeans = KMeansScratch(n_clusters=k, init="k-means++", max_iter=100, random_state=random_state)
    kmeans.fit(pixel_features)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    # Thay the moi pixel bang mau cua tam cum tuong ung
    segmented_pixels = centroids[labels]
    segmented_image = (segmented_pixels.reshape(h, w, c) * 255.0).clip(0, 255).astype(np.uint8)
    return segmented_image, kmeans.inertia_

def run_image_segmentation_pipeline(image_path=None, k_values=[2, 3, 5, 8]):
    """Task 5: Chay phan doan anh va truc quan hoa side-by-side."""
    if image_path is None:
        image_path = ROOT / "data" / "raw" / "nature_landscape.jpg"

    print("=" * 70)
    print("PRACTICE EXERCISE 3 (ADDITIONAL): K-MEANS IMAGE SEGMENTATION")
    print("=" * 70)
    print(f"[+] 1. Đọc và tiền xử lý ảnh từ: {image_path}")
    orig_img, pixel_features, shape = load_and_preprocess_image(str(image_path))
    h, w, c = shape
    print(f"[+] 2. Kích thước ảnh sau resize: {w}x{h} ({pixel_features.shape[0]} pixels, 3 channels RGB)")

    fig, axes = plt.subplots(1, len(k_values) + 1, figsize=(4 * (len(k_values) + 1), 4))
    axes[0].imshow((orig_img * 255.0).astype(np.uint8))
    axes[0].set_title(f"Ảnh Gốc (Original)\n{w}x{h}", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    md_report = [
        "# BÁO CÁO PHÂN ĐOẠN ẢNH BẰNG K-MEANS (IMAGE SEGMENTATION)\n\n",
        f"- **Đường dẫn ảnh:** `{image_path}`\n",
        f"- **Kích thước xử lý:** `{w} x {h}` ({w * h} pixels)\n",
        "- **Không gian màu:** RGB (3 đặc trưng: Red, Green, Blue)\n\n",
        "| Số Cụm Màu (K) | Inertia (WCSS) | Tỷ Lệ Nén Màu | Nhận Xét |\n",
        "|---|---|---|---|\n"
    ]

    for idx, k in enumerate(k_values, start=1):
        print(f"[*] 3 & 4. Tiến hành phân đoạn ảnh với K = {k} màu...")
        segmented_img, inertia = segment_image_with_kmeans(pixel_features, shape, k)
        axes[idx].imshow(segmented_img)
        axes[idx].set_title(f"Segmented (K={k})\nWCSS: {inertia:.1f}", fontsize=11, fontweight="bold")
        axes[idx].axis("off")
        md_report.append(f"| **K = {k}** | {inertia:.2f} | Nén về {k} màu | Tách biệt các vùng màu chủ đạo |\n")

    plt.suptitle("K-Means Image Segmentation (From-Scratch Lloyd Algorithm)", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()

    # Luu ket qua truc quan
    out_dir = ROOT / "reports" / "figures"
    rep_dir = ROOT / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "image_segmentation_comparison.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.savefig(rep_dir / "image_segmentation_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Luu bao cao markdown
    report_file = ROOT / "reports" / "image_segmentation_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.writelines(md_report)
        f.write("\n![Image Segmentation Comparison](figures/image_segmentation_comparison.png)\n")

    print(f"[+] 5. Đã xuất biểu đồ so sánh side-by-side: {fig_path}")
    print(f"[+] Đã lưu báo cáo markdown tại: {report_file}")
    print("=" * 70)

if __name__ == "__main__":
    run_image_segmentation_pipeline()

