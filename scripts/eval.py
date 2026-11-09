import os, sys, json
import numpy as np, pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import PROCESSED_DATA_PATH, MODEL_PATH
from src.evaluate.metrics import compute_inertia, silhouette_score

def evaluate():
    print("[Evaluation] Danh gia mo hinh K-Means tren tap du lieu da phan cum...")
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    centroids = np.array(meta["centroids_scaled"])
    X_scaled = pd.read_csv(PROCESSED_DATA_PATH).values

    diff = X_scaled[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    labels = np.argmin(np.sum(diff ** 2, axis=2), axis=1)

    inertia = compute_inertia(X_scaled, labels, centroids)
    sil = silhouette_score(X_scaled, labels)

    print("\n" + "=" * 50)
    print("KET QUA DANH GIA MO HINH K-MEANS SCRATCH")
    print("=" * 50)
    print(f"So luong cum (K)     : {meta['metadata']['optimal_k']}")
    print(f"Inertia (WCSS)       : {inertia:.4f}")
    print(f"Silhouette Score     : {sil:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    evaluate()
