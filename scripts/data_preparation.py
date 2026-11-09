import os, sys
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, FEATURE_COLS
from src.EDA.data_loader import load_dataset
from src.EDA.validation import validate_dataset
from src.Preprocessing.cleaning import clean_customer_data
from src.Preprocessing.feature_selection import extract_features
from src.Preprocessing.scaler import Scaler

def prepare_data():
    print("[Data Preparation] Dang nap du lieu tu:", RAW_DATA_PATH)
    df = load_dataset(RAW_DATA_PATH)
    validate_dataset(df)
    df_clean = clean_customer_data(df)
    X, _ = extract_features(df_clean, FEATURE_COLS)
    scaler = Scaler(method="standardization")
    X_scaled = scaler.fit_transform(X)
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    pd.DataFrame(X_scaled, columns=FEATURE_COLS).to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"[Data Preparation] Thanh cong! Da luu tap dac trung vao: {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    prepare_data()
