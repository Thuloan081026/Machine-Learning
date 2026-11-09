import sys
from pathlib import Path

# Thêm thư mục Lab3_KMeans_Clustering vào sys.path để pytest import được src.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

