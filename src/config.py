from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "datos"

SALES_START_DATE = date(2026, 6, 1)
SALES_END_DATE = date(2026, 8, 31)
INVENTORY_SNAPSHOT_DATE = date(2026, 9, 1)