import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

API_TOKEN = os.getenv("API_TOKEN", "")
BASE_URL  = "https://api.football-data.org/v4"
HEADERS   = {"X-Auth-Token": API_TOKEN}

ROLLING_WINDOW = 8
BACKFILL_FROM  = "2022-01-01"
BACKFILL_LIMIT = 50

# All WC 2026 teams — populated on first fetch, extended as tournament progresses
WC_COMPETITION_CODE = "WC"
