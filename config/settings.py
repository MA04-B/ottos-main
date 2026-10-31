import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("OTOSLOTTO_DATA_DIR") or ROOT / "data").resolve()
DATABASE = DATA / "lottery.db"
RAW = DATA / "raw"
MODELS = ROOT / "models" / "trained"
SOURCE = "https://bet.szerencsejatek.hu/cmsfiles/otos.html"


def initialize_directories():
    for directory in (DATA, RAW, DATA / "exports", MODELS, ROOT / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
