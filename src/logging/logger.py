import logging
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path("logs")
Path.mkdir(LOGS_DIR, exist_ok=True)

LOF_FILE = Path(f"{LOGS_DIR}/log_{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    filename=LOF_FILE,
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
