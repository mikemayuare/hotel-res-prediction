from pathlib import Path

# Data ingestion
RAW_DIR = Path("artifacts/raw")
RAW_FILE_PATH = Path(f"{RAW_DIR}/raw.csv")
TRAIN_FILE_PATH = Path(f"{RAW_DIR}/train.csv")
TEST_FILE_PATH = Path(f"{RAW_DIR}/test.csv")

# config
CONFIG_PATH = Path("config/config.yaml")
