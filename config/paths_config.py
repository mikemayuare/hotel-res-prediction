from pathlib import Path

# Data ingestion
RAW_DIR = Path("artifacts/raw")
RAW_FILE_PATH = Path(f"{RAW_DIR}/raw.csv")
TRAIN_FILE_PATH = Path(f"{RAW_DIR}/train.parquet")
TEST_FILE_PATH = Path(f"{RAW_DIR}/test.parquet")

# config
CONFIG_PATH = Path("config/config.yaml")

# data processing
PROCESSED_DIR = Path("artifacts/processed")
PROCESSED_TRAIN_DATA_PATH = Path(f"{PROCESSED_DIR}/processed_train.parquet")
PROCESSED_TEST_DATA_PATH = Path(f"{PROCESSED_DIR}/processed_test.parquet")

# Model training
MODEL_OUPUT_PATH = Path("artifacts/models")
MODEL_FILE_NAME = Path("model.pkl")
