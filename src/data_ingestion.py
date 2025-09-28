import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage
from sklearn.model_selection import train_test_split

from config.paths_config import (
    CONFIG_PATH,
    RAW_DIR,
    RAW_FILE_PATH,
    TEST_FILE_PATH,
    TRAIN_FILE_PATH,
)
from src.custom_exceptions import CustomException
from src.logger import get_logger
from utils.common_functions import read_yaml

load_dotenv()
creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

logger = get_logger(__name__)


class DataIngestion:
    def __init__(self, config):
        self.config = config
        self.bucket_name = self.config["data_ingestion"]["bucket_name"]
        self.file_name = self.config["data_ingestion"]["bucket_file_name"]
        self.train_ratio = self.config["data_ingestion"]["train_ratio"]
        self.target = self.config["data_processing"]["target"]

        RAW_DIR.mkdir(exist_ok=True)
        logger.info(
            "Data ingestion started with %s and file is %s",
            self.bucket_name,
            self.file_name,
        )

    def download_csv_from_gcp(self):
        try:
            client = storage.Client.from_service_account_json(creds)
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            blob.download_to_filename(RAW_FILE_PATH)

            logger.info("Raw file successfully downloaded to %s", RAW_FILE_PATH)

        except Exception as e:
            logger.error("Error while downloading the csv file - %s", str(e))
            raise CustomException("Failed to download csv file") from e

    def split_data(self):
        try:
            logger.info("Starting the data splitting")
            data = pd.read_csv(RAW_FILE_PATH)
            x = data.drop(columns=[self.target])
            y = data[self.target]

            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                train_size=self.train_ratio,
                stratify=y,
                random_state=42,
            )

            train_data = x_train
            train_data[self.target] = y_train
            test_data = x_test
            test_data[self.target] = y_test
            train_data.to_parquet(TRAIN_FILE_PATH)  # type:ignore
            test_data.to_parquet(TEST_FILE_PATH)  # type:ignore

            logger.info("Train data saved to %s", TRAIN_FILE_PATH)
            logger.info("Test data saved to %s", TEST_FILE_PATH)

        except Exception as e:
            logger.error("Error while splitting the data")
            raise CustomException("Failed to split the data") from e

    def run(self):
        try:
            logger.info("Starting data ingestion process")

            self.download_csv_from_gcp()
            self.split_data()

            logger.info("Data ingestion completed successfully")

        except CustomException as ce:
            logger.error("CustomException: %s", str(ce))

        finally:
            logger.info("Data ingestion completed")


if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
