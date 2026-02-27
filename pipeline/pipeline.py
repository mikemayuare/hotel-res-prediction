from config.paths_config import (
    CONFIG_PATH,
    MODEL_FILE_NAME,
    MODEL_OUPUT_PATH,
    PROCESSED_DIR,
    PROCESSED_TEST_DATA_PATH,
    PROCESSED_TRAIN_DATA_PATH,
    TEST_FILE_PATH,
    TRAIN_FILE_PATH,
)
from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcesser
from src.model_training import ModelTraining
from utils.common_functions import read_yaml

if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_preprocessing = DataProcesser(
        TRAIN_FILE_PATH,
        TEST_FILE_PATH,
        PROCESSED_DIR,
        CONFIG_PATH,
    )
    model_training = ModelTraining(
        PROCESSED_TRAIN_DATA_PATH,
        PROCESSED_TEST_DATA_PATH,
        MODEL_OUPUT_PATH,
        MODEL_FILE_NAME,
        CONFIG_PATH,
    )

    data_ingestion.run()
    data_preprocessing.process()
    model_training.run_training_pipeline()
