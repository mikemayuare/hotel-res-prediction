import os
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack

import config.paths_config
from src.custom_exceptions import CustomException
from src.logger import get_logger
from utils.common_functions import load_data, read_yaml

logger = get_logger(__name__)

class DataProcesser:
    def __init__(self, train_path: Path, test_path: Path, processed_dir: Path, config_path: Path) -> None:
        """

        Args:
            train_path: Training data file path
            test_path: Test data file path
            processed_dir: Directory to store processed files
            config_path: Path of the config YAML file
        """
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)

        if not Path(self.processed_dir).exists():
            Path(self.processed_dir).mkdir(parents=True, exist_ok=True)

    def preprocess_data(self, df: pd.DataFrame):
        try:
            logger.info("Starting data processing")
            logger.info("Dropping ID column")

            df = df.drop(columns="Booking_ID")
            df = df.drop_duplicates()

            cat_cols = self.config["data_preprocessing"]["categorical_columns"]
            num_cols = self.config["data_preprocessing"]["numerical_columns"]

            logger.info("Applying preprocesing")
            ohe = OneHotEncoder(drop="first")
            x_cat = ohe.fit_transform(df[cat_cols])
            logger.info("Categorical column preprocessed")

            logger.info("Preprocessing numerical columns")
            ss = StandardScaler()
            x_num = ss.fit_transform(df[num_cols])
            logger.info("Numerical columns preprocessed")

            processed_data = hstack(x_cat, x_num)
