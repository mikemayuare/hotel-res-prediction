from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.stats import skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config.paths_config
from src.custom_exceptions import CustomException
from src.logger import get_logger
from utils.common_functions import load_data, read_yaml

logger = get_logger(__name__)


class DataProcesser:
    def __init__(
        self, train_path: Path, test_path: Path, processed_dir: Path, config_path: Path
    ) -> None:
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

            logger.info("Applying preprocessing")
            ohe = OneHotEncoder(drop="first", sparse_output=False)
            x_cat = ohe.fit_transform(df[cat_cols])
            logger.info("Categorical column preprocessed")

            logger.info("Preprocessing numerical columns")
            skew_threshold = self.config["data_preprocessing"]["skewness_threshold"]
            skewness = df[num_cols].skew()
            skewed_cols = skewness[skewness > skew_threshold].index
            df[skewed_cols] = np.log1p(df[skewed_cols])

            ss = StandardScaler()
            x_num = ss.fit_transform(df[num_cols])
            logger.info("Numerical columns preprocessed")

            cat_feature_names = ohe.get_feature_names_out(cat_cols)
            df = pd.DataFrame(
                np.hstack([x_num, x_cat]), columns=num_cols + list(cat_feature_names)
            )
            return df

        except Exception as e:
            logger.error("%s - Error during preprocesing", str(e))
            raise CustomException("Error while preprocessing") from e

    def balance_data(self, df):
        try:
            logger.info("Handling imbalanced data")
            x = df.drop(columns=df.columns[-1])
            y = df[df.columns[-1]]

            smote = SMOTE(random_state=42)
            x_resampled, y_resampled = smote.fit_resample(x, y)
            balanced_df = pd.DataFrame(x_resampled, columns=x.columns)
            balanced_df["booking_status"] = y_resampled

            logger.info("Data balanced successfully")
            return balanced_df

        except Exception as e:
            logger.error("%s - Error while balancing data", str(e))
            raise CustomException("Error while balancing data") from e
