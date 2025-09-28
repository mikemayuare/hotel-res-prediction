from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config.paths_config as paths
from src.custom_exceptions import CustomException
from src.logger import get_logger
from utils.common_functions import load_data, read_yaml

logger = get_logger(__name__)


class DataProcesser:
    def __init__(
        self, train_path: Path, test_path: Path, processed_dir: Path, config_path: Path
    ) -> None:
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """
        Preprocess dataset for LightGBM:
        - Drop ID + duplicates
        - Label encode target
        - Label encode categorical features
        - Log-transform skewed numeric features
        - Standard scale numeric features (optional)
        Returns:
            x: Feature DataFrame
            y: Target Series
            cat_cols: List of categorical feature names
        """
        try:
            logger.info("Starting data preprocessing")

            # Drop ID and duplicates
            df = df.drop(columns="Booking_ID")
            df = df.drop_duplicates()

            # --------------------
            # Target
            # --------------------
            self.target_col: str = self.config["data_processing"]["target"]
            y_raw = df[self.target_col]
            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            self.target_encoder = le

            # --------------------
            # Features
            # --------------------
            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            # Label encode categorical features
            self.cat_encoders = {}
            for col in cat_cols:
                le_col = LabelEncoder()
                df[col] = le_col.fit_transform(df[col].astype(str))
                self.cat_encoders[col] = le_col

            # Numerical preprocessing
            skew_threshold = self.config["data_processing"]["skewness_threshold"]
            skewness = df[num_cols].skew()
            skewed_cols = skewness[skewness > skew_threshold].index
            for col in skewed_cols:
                df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

            # Optional scaling
            ss = StandardScaler()
            df[num_cols] = ss.fit_transform(df[num_cols])

            # Features and target
            x = df.drop(columns=self.target_col)
            y = pd.Series(y, name=self.target_col)

            logger.info("Preprocessing completed")
            return x, y, cat_cols

        except Exception as e:
            logger.error("%s - Error during preprocessing", str(e))
            raise CustomException("Error while preprocessing") from e

    def balance_data(
        self, x: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to balance the dataset.
        """
        try:
            logger.info("Handling imbalanced data with SMOTE")
            smote = SMOTE(random_state=42)
            x_res, y_res = smote.fit_resample(x, y)
            logger.info("Data balanced successfully")
            return x_res, y_res
        except Exception as e:
            logger.error("%s - Error during balancing data", str(e))
            raise CustomException("Error while balancing data") from e

    def select_features(
        self, x: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Select top features using RandomForestClassifier feature importances.
        """
        try:
            logger.info("Starting feature selection step")
            model = RandomForestClassifier(random_state=42)
            model.fit(x, y)

            feature_importance = pd.DataFrame(
                {"feature": x.columns, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)

            num_features_to_select = self.config["data_processing"]["no_of_features"]
            top_features = feature_importance.head(num_features_to_select)[
                "feature"
            ].tolist()

            x_selected = x[top_features]
            logger.info("Selected features: %s", top_features)
            return x_selected, top_features
        except Exception as e:
            logger.error("%s - Error during feature selection", str(e))
            raise CustomException("Error while selecting features") from e

    def save_data(self, df: pd.DataFrame, file_path: Path) -> None:
        try:
            logger.info("Saving data to processed folder: %s", str(file_path))
            df.to_parquet(file_path)
            logger.info("Data saved successfully")
        except Exception as e:
            logger.error("%s - Error during saving file", str(e))
            raise CustomException("Error while saving processed file") from e

    def process(self):
        """
        Full pipeline:
        - Load train/test
        - Preprocess
        - Balance train
        - Feature selection
        - Save processed datasets
        """
        try:
            logger.info("Starting full data processing pipeline")

            # Load raw data
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            # Preprocess
            x_train, y_train, cat_cols = self.preprocess_data(train_df)
            x_test, y_test, _ = self.preprocess_data(test_df)

            # Balance training data
            x_train, y_train = self.balance_data(x_train, y_train)

            # Feature selection
            x_train, top_features = self.select_features(x_train, y_train)
            x_test = x_test[top_features]

            # Save processed data
            train_df = x_train.copy()
            train_df[self.target_col] = y_train
            test_df = x_test.copy()
            test_df[self.target_col] = y_test

            self.save_data(train_df, paths.PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, paths.PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing pipeline completed successfully")

        except Exception as e:
            logger.error("%s - Error during full pipeline", str(e))
            raise CustomException("Error while processing pipeline") from e


if __name__ == "__main__":
    processor = DataProcesser(
        paths.TRAIN_FILE_PATH,
        paths.TEST_FILE_PATH,
        paths.PROCESSED_DIR,
        paths.CONFIG_PATH,
    )
    processor.process()
