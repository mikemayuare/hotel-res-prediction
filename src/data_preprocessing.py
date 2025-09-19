from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

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

        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)

    def preprocess_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess dataset:
        - Drop ID + duplicates
        - Separate and encode target column
        - One-hot encode categorical features (excluding target)
        - Log-transform skewed numeric features
        - Standard scale numeric features
        """
        try:
            logger.info("Starting data preprocessing")

            # Drop ID and duplicates
            df = df.drop(columns="Booking_ID")
            df = df.drop_duplicates()

            # --------------------
            # Extract and encode target
            # --------------------
            target_col = "booking_status"
            y_raw = df[target_col]

            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            self.target_encoder = le  # save encoder for inverse_transform later

            # --------------------
            # Features
            # --------------------
            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            # Categorical preprocessing
            ohe = OneHotEncoder(drop="first", sparse_output=False)
            x_cat = ohe.fit_transform(df[cat_cols])
            cat_feature_names = ohe.get_feature_names_out(cat_cols)
            logger.info("Categorical columns preprocessed (sparse)")

            # Numerical preprocessing
            skew_threshold = self.config["data_processing"]["skewness_threshold"]
            skewness = df[num_cols].skew()
            skewed_cols = skewness[skewness > skew_threshold].index

            for col in skewed_cols:
                df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

            ss = StandardScaler()  # not needed on tree based but i will leave it
            x_num = ss.fit_transform(df[num_cols])
            logger.info("Numerical columns preprocessed")

            x = np.hstack([x_num, x_cat])
            x = pd.DataFrame(x, columns=num_cols + list(ohe.get_feature_names_out()))
            y = pd.Series(y, name=target_col)

            return x, y

        except Exception as e:
            logger.error("%s - Error during preprocessing", str(e))
            raise CustomException("Error while preprocessing") from e

    def balance_data(
        self, x: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to balance the dataset.
        Converts sparse + dense features to dense for SMOTE.

        Args:
            x_cat: sparse matrix (from OneHotEncoder with sparse=True)
            x_num: dense numpy array (from StandardScaler)
            y: array-like (pandas Series or ndarray)
        Returns:
            x_resampled: balanced features (dense)
            y_resampled: balanced target
        """
        try:
            logger.info("Handling imbalanced data with SMOTE")

            smote = SMOTE(random_state=42)
            x, y = smote.fit_resample(x, y)

            logger.info("Data balanced successfully")

            return x, y

        except Exception as e:
            logger.error("%s - Error during balancing data", str(e))
            raise CustomException("Error while balancing data") from e

    def select_features(
        self, x: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, list]:
        """
        Select top features using feature importance from a RandomForestClassifier.

        Args:
            x: Feature DataFrame
            y: Target Series

        Returns:
            top_features_df: DataFrame with selected features + target column
        """
        try:
            logger.info("Starting feature selection step")

            model = RandomForestClassifier(random_state=42)
            model.fit(x, y)

            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame(
                {"feature": x.columns, "importance": feature_importance}
            )

            top_features_importance_df = feature_importance_df.sort_values(
                by="importance", ascending=False
            )

            num_features_to_select = self.config["data_processing"]["no_of_features"]
            top_features = (
                top_features_importance_df["feature"]
                .head(num_features_to_select)
                .tolist()
            )

            logger.info("Features selected: %s", top_features)

            # Return DataFrame with selected features + target
            x = x[top_features]

            logger.info("Feature selection completed successfully")
            return x, top_features

        except Exception as e:
            logger.error("%s - Error during feature selection", str(e))
            raise CustomException("Error while selecting features") from e

    def save_data(self, df: pd.DataFrame, file_path: Path) -> None:
        """
        Save processing file to processed path

        Args:
            df: processed DataFrame
            file_path: processed directory
        """
        try:
            logger.info("Saving dta in processed folder")
            df.to_parquet(file_path)
            logger.info("Data saved on %s", str(file_path))

        except Exception as e:
            logger.error("%s - Error during saving file", str(e))
            raise CustomException("Error while saving processed file") from e

    def process(self):
        """
        Full pipeline:
        1. Load train/test data
        2. Preprocess
        3. Balance training data
        4. Feature selection
        5. Save processed datasets
        """
        try:
            logger.info("Starting full data processing pipeline")

            # --------------------
            # Load raw data
            # --------------------
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            # --------------------
            # Preprocess train + test
            # --------------------
            x_train, y_train = self.preprocess_data(train_df)
            x_test, y_test = self.preprocess_data(test_df)

            # --------------------
            # Balance train data
            # --------------------
            x_train, y_train = self.balance_data(x_train, y_train)

            # --------------------
            # Feature selection
            # --------------------
            x_train, top_features = self.select_features(x_train, y_train)
            x_test = x_test[top_features]

            train_df = x_train
            train_df["target"] = y_train
            test_df = x_test
            test_df["target"] = y_test
            # --------------------
            # Save outputs
            # --------------------
            self.save_data(train_df, paths.PROCESSED_TRAIN_DATA_PATH)

            self.save_data(test_df, paths.PROCESSED_TEST_DATA_PATH)

            logger.info("Full data processing pipeline completed successfully")

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
