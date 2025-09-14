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
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)

        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)

    def preprocess_data(self, df: pd.DataFrame):
        """
        Preprocess the dataset:
        - Drop ID column and duplicates
        - One-hot encode categorical features (sparse)
        - Log-transform skewed numeric columns
        - Standard scale numeric columns
        Returns:
            x_num: scaled numeric features (dense)
            x_cat: encoded categorical features (sparse)
            feature_names: list of column names for later use
        """
        try:
            logger.info("Starting data preprocessing")
            df = df.drop(columns="Booking_ID")
            df = df.drop_duplicates()

            cat_cols = self.config["data_preprocessing"]["categorical_columns"]
            num_cols = self.config["data_preprocessing"]["numerical_columns"]

            # --------------------
            # Categorical preprocessing
            # --------------------
            ohe = OneHotEncoder(drop="first", sparse=True)  # keep sparse for memory
            x_cat = ohe.fit_transform(df[cat_cols])
            cat_feature_names = ohe.get_feature_names_out(cat_cols)
            logger.info("Categorical columns preprocessed (sparse)")

            # --------------------
            # Numerical preprocessing
            # --------------------
            skew_threshold = self.config["data_preprocessing"]["skewness_threshold"]
            skewness = df[num_cols].skew()
            skewed_cols = skewness[skewness > skew_threshold].index

            # Safe log transform for numeric skewed features
            for col in skewed_cols:
                df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

            ss = StandardScaler()
            x_num = ss.fit_transform(df[num_cols])
            logger.info("Numerical columns preprocessed")

            return x_num, x_cat, num_cols, cat_feature_names

        except Exception as e:
            logger.error("%s - Error during preprocessing", str(e))
            raise CustomException("Error while preprocessing") from e

    def balance_data(self, x_num, x_cat, y):
        """
        Apply SMOTE to balance the dataset.
        Converts sparse + dense features to dense for SMOTE.
        Returns:
            x_resampled: balanced features (dense)
            y_resampled: balanced target
        """
        try:
            logger.info("Handling imbalanced data with SMOTE")

            # Convert sparse + dense to a single dense matrix

            x_combined = hstack([x_cat, x_num]).toarray()

            smote = SMOTE(random_state=42)
            x_resampled, y_resampled = smote.fit_resample(x_combined, y)

            logger.info("Data balanced successfully")

            return x_resampled, y_resampled

        except Exception as e:
            logger.error("%s - Error during balancing data", str(e))
            raise CustomException("Error while balancing data") from e

    def select_features(self, x: pd.DataFrame, y: pd.Series):
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
                .values
            )

            logger.info("Features selected: %s", top_features)

            # Return DataFrame with selected features + target
            top_features_df = pd.DataFrame(x[top_features], columns=top_features)
            top_features_df["booking_status"] = y.values

            logger.info("Feature selection completed successfully")
            return top_features_df

        except Exception as e:
            logger.error("%s - Error during feature selection", str(e))
            raise CustomException("Error while selecting features") from e

    def save_data(self, df: pd.DataFrame, file_path: Path) -> None:
        try:
            logger.info("Saving dta in processed folder")
            df.to_parquet(file_path)
            logger.info("Data saved on %s", str(file_path))

        except Exception as e:
            logger.error("%s - Error during saving file", str(e))
            raise CustomException("Error while saving processed file") from e
