from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import optuna
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exceptions import CustomException
from config import paths_config as pc
from config.model_params import lgbm_search_space
from utils.common_functions import read_yaml, load_data
from optuna.integration import LightGBMPruningCallback
from optuna.exceptions import TrialPruned


logger = get_logger(__name__)


class ModelTraining:
    def __init__(
        self,
        train_path: Path,
        test_path: Path,
        model_output_path: Path,
        model_filename: Path,
        config_path: Path,
    ) -> None:
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        self.model_filename = model_filename
        self.config = read_yaml(config_path)
        self.target: str = self.config["data_processing"]["target"]
        self.accuracy: float
        self.precision: float
        self.recall: float
        self.f1: float

    def load_and_split(self):
        try:
            logger.info("loading training data from %s", self.train_path)
            train_df: pd.DataFrame = load_data(self.train_path)

            logger.info("loading test data from %s", self.train_path)
            test_df: pd.DataFrame = load_data(self.test_path)

            self.x_train: pd.DataFrame = train_df.drop(columns=[self.target])
            self.y_train = train_df[self.target]

            self.x_test: pd.DataFrame = test_df.drop(columns=[self.target])
            self.y_test = test_df[self.target]

            logger.info("Splitting data")
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
                self.x_train,
                self.y_train,
                test_size=0.1,
                random_state=42,
                shuffle=True,
                stratify=self.y_train,
            )

            logger.info("Data loaded and splitted successfully")

        except Exception as e:
            logger.error("%s - Error while loading data", str(e))
            raise CustomException("Error while loading data") from e

    def objective(self, trial):
        try:
            # load parameters
            params = lgbm_search_space(trial)

            # train
            gbm = lgb.train(
                params,
                self.dtrain,
                num_boost_round=1000,
                valid_sets=[self.dvalid],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100),
                ],
            )

            # predict and evaluate
            preds = gbm.predict(self.x_val)
            pred_labels = np.rint(preds)

            acc = accuracy_score(self.y_val, pred_labels)

            return acc

        except Exception as e:
            logger.error("%s - Error while tuning", e)
            raise CustomException("Error while tunning") from e

    def train_model(self):
        try:
            self.dtrain = lgb.Dataset(self.x_train, label=self.y_train)
            self.dvalid = lgb.Dataset(self.x_val, label=self.y_val)
            study = optuna.create_study(direction="maximize")
            study.optimize(
                self.objective, n_trials=50, n_jobs=1, show_progress_bar=True
            )

            params = study.best_params

            self.model = lgb.train(
                params,
                self.dtrain,
                num_boost_round=1000,
                valid_sets=[self.dvalid],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100),
                ],
            )

            return self.model

        except Exception as e:
            logger.error("%s - error while training the model", str(e))
            raise CustomException("Error while training the model") from e

    def model_evaluation(self):
        try:
            logger.info("Evaluating model")

            y_pred = self.model.predict(self.x_test)
            y_pred = np.rint(y_pred)

            self.accuracy = accuracy_score(self.y_test, y_pred)
            self.precision = precision_score(self.y_test, y_pred)
            self.recall = recall_score(self.y_test, y_pred)
            self.f1 = f1_score(self.y_test, y_pred)

            logger.info("Accuracy %s", self.accuracy)
            logger.info("Precision %s", self.precision)
            logger.info("Recall %s", self.recall)
            logger.info("F1 %s", self.f1)

        except Exception as e:
            logger.error("%s, Error evaluating model", str(e))
            raise CustomException("Error evaluating model") from e

    def save_model(self):
        try:
            logger.info("Saving model to %s", self.model_output_path)
            self.model_output_path.mkdir(exist_ok=True, parents=True)

            joblib.dump(self.model, self.model_output_path / self.model_filename)

        except Exception as e:
            logger.error("%s, Error saving model", str(e))
            raise CustomException("Error saving model") from e

    def run_training_pipeline(self):
        try:
            logger.info("Running training pipeline")

            # LOAD DATA #
            self.load_and_split()

            # TRAIN DATA #
            self.train_model()

            # EVALUATE DATA #
            self.model_evaluation()

            # SAVE MODEL #
            self.save_model()

        except Exception as e:
            logger.error("%s - Error during pipeline", str(e))
            raise CustomException("Error running the pipeline, check logs") from e


if __name__ == "__main__":
    trainer = ModelTraining(
        pc.PROCESSED_TRAIN_DATA_PATH,
        pc.PROCESSED_TEST_DATA_PATH,
        pc.MODEL_OUPUT_PATH,
        pc.MODEL_FILE_NAME,
        pc.CONFIG_PATH,
    )
    trainer.run_training_pipeline()
