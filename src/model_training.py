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


logger = get_logger(__name__)


class ModelTraining:
    def __init__(
        self,
        train_path: Path,
        test_path: Path,
        model_output_path: Path,
        config_path: Path,
    ) -> None:
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        self.config = read_yaml(config_path)
        self.target: str = self.config["data+prcessing"]["target"]
        self.accuracy: float
        self.precision: float
        self.recall: float
        self.f1: float

    def load_and_split(self):
        try:
            logger.info("loading training data from %s", self.train_path)
            train_df: pd.DataFrame = load_data(self.train_path)

            logger.info("loading training data from %s", self.train_path)
            test_df: pd.DataFrame = load_data(self.test_path)

            x_train: pd.DataFrame = train_df.drop(columns=[self.target])
            y_train = train_df[self.target]

            x_test: pd.DataFrame = test_df.drop(columns=[self.target])
            y_test = test_df[self.target]

            logger.info("Splitting data")
            x_train, x_val, y_train, y_val = train_test_split(
                x_train,
                y_train,
                test_size=0.2,
                random_state=42,
            )

            logger.info("Data loaded and splitted successfully")

            return (
                x_train,
                x_val,
                x_test,
                y_train,
                y_val,
                y_test,
            )

        except Exception as e:
            logger.error("%s - Error while loading data", e)
            raise CustomException("Error while loading data") from e

    def objective(self, trial, x_train, y_train, x_val, y_val):
        try:
            # build lightgbm datasets objects
            dtrain = lgb.Dataset(x_train, label=y_train)
            dvalid = lgb.Dataset(x_val, label=y_val)

            # load parameters
            params = lgbm_search_space(trial)

            # train
            gbm = lgb.train(
                params,
                dtrain,
                num_boost_round=100,
                valid_sets=[dvalid],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30),
                    LightGBMPruningCallback(trial, "binary_logloss"),
                ],
            )

            # predict and evaluate
            preds = gbm.predict(x_val)
            pred_labels = np.rint(preds)

            return accuracy_score(y_val, pred_labels)

        except Exception as e:
            logger.error("%s - Error while tuning", e)
            raise CustomException("Error while tunning") from e

    def train_model(self, x_train, y_train, x_val, y_val):
        try:
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=50, n_jobs=-1)

            params = study.best_params

            dtrain = lgb.Dataset(x_train, label=y_train)
            dvalid = lgb.Dataset(x_val, label=y_val)

            self.model = lgb.train(
                params,
                dtrain,
                valid_sets=[dvalid],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30),
                ],
            )

            return self.model

        except Exception as e:
            logger.error("%s - error while training the model")
            raise CustomException("Error while training the model") from e

    def model_evaluation(self, model, x_test, y_test):
        try:
            logger.info("Evaluating model")

            y_pred = model.predict(x_test, y_test)

            self.accuracy = accuracy_score(y_test, y_pred)
            self.precision = precision_score(y_test, y_pred)
            self.recall = recall_score(y_test, y_pred)
            self.f1 = f1_score(y_test, y_pred)

            logger.info("Accuracy %", self.accuracy)
            logger.info("Precision %", self.precision)
            logger.info("Recall %", self.recall)
            logger.info("F1 %", self.f1)

        except Exception as e:
            logger.error("%s, Error evaluating model", e)
            raise CustomException("Error evaluating model") from e

    def save_model(self, model):
        try:
            logger.info("Saving model to %s", self.model_output_path)
            self.model_output_path.mkdir(exist_ok=True)

            joblib.dump(model, self.model_output_path)

        except Exception as e:
            logger.error("%s, Error saving model", e)
            raise CustomException("Error saving model") from e
