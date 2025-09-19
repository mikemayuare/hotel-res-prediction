import pandas as pd
from pathlib import Path

import yaml

from src.custom_exceptions import CustomException
from src.logger import get_logger

logger = get_logger(__name__)


def read_yaml(file_path: Path) -> None:
    """Reads a YAML file

    Args:
        file_path: Path, path of the YAML file.

    Returns:
        None

    Raises:
        FileNotFoundError: in case file_path is not found
        CustomException: In case YAML is not readable
    """
    try:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"file {file_path} not found")

        with Path(file_path).open("r") as yml_file:
            config = yaml.safe_load(yml_file)
            logger.info("YAML file successfully read")
            return config

    except Exception as e:
        logger.error("%s - Error while reading YAML file", str(e))
        raise CustomException("Failed to read YAML file") from e


def load_data(file_path: Path) -> pd.DataFrame:
    """Transforms CSV into a pandas DataFrame

    Args:
        file_path: Path, path of the CSV file

    Returns:
        pd.DataFrame: DataFrame with the CSV file content

    Raises:
        CustomException: In case pandas fails to read the file.
    """
    try:
        logger.info("Loading data")
        return pd.read_parquet(file_path)
    except Exception as e:
        logger.error("%s - Error loading the data", str(e))
        raise CustomException("Failed to load data") from e
