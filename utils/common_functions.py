from pathlib import Path
import os

import pandas
import yaml

from src.custom_exceptions import CustomException
from src.logger import get_logger


logger = get_logger(__name__)


def read_yaml(file_path):
    try:
        if not os.path.exist(file_path):
            raise FileNotFoundError(f"file {file_path} not found")

        with open(file_path, "r") as yml_file:
            config = yaml.safe_load(yml_file)
            logger.info("YAML file successfully read")
            return config

    except Exception as e:
        logger.error("Error while reading YAML file")
