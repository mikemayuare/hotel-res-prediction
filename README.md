# Hotel Reservation Prediction

An end-to-end MLOps pipeline for predicting hotel reservation cancellations. This project leverages LightGBM for classification, Optuna for hyperparameter optimization, and MLflow for experiment tracking and model management.

## 🚀 Overview

The goal of this project is to predict the `booking_status` (Canceled vs. Not Canceled) of hotel reservations. This enables:
- **Revenue Management**: Better forecasting of cancellations.
- **Targeted Marketing**: Identifying customers likely to cancel and offering incentives.
- **Operational Efficiency**: Optimizing room allocation.

## 🛠️ Tech Stack

- **Language**: Python 3.12+
- **Dependency Management**: [uv](https://github.com/astral-sh/uv)
- **Model**: [LightGBM](https://lightgbm.readthedocs.io/)
- **Hyperparameter Tuning**: [Optuna](https://optuna.org/)
- **Experiment Tracking**: [MLflow](https://mlflow.org/)
- **Data Storage**: Google Cloud Storage (GCP)
- **Data Format**: Parquet (for intermediate storage)
- **Preprocessing**: Scikit-learn, Imbalanced-learn (SMOTE)

## 📁 Project Structure

```text
├── artifacts/              # Data and model artifacts
├── config/                 # Configuration files (YAML and Python)
├── logs/                   # Application logs
├── mlruns/                 # MLflow tracking data
├── pipeline/               # Pipeline script
    └── pipeline.py
├── src/                    # Source code for the pipeline
│   ├── data_ingestion.py   # GCP download and data splitting
│   ├── data_preprocessing.py # Feature engineering and balancing
│   ├── model_training.py    # Training with Optuna and MLflow
│   ├── logger.py           # Logging configuration
│   └── custom_exceptions.py # Custom error handling
├── utils/                  # Shared utility functions
├── pyproject.toml          # Project dependencies and metadata
└── README.md               # Project documentation
```

## ⚙️ Setup

### 1. Prerequisites
- [uv](https://github.com/astral-sh/uv) installed.
- GCP Service Account Key (JSON) with access to the configured bucket.

### 2. Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
```

### 3. Install Dependencies
```bash
uv sync
```

## 🛤️ Pipeline Execution

The pipeline is split into three main stages. You can run them sequentially or with the pipeline script:

### Sequential Execution

#### 1. Data Ingestion
Downloads the raw dataset from GCP and performs an initial train-test split.
```bash
uv run python -m src.data_ingestion
```

#### 2. Data Preprocessing
Handles categorical encoding, log transformation for skewed features, data balancing using **SMOTE**, and feature selection using a Random Forest importance ranking.
```bash
uv run python -m src.data_preprocessing
```

#### 3. Model Training
Trains a LightGBM model with:
- **Hyperparameter Tuning**: 100 trials using Optuna.
- **MLflow Tracking**: Automatically logs parameters, metrics (Accuracy, Precision, Recall, F1), and artifacts (datasets and the final model).
```bash
uv run python -m src.model_training
```

### Pipeline Execution

```bash
uv run pipeline.pipeline
```

## 📊 Experiment Tracking with MLflow

To visualize the training results, metrics, and compare different runs:

1. Start the MLflow UI:
   ```bash
   mlflow ui
   ```
2. Open `http://localhost:5000` in your browser.

## 📈 Dataset
The project uses the [Hotel Reservations Classification Dataset](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset) from Kaggle.

- **Target**: `booking_status`
- **Features**: Includes lead time, average price per room, special requests, and more.
