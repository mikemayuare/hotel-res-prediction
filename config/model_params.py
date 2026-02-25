# Defines the hyperparameter search space for LightGBM (low-level API).
def lgbm_search_space(trial):
    return {
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": [
            "binary_logloss",
            "auc",
        ],  # 👈 logloss for stability, auc for pruning
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256, step=8),  # avoid too low
        "max_depth": trial.suggest_int("max_depth", -1, 12),  # -1 = no limit
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1e-4, 10.0, log=True
        ),
        "feature_pre_filter": False,
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),  # bagging_fraction
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 15),  # bagging_freq
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.01, 1.0
        ),  # feature_fraction
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight", 0.5, 2.3
        ),  # useful if class imbalance
        "random_state": 42,
        "verbose_eval": False,
        "verbosity": -1,
        # "device": "gpu",
    }
