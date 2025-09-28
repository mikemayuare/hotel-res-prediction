
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
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50, n_jobs=-1)

        params = study.best_params

        dtrain = lgb.Dataset(x_train, label=y_train)
        dvalid = lgb.Dataset(x_val, label=y_val)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
            ],
        )

        model.save_model(pc.MODEL_OUPUT_PATH)
