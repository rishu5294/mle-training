import argparse
import configparser
import logging
import logging.config
import os

import mlflow
import numpy as np
import pandas as pd
import train
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

LOGGING_DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {"format": "%(message)s"},
    },
    "root": {"level": "INFO"},
}


def configure_logger(
    logger=None,
    cfg=None,
    log_file=None,
    console=False,
    log_level_var="INFO",
):
    """Function to configure the logger

    Parameters
    ----------
    logger
    cfg
    log_file
    console
    log_level_var

    Returns
    -------
    logger

    """

    if not cfg:
        logging.config.dictConfig(LOGGING_DEFAULT_CONFIG)
    else:
        logging.config.dictConfig(cfg)

    logger = logger or logging.getLogger()

    if log_file or console:
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)

        if log_file:
            file = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_file)
            fh = logging.FileHandler(file)
            fh.setLevel(getattr(logging, log_level_var))
            logger.addHandler(fh)
        if not console:
            sh = logging.StreamHandler()
            sh.setLevel(getattr(logging, log_level_var))
            logger.addHandler(sh)

    return logger


def score(experiment_id):
    """Function to score the modules.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", help="the path to access the housing data")
    parser.add_argument(
        "--config_file", help="Specify the path to the configuration file"
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Specify the log level",
    )
    parser.add_argument("--log_path", help="Specify the path to the log file")
    parser.add_argument(
        "--no_console_log", action="store_true", help="Disable console logging"
    )
    args = parser.parse_args()
    config = configparser.ConfigParser()

    if args.config_file:
        config.read(args.config_file)
    else:
        config.read("setup.cfg")
    log_level = config.get("Logging", "log_level", fallback="INFO")
    log_path = config.get("Logging", "log_path", fallback=None)
    console_log_not_enabled = config.getboolean(
        "Logging", "console_log_not_enabled", fallback=False
    )

    if args.log_level:
        log_level = args.log_level
    if args.log_path:
        log_path = args.log_path
    if args.no_console_log:
        console_log_not_enabled = True

    # configuring and assigning in the logger can be done by the below function
    logger = configure_logger(
        log_file=log_path, console=console_log_not_enabled, log_level_var=log_level
    )
    logger.warning(log_level)
    logger.info("Logging Test - Start")
    logger.info("Logging Test - Test 1 Done")
    logger.warning("Watch out!")

    if args.data_path:
        (
            housing_prepared,
            housing_labels,
            strat_test_set,
            imputer,
        ) = train.training_data(args.data_path)
    else:
        (
            housing_prepared,
            housing_labels,
            strat_test_set,
            imputer,
        ) = train.training_data()

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    lin_rmse

    lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    lin_mae

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)

    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    tree_rmse

    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing_prepared, housing_labels)
    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        logger.info(" %s, %s", np.sqrt(-mean_score), params)
        with mlflow.start_run(
            run_name="CHILD_RUN",
            experiment_id=experiment_id,
            nested=True,
        ):
            mlflow.log_metric("rmse", np.sqrt(-mean_score))
            mlflow.log_metric("max_features", params["max_features"])
            mlflow.log_metric("n_estimators", params["n_estimators"])

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)

    grid_search.best_params_
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        logger.info(" %s, %s", np.sqrt(-mean_score), params)

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

    final_model = grid_search.best_estimator_

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)

    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print(final_rmse)
    return
