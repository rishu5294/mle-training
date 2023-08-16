import argparse
import logging
import os

import joblib
import mlflow
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = "/home/rishu_singh/Desktop/Assignment_2.1/mle-training/"


def train_model(model_name, dataset_path, model_folder):
    # Load the dataset from the provided folder
    os.chdir(PROJECT_ROOT)
    train_df = pd.read_csv(dataset_path)
    # test_dataset = f"{dataset_folder}/housing_test.csv"
    # test_df = pd.read_csv(test_dataset)

    # Split the dataset into features and labels
    X_train = train_df.drop("median_house_value", axis=1)
    y_train = train_df["median_house_value"]
    # X_val = test_df.drop("median_house_value", axis=1)
    # y_val = test_df["median_house_value"]

    # Select the model based on the provided model name
    if model_name == "linear_regression":
        model = LinearRegression()
    elif model_name == "decision_tree_regressor":
        model = DecisionTreeRegressor()
    elif model_name == "random_forest-randomized_search":
        param_distribs = {
            "n_estimators": randint(low=1, high=200),
            "max_features": randint(low=1, high=8),
        }

        forest_reg = RandomForestRegressor(random_state=42)
        model = RandomizedSearchCV(
            forest_reg,
            param_distributions=param_distribs,
            n_iter=10,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
    elif model_name == "random_forest-grid_search":
        param_grid = [
            # try 12 (3×4) combinations of hyperparameters
            {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
            # then try 6 (2×3) combinations with bootstrap set as False
            {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
        ]

        forest_reg = RandomForestRegressor(random_state=42)
        # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
        model = GridSearchCV(
            forest_reg,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            return_train_score=True,
        )
    else:
        logger.error(f"Invalid model name: {model_name}")
        raise ValueError(f"Invalid model name: {model_name}")

    # Train the selected model
    model.fit(X_train, y_train)

    # Save the trained model to the output folder
    os.chdir(PROJECT_ROOT)
    os.makedirs(model_folder, exist_ok=True)
    model_path = f"{model_folder}/{model_name}.pkl"
    logger.info(f"Created {model_name}.pkl")
    joblib.dump(model, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        help="Model name ('linear_regression'/'decision_tree_regressor'/ \
        'random_forest-randomized_search'/'random_forest-grid_search')",
    )
    parser.add_argument(
        "experiment_name",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "run_id",
        help="MLflow run id",
    )
    parser.add_argument(
        "--dataset_path",
        metavar="<path>",
        default="data/processed/housing/housing_train.csv",
        required=False,
        help="Input dataset folder path",
    )
    parser.add_argument(
        "--model_folder",
        metavar="<path>",
        default="artifacts/models",
        required=False,
        help="Output model folder path",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Specify the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--log-path",
        metavar="<path>",
        default="logs/log.txt",
        required=False,
        help="Specify the log file path",
    )
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        required=False,
        help="Toggle whether or not to write logs to the console",
    )
    args = parser.parse_args()

    log_level = args.log_level.upper()
    log_file = args.log_path

    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if log_file:
        # Assume logs directory is in the project root
        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler("logs/log.txt", mode="a")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    if not args.no_console_log:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_id=args.run_id):
        with mlflow.start_run(run_name="Train Model", nested=True):
            mlflow.sklearn.autolog()
            train_model(args.model_name, args.dataset_path, args.model_folder)
