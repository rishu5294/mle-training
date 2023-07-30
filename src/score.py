import argparse
import logging
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = "/home/rishu_singh/Desktop/Assignment_2.1/mle-training/"


def score_model(model_name, model_folder, dataset_folder, output_path):
    # Load the trained model from the provided folder
    model_path = f"{model_folder}/{model_name}.pkl"
    model = joblib.load(model_path)

    # Load the dataset from the provided folder
    dataset_path = f"{dataset_folder}/housing/housing_val.csv"
    df = pd.read_csv(dataset_path)

    # Perform scoring/prediction using the loaded model
    y_true = df.iloc[:, -1]
    predictions = model.predict(df.iloc[:, :-1])
    mse = mean_squared_error(df.iloc[:, -1], predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(df.iloc[:, -1], predictions)
    mape = np.sum(np.abs((y_true - predictions) / y_true)) / len(df) * 100
    prediction_score_dir = f"{output_path}/model_prediction_scores"
    os.makedirs(prediction_score_dir, exist_ok=True)
    prediction_score_file = f"{prediction_score_dir}/{model_name}-prediction_score.txt"

    if model_name == "linear_regression":
        with open(prediction_score_file, "a") as file:
            file.write(f"Linear Reg-RMSE: {rmse}\n")
            file.write(f"Linear Reg-MAE: {mae}\n")
            file.write(f"Linear Reg-MAPE: {mape}\n")

    elif model_name == "decision_tree_regressor":
        with open(prediction_score_file, "a") as file:
            file.write(f"Tree RMSE: {rmse}\n")
            file.write(f"Tree MAE: {mae}\n")
            file.write(f"Tree MAPE: {mape}\n")

    elif model_name == "random_forest-randomized_search":
        model.best_params_
        cvres = model.cv_results_
        with open(prediction_score_file, "a") as file:
            for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
                file.write(f"{np.sqrt(-mean_score)}, {params}\n")

        with open(prediction_score_file, "a") as file:
            file.write(f"Random Forest - Randomized Search RMSE: {rmse}\n")
            file.write(f"Random Forest - Randomized Search MAE: {mae}\n")
            file.write(f"Random Forest - Randomized Search MAPE: {mape}\n")

    elif model_name == "random_forest-grid_search":
        model.best_params_
        cvres = model.cv_results_
        with open(prediction_score_file, "a") as file:
            for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
                file.write(f"{np.sqrt(-mean_score)}, {params}\n")

        feature_importances = model.best_estimator_.feature_importances_
        with open(prediction_score_file, "a") as file:
            for impt, feature in zip(feature_importances, df.iloc[:, :-1].columns):
                file.write(f"{impt}, {feature}\n")

        model_ = model.best_estimator_
        predictions = model_.predict(df.iloc[:, :-1])
        mse = mean_squared_error(df.iloc[:, -1], predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(df.iloc[:, -1], predictions)
        mape = np.sum(np.abs((y_true - predictions) / y_true)) / len(df) * 100

        with open(prediction_score_file, "a") as file:
            file.write(f"Random Forest - Grid Search CV RMSE: {rmse}\n")
            file.write(f"Random Forest - Grid Search CV MAE: {mae}\n")
            file.write(f"Random Forest - Grid Search CV MAPE: {mape}\n")

        mlflow.log_params(model.best_params_)

    else:
        logger.error(f"Invalid model name: {model_name}")
        raise ValueError(f"Invalid model name: {model_name}")

    # Save the results to the specified output path
    prediction_dir = f"{output_path}/model_predictions"
    os.makedirs(prediction_dir, exist_ok=True)
    prediction_file = f"{prediction_dir}/{model_name}_prediction.txt"
    with open(prediction_file, "w") as file:
        for prediction in predictions:
            file.write(f"{prediction}\n")
    logger.info(f"Created {model_name}_prediction.txt")
    logger.info(f"Created {model_name}-prediction_scores.txt")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
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
        "--model_folder",
        metavar="<path>",
        default="artifacts/models",
        required=False,
        help="Model folder path",
    )
    parser.add_argument(
        "--dataset_folder",
        metavar="<path>",
        default="data/processed",
        required=False,
        help="Dataset folder path",
    )
    parser.add_argument(
        "--output_path",
        metavar="<path>",
        default="artifacts",
        required=False,
        help="Output file path",
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
        with mlflow.start_run(run_name="Score Model", nested=True):
            mlflow.sklearn.autolog()
            score_model(
                args.model_name,
                args.model_folder,
                args.dataset_folder,
                args.output_path,
            )
