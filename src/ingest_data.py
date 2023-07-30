import argparse
import logging
import os
import tarfile

import mlflow
import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

PROJECT_ROOT = "/home/rishu_singh/Desktop/Assignment_2.1/mle-training/"


def download_data(output_path):
    # Download data from a remote source
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    os.chdir(PROJECT_ROOT)
    os.makedirs(output_path, exist_ok=True)
    tgz_path = os.path.join(output_path, "housing.tgz")
    urllib.request.urlretrieve(HOUSING_URL, tgz_path)
    logger.info(f"Downloading data to: {output_path}")
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=output_path)
    housing_tgz.close()


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        metavar="<path>",
        default="data/raw/housing/",
        required=False,
        help="Output folder/filepath",
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
        with mlflow.start_run(run_name="Data Loading & Preprocessing", nested=True):
            download_data(args.output_path)
            housing = pd.read_csv(rf"{args.output_path}housing.csv")
            print(housing)

            housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
            housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
            housing["population_per_household"] = housing["population"] / housing["households"]

            housing["income_cat"] = pd.cut(
                housing["median_income"],
                bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                labels=[1, 2, 3, 4, 5],
            )

            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

            for train_index, test_index in split.split(housing, housing["income_cat"]):
                strat_train_set = housing.loc[train_index]
                strat_test_set = housing.loc[test_index]

            train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

            compare_props = pd.DataFrame(
                {
                    "Overall": income_cat_proportions(housing),
                    "Stratified": income_cat_proportions(strat_test_set),
                    "Random": income_cat_proportions(test_set),
                }
            ).sort_index()
            compare_props["Rand. %error"] = (
                100 * compare_props["Random"] / compare_props["Overall"] - 100
            )
            compare_props["Strat. %error"] = (
                100 * compare_props["Stratified"] / compare_props["Overall"] - 100
            )
            print("\n", compare_props)

            for set_ in (strat_train_set, strat_test_set):
                set_.drop("income_cat", axis=1, inplace=True)

            housing_train = strat_train_set.copy()
            housing_val = strat_test_set.copy()

            housing_train.plot(kind="scatter", x="longitude", y="latitude")
            housing_train.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

            housing_train_num = housing_train.select_dtypes(include=[np.number])
            corr_matrix = housing_train_num.corr()
            print("\n", corr_matrix["median_house_value"].sort_values(ascending=False))

            housing_train = strat_train_set.drop("median_house_value", axis=1)
            housing_val = strat_test_set.drop("median_house_value", axis=1)

            # drop labels for both sets
            housing_train_labels = strat_train_set["median_house_value"].copy()
            housing_val_labels = strat_test_set["median_house_value"].copy()

            # Data preprocessing
            print("Data preprocessing...")
            imputer = SimpleImputer(strategy="median")

            housing_train_num = housing_train.drop("ocean_proximity", axis=1)
            housing_val_num = housing_val.drop("ocean_proximity", axis=1)

            imputer.fit(housing_train_num)
            housing_train_nums = imputer.transform(housing_train_num)
            housing_val_nums = imputer.transform(housing_val_num)

            housing_train_num = pd.DataFrame(
                housing_train_nums,
                columns=housing_train_num.columns,
                index=housing_train.index,
            )

            housing_val_num = pd.DataFrame(
                housing_val_nums,
                columns=housing_val_num.columns,
                index=housing_val.index,
            )

            housing_train_cat = housing_train[["ocean_proximity"]]
            housing_val_cat = housing_val[["ocean_proximity"]]
            housing_train = housing_train_num.join(
                pd.get_dummies(housing_train_cat, drop_first=True)
            )
            housing_train = housing_train.join(housing_train_labels)

            housing_val = housing_val_num.join(pd.get_dummies(housing_val_cat, drop_first=True))
            housing_val = housing_val.join(housing_val_labels)

            os.makedirs("data/processed/housing", exist_ok=True)
            housing_train.to_csv("data/processed/housing/housing_train.csv")
            housing_val.to_csv("data/processed/housing/housing_val.csv")

            mlflow.log_artifact("data/processed/housing/housing_val.csv")
