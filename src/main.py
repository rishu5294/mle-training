# main.py
import argparse
import os
import subprocess

import mlflow

if __name__ == "__main__":
    PROJECT_ROOT = "/home/rishu_singh/Desktop/Assignment_2.1/mle-training/"
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        help="Model name ('linear_regression'/'decision_tree_regressor'/ \
        'random_forest-randomized_search'/'random_forest-grid_search')",
    )
    args = parser.parse_args()

    remote_server_uri = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(remote_server_uri)
    experiment_name = "House_price_prediction"
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        run_id = mlflow.active_run().info.run_id
        print(run_id)

        # Call ingest_data.py with specific arguments
        subprocess.run(["python", "src/ingest_data.py", experiment_name, run_id])

        # Call train.py with specific arguments
        subprocess.run(
            [
                "python",
                "src/train.py",
                args.model_name,
                experiment_name,
                run_id,
            ]
        )

        # Call score.py with specific arguments
        subprocess.run(
            [
                "python",
                "src/score.py",
                args.model_name,
                experiment_name,
                run_id,
            ]
        )
