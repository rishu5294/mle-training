Assignment 2 - Machine Learning Workflow
This repository contains the code for Assignment 1 of the Machine Learning Engineering course. The goal of this assignment is to refactor and reformat the provided code into a production-ready version and implement a complete machine learning workflow.

Installation
Clone the repository to your local machine:
git clone <repository_url>
cd assignment-2

conda env create -f env.yaml
conda activate assignment1


1. Data Ingestion
Run the ingest_data.py script to download and create the training and validation datasets:

python src/ingest_data.py --output_folder data/

2. Model Training
Use the train.py script to train the model(s):

python src/train.py --input_folder data/ --output_folder artifacts/models/


3. Model Scoring
To score the model(s) on the test dataset, run the score.py script:

python src/score.py --model_folder artifacts/models/ --dataset_folder data/ --output_folder artifacts/scores/


Testing
To run the unit tests and functional tests, use the following command:

python -m unittest discover tests


Navigate to the "docs" folder:

cd docs
Open the "index.html" file in your web browser.
Note: This is a sample README file for the assignment. Be sure to replace <repository_url> with the actual URL of your GitHub repository and make any necessary adjustments based on your project structure and specific requirements. Additionally, ensure that you have provided sufficient information in the README to guide users on installing, running, and testing your code.

