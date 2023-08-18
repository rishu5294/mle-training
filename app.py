import argparse
import os
import subprocess

import mlflow
from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Hello, World! This is your House pridiction application."


if __name__ == "__main__":
    # Start the Flask web server
    app.run(host="0.0.0.0", port=8080, debug=True)
