import mlflow

from mlproject.tracking import setup
from previous import california_w1

setup("w04-california-torch")
with mlflow.start_run(run_name="torch-sgd"):
