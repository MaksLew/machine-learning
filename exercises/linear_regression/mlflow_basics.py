import mlflow
from mlproject.tracking import setup
from data import load_frame, split_data

setup("w01-mlflow-basics")

with mlflow.start_run(run_name="basics") as run:
    mlflow.log_artifact("exercises/linear_regression/data.py", artifact_path="code")

    data = load_frame()
    train_data, val_data, test_data = split_data(data)

    mlflow.log_dict(
        {"train": len(train_data), "val": len(val_data), "test": len(test_data)},
        "split.json",
    )

    train_dataset = mlflow.data.from_pandas(
        train_data, name="california-train", targets="MedHouseVal"
    )
    mlflow.log_input(train_dataset, context="training")
