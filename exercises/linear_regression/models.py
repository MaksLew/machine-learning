import mlflow
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from data import load_frame, split_data  # ty: ignore
from mlproject.tracking import setup

MEDINC_PATH = "data/medinc.joblib"
NO_LOCATION_PATH = "data/no-location.joblib"
MEDINC_LOCATION_PATH = "data/medinc-location.joblib"
ALL_PATH = "data/all.joblib"

DATA_LOADING_FILE = "exercises/linear_regression/data.py"

setup("w01-models")


def run(
    run_name: str,
    model_name: str,
    model_path: str,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
) -> None:
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_artifact(DATA_LOADING_FILE, artifact_path="code")

        train_ds = mlflow.data.from_pandas(  # ty: ignore
            train_data, name="train", targets="MedHouseVal"
        )
        mlflow.log_input(train_ds, context="training")

        train_features = train_data.drop(columns="MedHouseVal")
        model = LinearRegression().fit(train_features, train_data["MedHouseVal"])

        with open(model_path, "wb") as f:
            dump(model, f, protocol=5)

        mlflow.log_artifact(model_path, artifact_path="model-file")
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=train_features.head(5),
            registered_model_name=model_name,
        )

        val_ds = mlflow.data.from_pandas(  # ty: ignore
            val_data, name="val", targets="MedHouseVal"
        )
        mlflow.log_input(val_ds, context="validation")

        preds = model.predict(val_data.drop("MedHouseVal", axis="columns"))
        mse = mean_squared_error(val_data["MedHouseVal"], preds)

        mlflow.log_metric("mse", mse)


frame: pd.DataFrame = load_frame()
train_data, val_data, test_data = split_data(frame)

with mlflow.start_run(run_name="baseline-mean"):
    mlflow.log_artifact(DATA_LOADING_FILE, artifact_path="code")

    val_ds = mlflow.data.from_pandas(  # ty: ignore
        val_data, name="val", targets="MedHouseVal"
    )
    mlflow.log_input(val_ds, context="validation")

    mean = val_data["MedHouseVal"].mean()
    mse = mean_squared_error(
        val_data["MedHouseVal"], pd.Series(mean * np.ones(len(val_data)))
    )

    mlflow.log_metric("mse", mse)

run(
    "medinc",
    "medinc",
    MEDINC_PATH,
    train_data[["MedInc", "MedHouseVal"]],
    val_data[["MedInc", "MedHouseVal"]],
)

run(
    "no_location",
    "no_location",
    NO_LOCATION_PATH,
    train_data.drop(["Longitude", "Latitude"], axis="columns"),
    val_data.drop(["Longitude", "Latitude"], axis="columns"),
)

run(
    "medinc-location",
    "medinc-location",
    MEDINC_LOCATION_PATH,
    train_data[["MedInc", "Longitude", "Latitude", "MedHouseVal"]],
    val_data[["MedInc", "Longitude", "Latitude", "MedHouseVal"]],
)

run(
    "all",
    "all",
    ALL_PATH,
    train_data,
    val_data,
)
