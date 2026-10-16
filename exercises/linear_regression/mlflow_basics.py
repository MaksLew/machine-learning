from pathlib import Path
import mlflow
import pandas as pd
from joblib import load, dump
from sklearn.linear_model import LinearRegression
from mlproject.tracking import setup
from data import load_frame, split_data # ty: ignore

MEDINC_MODEL_PATH = "data/medinc.joblib"

setup("w01-mlflow-basics")

with mlflow.start_run(run_name="basics") as run:
    mlflow.log_artifact("exercises/linear_regression/data.py", artifact_path="code")

    data = load_frame()
    train_data, val_data, test_data = split_data(data)

    mlflow.log_dict(
        {"train": len(train_data), "val": len(val_data), "test": len(test_data)},
        "split.json",
    )

    train_dataset = mlflow.data.from_pandas( # ty: ignore
        train_data, name="california-train", targets="MedHouseVal"
    )
    mlflow.log_input(train_dataset, context="training")

    model = LinearRegression().fit(pd.DataFrame(train_data["MedInc"]), pd.DataFrame(train_data["MedHouseVal"]))
    val_preds_before_loading = model.predict(pd.DataFrame(val_data["MedInc"]))

    with open(MEDINC_MODEL_PATH, "wb") as f:
        dump(model, f, protocol=5)

    with open(MEDINC_MODEL_PATH, "rb") as f:
        model = load(f)

    val_preds_after_loading = model.predict(pd.DataFrame(val_data["MedInc"]))
    print(val_preds_before_loading == val_preds_after_loading)

    mlflow.log_artifact(MEDINC_MODEL_PATH, artifact_path="model-file")

    info = mlflow.sklearn.log_model(model, name="model", input_example=pd.DataFrame(train_data["MedInc"]).head(5), registered_model_name="california-medinc")
