import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from data import add_region, load_frame, one_hot, split_data  # ty: ignore
from mlproject.tracking import setup

TARGET = "MedHouseVal"
NUMERIC_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def prepare(
    train: pd.DataFrame,
    val: pd.DataFrame,
    numeric: list[str],
    use_region: bool,
    drop_first: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_x, val_x = train[numeric].copy(), val[numeric].copy()
    encoded = []
    if use_region:
        train_x["region"], val_x["region"] = train["region"], val["region"]
        train_x, val_x, encoded = one_hot(train_x, val_x, "region", drop_first)
    if numeric:
        scaler = StandardScaler().fit(train_x[numeric])
        train_x[numeric] = scaler.transform(train_x[numeric])
        val_x[numeric] = scaler.transform(val_x[numeric])
    return train_x, val_x, encoded


setup("w01-categorical")
train, val, _ = split_data(add_region(load_frame()))

with mlflow.start_run(run_name="region-onehot"):
    counts = {name: int(count) for name, count in train["region"].value_counts().items()}
    mlflow.log_dict(counts, "region-counts.json")

    ranks = {}
    for label, drop_first in (("full", False), ("drop_first", True)):
        encoded, _, columns = one_hot(train[["region"]], val[["region"]], "region", drop_first)
        matrix = np.column_stack((np.ones(len(encoded)), encoded[columns].to_numpy()))
        ranks[f"rank_{label}"] = np.linalg.matrix_rank(matrix)
    mlflow.log_metrics(ranks)

    variants = {
        "all": (NUMERIC_FEATURES, False, True, "none"),
        "region-only": ([], True, True, "drop-first"),
        "no-location+region": (NUMERIC_FEATURES[:-2], True, True, "drop-first"),
        "all+region": (NUMERIC_FEATURES, True, True, "drop-first"),
        "all+region-full": (NUMERIC_FEATURES, True, False, "full"),
    }
    for name, (numeric, use_region, drop_first, encoding) in variants.items():
        train_x, val_x, encoded = prepare(train, val, numeric, use_region, drop_first)
        with mlflow.start_run(run_name=name, nested=True):
            mlflow.log_params(
                {
                    "features": ",".join([*numeric, *encoded]),
                    "n_features": train_x.shape[1],
                    "encoding": encoding,
                }
            )
            model = LinearRegression().fit(train_x, train[TARGET])
            mlflow.log_metric("val_mse", mean_squared_error(val[TARGET], model.predict(val_x)))
