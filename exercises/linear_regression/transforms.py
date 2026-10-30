import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from data import load_frame, split_data  # ty: ignore
from mlproject.tracking import setup

TARGET = "MedHouseVal"
FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
LOG_FEATURES = ["MedInc", "AveRooms", "AveBedrms", "Population", "AveOccup"]
CLIP_FEATURES = ["AveRooms", "AveBedrms", "Population", "AveOccup"]
VARIANTS = ["none", "log1p", "clip-q99", "poly2", "poly3"]


def prepare(
    train_x: pd.DataFrame, other_x: pd.DataFrame, variant: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_t, other_t = train_x.copy(), other_x.copy()
    if variant == "log1p":
        train_t[LOG_FEATURES] = np.log1p(train_t[LOG_FEATURES])
        other_t[LOG_FEATURES] = np.log1p(other_t[LOG_FEATURES])
    elif variant == "clip-q99":
        limits = train_t[CLIP_FEATURES].quantile(0.99)
        train_t[CLIP_FEATURES] = train_t[CLIP_FEATURES].clip(upper=limits, axis=1)
        other_t[CLIP_FEATURES] = other_t[CLIP_FEATURES].clip(upper=limits, axis=1)
    return train_t, other_t


def transform(
    train_x: pd.DataFrame, other_x: pd.DataFrame, variant: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_t, other_t = prepare(train_x, other_x, variant)
    scaler = StandardScaler().fit(train_t)
    train_t = pd.DataFrame(scaler.transform(train_t), columns=FEATURES, index=train_t.index)
    other_t = pd.DataFrame(scaler.transform(other_t), columns=FEATURES, index=other_t.index)

    if variant in ("poly2", "poly3"):
        polynomial = PolynomialFeatures(degree=int(variant[-1]), include_bias=False)
        train_values = polynomial.fit_transform(train_t)
        other_values = polynomial.transform(other_t)
        columns = polynomial.get_feature_names_out(FEATURES)
        train_t = pd.DataFrame(train_values, columns=columns, index=train_t.index)
        other_t = pd.DataFrame(other_values, columns=columns, index=other_t.index)
    return train_t, other_t


setup("w01-transforms")
train, val, test = split_data(load_frame())

for variant in VARIANTS:
    train_t, val_t = transform(train[FEATURES], val[FEATURES], variant)
    model = LinearRegression().fit(train_t, train[TARGET])
    with mlflow.start_run(run_name=variant, tags={"stage": "selection"}):
        mlflow.log_param("variant", variant)
        mlflow.log_metrics(
            {
                "train_mse": mean_squared_error(train[TARGET], model.predict(train_t)),
                "val_mse": mean_squared_error(val[TARGET], model.predict(val_t)),
            }
        )

runs = mlflow.search_runs(
    experiment_names=["w01-transforms"],
    filter_string="tags.stage = 'selection'",
    order_by=["metrics.val_mse ASC"],
    max_results=1,
)
variant = runs.iloc[0]["params.variant"]
train_t, test_t = prepare(train[FEATURES], test[FEATURES], variant)
steps = [StandardScaler()]
if variant in ("poly2", "poly3"):
    steps.append(PolynomialFeatures(degree=int(variant[-1]), include_bias=False))
model = make_pipeline(*steps, LinearRegression()).fit(train_t, train[TARGET])

with mlflow.start_run(
    run_name=f"final-test-{variant}",
    tags={"stage": "final", "test_evaluated": "true"},
):
    mlflow.log_param("variant", variant)
    info = mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=train_t.head(5),
        registered_model_name="california-linear",
    )
    MlflowClient().set_registered_model_alias(
        "california-linear", "champion", info.registered_model_version
    )

    test_dataset = mlflow.data.from_pandas(test, name="test", targets=TARGET)  # ty: ignore
    mlflow.log_input(test_dataset, context="testing")
    registered_model = mlflow.pyfunc.load_model("models:/california-linear@champion")
    mlflow.log_metric(
        "test_mse", mean_squared_error(test[TARGET], registered_model.predict(test_t))
    )
