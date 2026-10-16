import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from data import load_frame, split_data # ty: ignore
from mlproject.tracking import setup


DATA_LOADING_FILE = "exercises/linear_regression/data.py"


setup("w01-correlation")

with mlflow.start_run(run_name="correlation-selection") as run:
    mlflow.log_artifact(DATA_LOADING_FILE)

    train, val, test = split_data(load_frame())

    train_ds = mlflow.data.from_pandas( # ty: ignore
       train, name="train", targets="MedHouseVal"
    )
    mlflow.log_input(train_ds, context="calculate feature correlation")

    corr_matrix: pd.DataFrame = train.corr()

    mlflow.log_dict(corr_matrix.to_dict(), "correlation_with_target.json")

    fig, ax = plt.subplots()
    im = ax.imshow(corr_matrix)
    fig.tight_layout()

    mlflow.log_figure(fig, "correlation_train.png")

    ts = [0, 0.05, 0.10, 0.15, 0.30]

    for t in ts:
        # MedHouseVal always in
        features = corr_matrix[abs(corr_matrix["MedHouseVal"]) >= t].index

        if features.any():
            with mlflow.start_run(run_name=f"filtered training | {t=}", nested=True) as run:
                filtered_train = train[features]
                filtered_train_ds = mlflow.data.from_pandas( # ty: ignore
                   filtered_train, name="filtered_train", targets="MedHouseVal"
                )

                x = filtered_train.drop(columns="MedHouseVal")
                y = filtered_train["MedHouseVal"]

                model = LinearRegression().fit(x, y)

                preds = model.predict(val[features].drop(columns="MedHouseVal"))
                mse = mean_squared_error(val["MedHouseVal"], preds)

                mlflow.log_metric("val_mse", mse)
