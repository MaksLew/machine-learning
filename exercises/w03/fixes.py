import mlflow
from sklearn.linear_model import LinearRegression
import pandas as pd

from mlproject.tracking import setup
from problems import load_problem, fit_and_log, standardized
from weights import weight_summary, top_weights

data = load_problem("noise")

setup("w03-noise")
with mlflow.start_run(run_name="fix-filter-features") as run:
    train: pd.DataFrame = data.X_train.join(data.y_train)
    corr = train.corr(numeric_only=True)
    adjusted_features = corr.index[abs(corr["target"]) >= 0.1].drop("target").tolist()
    model, w = fit_and_log(standardized(LinearRegression()), data=data, columns=adjusted_features)

    mlflow.log_metrics(weight_summary(w))
    mlflow.log_dict(top_weights(w), "top_weights.json")

    mlflow.log_dict(train.corr()["target"].to_dict(), "correlation_with_target.json")
