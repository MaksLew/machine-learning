import mlflow
from sklearn.linear_model import LinearRegression

from mlproject.tracking import setup
from problems import load_problem, fit_and_log, standardized
from weights import weight_summary, top_weights

data = load_problem("noise")

setup("w03-noise")
with mlflow.start_run(run_name="ols") as run:
    model, w = fit_and_log(standardized(LinearRegression()), data)
    mlflow.log_metrics(weight_summary(w))
    mlflow.log_dict(top_weights(w), "top_weights.json")

    train = data.X_train.join(data.y_train)
    mlflow.log_dict(train.corr()["target"].to_dict(), "correlation_with_target.json")
