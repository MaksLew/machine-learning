import mlflow
from sklearn.linear_model import Ridge, Lasso

from mlproject.tracking import setup
from problems import search_alpha, load_problem, fit_and_log, standardized, RIDGE_ALPHAS, LASSO_ALPHAS, plot_search, plot_path
from weights import weight_summary, top_weights, choose_alpha

data = load_problem("noise")

setup("w03-noise")
with mlflow.start_run(run_name="ridge") as run:
    search, path = search_alpha(lambda a: Ridge(alpha=a), RIDGE_ALPHAS, data)
    alpha = choose_alpha(search)

    mlflow.log_metric("alpha", alpha)
    mlflow.log_table(search.to_dict(), "alpha_search.json")
    mlflow.log_figure(plot_search(search, alpha), "search.jpg")
    mlflow.log_figure(plot_path(search, alpha), "path.jpg")

    model, w = fit_and_log(standardized(Ridge(alpha=alpha)), data)
    mlflow.log_metrics(weight_summary(w))
    mlflow.log_dict(top_weights(w), "top_weights.json")

    train = data.X_train.join(data.y_train)
    mlflow.log_dict(train.corr()["target"].to_dict(), "correlation_with_target.json")

with mlflow.start_run(run_name="lasso") as run:
    search, path = search_alpha(lambda a: Lasso(alpha=a), LASSO_ALPHAS, data)
    alpha = choose_alpha(search)

    mlflow.log_metric("alpha", alpha)
    mlflow.log_table(search.to_dict(), "alpha_search.json")
    mlflow.log_figure(plot_search(search, alpha), "search.jpg")
    mlflow.log_figure(plot_path(search, alpha), "path.jpg")

    model, w = fit_and_log(standardized(Lasso(alpha=alpha)), data)
    mlflow.log_metrics(weight_summary(w))
    mlflow.log_dict(top_weights(w), "top_weights.json")

    train = data.X_train.join(data.y_train)
    mlflow.log_dict(train.corr()["target"].to_dict(), "correlation_with_target.json")
