import time
from ast import literal_eval
import mlflow
import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from mlproject.models.mlp import MLP
from mlproject.tracking import setup
from mlproject.training.trainer import evaluate, train_one_epoch


def get_data(seed: int = 42) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    df = fetch_california_housing(as_frame=True)
    X, y = df.data, df.target

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    def dataset(X, y):
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y.to_numpy(), dtype=torch.float32).unsqueeze(1),
        )

    return (dataset(X_train, y_train), dataset(X_val, y_val), dataset(X_test, y_test))


torch.manual_seed(42)

EXPERIMENT_NAME = "w05-california-mlp"

setup(EXPERIMENT_NAME)

START_TIME = time.strftime("%Y%m%d-%H%M%S")

HIDDENS_SIZES = [(), (64,), (64, 64), (256, 256)]
LR = 0.03
BATCH_SIZE = 256
EPOCHS = 50
OPTIMIZER = torch.optim.Adam
LOSS_FN = nn.MSELoss

train, val, test = get_data()
in_features = 8
n_classes = 1

for i, hidden_sizes in enumerate(HIDDENS_SIZES):
    with mlflow.start_run(run_name=f"search_run_{i}", tags={"stage": "search", "search_id": START_TIME}):
        model = MLP(in_features, n_classes, hidden_sizes)
        loss_fn = LOSS_FN()
        optimizer = OPTIMIZER(params=model.parameters(), lr=LR)

        train_loader = DataLoader(
            dataset=train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=torch.Generator().manual_seed(42)
        )
        val_loader = DataLoader(
            dataset=val,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=torch.Generator().manual_seed(42)
        )

        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
            val_loss = evaluate(model, val_loader, loss_fn)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss
                },
                step=epoch
            )

        mlflow.log_param("hidden_sizes", hidden_sizes)
        mlflow.log_param("lr", LR)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)

runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME], filter_string=f"tags.search_id = '{START_TIME}'", order_by=["metrics.val_loss ASC"])
best_run = runs.iloc[0]

best_hidden_sizes = literal_eval(best_run["params.hidden_sizes"]) # without literal_eval it runs a string so we cast it back to a tuple

with mlflow.start_run(run_name=f"final-test-{START_TIME}"):
    model = MLP(in_features, n_classes, best_hidden_sizes)
    loss_fn = LOSS_FN()
    optimizer = OPTIMIZER(params=model.parameters(), lr=LR)

    train_loader = DataLoader(
        dataset=train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(42)
    )
    test_loader = DataLoader(
        dataset=test,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(42)
    )

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)

    mlflow.log_param("hidden_sizes", hidden_sizes)
    mlflow.log_param("lr", LR)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)

    test_loss = evaluate(model, test_loader, loss_fn)

    mlflow.log_metric("test_loss", test_loss)
