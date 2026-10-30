import mlflow
import torch
from previous import california_w1, weights_table
from torch import nn
from torch.utils.data import DataLoader

from mlproject.tracking import setup
from mlproject.training.trainer import evaluate, train_one_epoch

torch.manual_seed(42)

LR = 0.01
BATCH_SIZE = 256
EPOCHS = 100

setup("w04-california-torch")
with mlflow.start_run(run_name="torch-sgd"):
    w1 = california_w1()

    model = nn.Linear(8, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LR)

    train_loader = DataLoader(
        dataset=w1.train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    val_loader = DataLoader(
        dataset=w1.val,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = evaluate(model, val_loader, loss_fn)

        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "max_abs_weight_diff": weights_table(w1, model)["diff"].max(),
            },
            step=epoch
        )

    mlflow.log_param("lr", LR)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)

    mlflow.pytorch.log_state_dict(model.state_dict(), "weights.json")
    mlflow.log_table(w1.reference, "w1-reference.json")
