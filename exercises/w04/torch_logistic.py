import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from previous import breast_cancer_w2

from mlproject.tracking import setup
from mlproject.training.trainer import evaluate, train_one_epoch

torch.manual_seed(42)

LR = 0.1
BATCH_SIZE = 32
EPOCHS = 1000

setup("w04-breast-cancer-torch")
with mlflow.start_run(run_name="no-penalty"):
    w2 = breast_cancer_w2()

    model = nn.Linear(30, 1)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LR)

    train_loader = DataLoader(
        dataset=w2.train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    val_loader = DataLoader(
        dataset=w2.val,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = evaluate(model, val_loader, loss_fn)

        if epoch % 10 == 0:
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
            )

    mlflow.log_param("lr", LR)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)

    mlflow.pytorch.log_state_dict(model.state_dict(), "weights.json")
    mlflow.log_metric("val_auc", roc_auc_score(w2.val[1], model(w2.val[1])))
