import time
import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms

from mlproject.data.dataset import ImageDataset, read_split
from mlproject.evaluation.classification import confusion, confusion_figure, predict
from mlproject.tracking import setup
from mlproject.models.mlp import MLP
from mlproject.models.cnn import SimpleCNN
from mlproject.training.trainer import evaluate, train_one_epoch

torch.manual_seed(42)

BATCH_SIZE = 64
LR = 0.001
EPOCHS = 15
LOSS_FN = nn.CrossEntropyLoss
OPTIMIZER = torch.optim.Adam

parser = argparse.ArgumentParser()
parser.add_argument("--model")
args = parser.parse_args()

if args.model not in ["mlp", "cnn"]:
    raise ValueError(f"no such arch as `{args.model}`. try `mlp` or `cnn`")


setup("w06-eurostat")

split_dir = Path("data/eurosat/splits")
transform = transforms.ToTensor()

train_dataset = ImageDataset(read_split(split_dir / "train.csv"), transform=transform)

scaler = StandardScaler()
for images, _ in DataLoader(train_dataset, batch_size=BATCH_SIZE):
    pixels = images.permute(0, 2, 3, 1).reshape(-1, 3).numpy()
    scaler.partial_fit(pixels)

assert scaler.mean_ is not None and scaler.scale_ is not None
mean = np.asarray(scaler.mean_).tolist()
std = np.asarray(scaler.scale_).tolist()
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)
train_dataset.transform = transform

val_dataset = ImageDataset(read_split(split_dir / "val.csv"), transform=transform)
test_dataset = ImageDataset(read_split(split_dir / "test.csv"), transform=transform)


with mlflow.start_run(run_name=f"eurosat_{time.strftime("%Y%m%d-%H%M%S")}"):
    if args.model == "mlp":
        model = MLP(3 * 64 * 64, 10, hidden=(256,))
    else:
        model = SimpleCNN(10, 64)
    loss_fn = LOSS_FN()
    optimizer = OPTIMIZER(params=model.parameters(), lr=LR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
        num_workers=4,
    )


    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = evaluate(model, val_loader, loss_fn)
        y_true, y_pred = predict(model, val_loader)
        val_accuracy = float((y_true == y_pred).mean())

        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            },
            step=epoch
        )

    class_names = json.loads((split_dir / "classes.json").read_text())
    y_true, y_pred = predict(model, test_loader)
    cm = confusion(y_true, y_pred, n_classes=len(class_names))
    fig = confusion_figure(cm, class_names)
    mlflow.log_figure(fig, "test_confusion_matrix.png")
    plt.close(fig)

    mlflow.log_param("lr", LR)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)

    model.eval()
    correct_shift16 = correct_rot90 = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            shifted = torch.roll(x, (16, 16), dims=(2, 3))
            rotated = torch.rot90(x, 1, dims=(2, 3))
            correct_shift16 += (model(shifted).argmax(dim=1) == y).sum().item()
            correct_rot90 += (model(rotated).argmax(dim=1) == y).sum().item()
            total += len(y)

    mlflow.log_metrics({
        "val_accuracy_shift16": correct_shift16 / total,
        "val_accuracy_rot90": correct_rot90 / total,
    })
