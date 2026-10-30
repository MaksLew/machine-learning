"""Reusable training loop (W4, task Z3). From W5 on, every model is trained with it.

Check it with:
    uv run pytest exercises/w04
"""

import torch

def train_one_epoch(model, loader, loss_fn, optimizer):
    """One pass over loader with a step per batch. Returns the mean loss over all examples of the epoch.

    Per batch: zero the gradients, compute the loss, backward, optimizer step. Put the model in training
    mode first (model.train()). The mean is over examples, so weight each batch loss by the batch size:
    the last batch may be smaller.
    """
    # TODO (Z3)
    model.train()
    total_loss = total_examples = 0

    for x, y in loader:
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        total_examples += len(y)

    return total_loss / total_examples



def evaluate(model, loader, loss_fn):
    """Mean loss over all examples of loader, without changing the model and without gradients.

    Use model.eval() and torch.no_grad(). Weight each batch loss by the batch size.
    """
    # TODO (Z3)

    model.eval()
    total_loss = total_examples = 0

    for x, y in loader:
        with torch.no_grad():
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item() * len(y)
            total_examples += len(y)

    return total_loss / total_examples
