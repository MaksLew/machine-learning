"""A Dataset for a table (W4, task Z4): one row on request.

The same three methods as every Dataset, including the image Dataset in W5; only __getitem__ changes.
Check it with:
    uv run pytest exercises/w04
"""

import torch
from torch.utils.data import Dataset


class Standardize:
    """x -> (x - mean) / std, with mean and std computed on the training part only (as StandardScaler in W1)."""

    def __init__(self, frame, features):
        # TODO (Z4): mean and standard deviation of every feature column of frame, as float32 tensors
        raise NotImplementedError("Z4: Standardize.__init__")

    def __call__(self, x):
        # TODO (Z4)
        raise NotImplementedError("Z4: Standardize.__call__")


class TabularDataset(Dataset):
    """Rows of a pandas DataFrame as (features, target) pairs.

    frame: the DataFrame of one part (training, validation or test);
    features: the list of feature columns; target: the name of the target column;
    transform: applied to the feature tensor of every row (e.g. Standardize), or None.
    __getitem__ returns x of shape (len(features),) and y of shape (1,), both float32.
    """

    def __init__(self, frame, features, target, transform=None):
        # TODO (Z4)
        raise NotImplementedError("Z4: TabularDataset.__init__")

    def __len__(self):
        # TODO (Z4)
        raise NotImplementedError("Z4: TabularDataset.__len__")

    def __getitem__(self, idx):
        # TODO (Z4)
        raise NotImplementedError("Z4: TabularDataset.__getitem__")
