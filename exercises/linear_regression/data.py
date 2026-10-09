from sklearn.datasets import fetch_california_housing
import pandas as pd


def load_frame():
    df = fetch_california_housing(as_frame=True)
    return df.data.join(df.target)


def split_data(df: pd.DataFrame, seed=42):
    n_rows = len(df)
    train_split = int(0.6 * n_rows)
    val_split = int(0.2 * n_rows)
    test_split = int(0.2 * n_rows)

    test = df.sample(n=test_split, random_state=seed)
    train_val = df.drop(test.index)
    val = train_val.sample(n=val_split, random_state=seed)
    train = train_val.drop(val.index)
    return train, val, test
