import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


REGION_CENTERS = {
    "LA": (34.05, -118.24),
    "SF": (37.77, -122.42),
    "SD": (32.72, -117.16),
    "SJ": (37.34, -121.89),
    "SAC": (38.58, -121.49),
}


def load_frame():
    df = fetch_california_housing(as_frame=True)
    return df.data.join(df.target)


def split_data(df: pd.DataFrame, seed=42):
    train_val, test = train_test_split(df, test_size=0.2, random_state=seed)
    train, val = train_test_split(train_val, test_size=0.25, random_state=seed)
    return train, val, test


def add_region(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    names = np.array(list(REGION_CENTERS))
    centers = np.array(list(REGION_CENTERS.values()))
    distances = np.linalg.norm(
        result[["Latitude", "Longitude"]].to_numpy()[:, None] - centers, axis=2
    )
    nearest = distances.argmin(axis=1)
    result["region"] = np.where(distances.min(axis=1) < 0.75, names[nearest], "other")
    return result


def one_hot(
    train: pd.DataFrame,
    other: pd.DataFrame,
    column: str,
    drop_first: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    categories = sorted(train[column].dropna().unique())
    if drop_first:
        categories = categories[1:]
    new_columns = [f"{column}_{category}" for category in categories]

    def encode(df: pd.DataFrame) -> pd.DataFrame:
        result = df.drop(columns=column).copy()
        for category, name in zip(categories, new_columns):
            result[name] = (df[column] == category).astype(int)
        return result

    return encode(train), encode(other), new_columns
