"""W3: what the weights of a fitted model say. Complete weight_summary and top_weights in Z1, choose_alpha in Z3."""

import pandas as pd
from problems import NOISE_PREFIX


def weight_summary(weights: pd.Series) -> dict:
    """Counts of weights, for mlflow.log_metrics.

    weights: one weight per feature, indexed by the feature name (as returned by fit_and_log).
    Returns a dict with the keys:
        nonzero        - weights different from 0
        zero           - weights exactly equal to 0
        noise_nonzero  - nonzero weights of the noise columns (names starting with NOISE_PREFIX)
        real_nonzero   - nonzero weights of the other columns
    All values are ints.
    """
    nonzero = sum(
        weights[row] != 0
        for row in weights.index
    )

    zero = sum(
        weights[row] == 0
        for row in weights.index
    )

    noise_nonzero = sum(
        row.startswith(NOISE_PREFIX) and weights[row] != 0
        for row in weights.index
    )

    real_nonzero = sum(
        not row.startswith(NOISE_PREFIX) and weights[row] != 0
        for row in weights.index
    )

    return {
        "nonzero": nonzero,
        "zero": zero,
        "noise_nonzero": noise_nonzero,
        "real_nonzero": real_nonzero,
    }


def top_weights(weights: pd.Series, k: int = 5) -> dict:
    """The k weights with the largest absolute value, for mlflow.log_dict.

    Returns {feature name: weight as a float}, the largest |weight| first.
    """
    return weights.sort_values(ascending=False).head(k).to_dict()


def choose_alpha(search: pd.DataFrame) -> float:
    """The alpha with the lowest validation MSE.

    search: a DataFrame with the columns alpha and val_mse (as returned by search_alpha).
    Returns the alpha as a float.
    """
    # TODO (Z3)
    raise NotImplementedError
