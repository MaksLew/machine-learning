"""Reusable classification metrics and threshold selection."""

from collections.abc import Iterable

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def binary_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def choose_threshold(
    y_true,
    probabilities,
    min_recall: float,
    thresholds: Iterable[float] = tuple(step / 100 for step in range(5, 100, 5)),
) -> float:
    """Return the highest candidate threshold that satisfies the required recall."""
    valid = [
        float(threshold)
        for threshold in thresholds
        if recall_score(y_true, probabilities >= threshold, zero_division=0) >= min_recall
    ]
    if not valid:
        raise ValueError(f"no threshold reaches min_recall={min_recall}")
    return max(valid)
