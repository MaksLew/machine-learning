import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlproject.evaluation.metrics import binary_metrics, choose_threshold
from mlproject.tracking import setup

SEED = 42
MIN_RECALL = 0.98
TARGET = "malignant"

setup("w02-breast-cancer")
data = load_breast_cancer(as_frame=True)
X = data.data
y = (data.target == 0).astype(int).rename(TARGET)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    stratify=y_train_val,
    random_state=SEED,
)

with mlflow.start_run(run_name="breast-cancer"):
    mlflow.log_dict(
        {
            "splits": {
                "train": {"rows": len(X_train), "malignant": int(y_train.sum())},
                "validation": {"rows": len(X_val), "malignant": int(y_val.sum())},
                "test": {"rows": len(X_test), "malignant": int(y_test.sum())},
            },
            "n_features": X.shape[1],
            "feature_types": {str(dtype): int(count) for dtype, count in X.dtypes.value_counts().items()},
            "missing_values": int(X.isna().sum().sum()),
        },
        "data_description.json",
    )
    mlflow.log_input(
        mlflow.data.from_pandas(X_train.assign(**{TARGET: y_train}), name="train", targets=TARGET),  # ty: ignore
        context="training",
    )
    mlflow.log_input(
        mlflow.data.from_pandas(X_val.assign(**{TARGET: y_val}), name="validation", targets=TARGET),  # ty: ignore
        context="validation",
    )

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    val_probabilities = model.predict_proba(X_val)[:, 1]
    threshold = choose_threshold(y_val, val_probabilities, MIN_RECALL, np.arange(0.05, 1.0, 0.05))
    val_predictions = val_probabilities >= threshold

    mlflow.log_params({"min_recall": MIN_RECALL, "threshold": threshold})
    mlflow.set_tag(
        "min_recall_rationale",
        "High recall limits missed malignant tumors; 0.98 allows at most about one miss on validation.",
    )
    mlflow.log_metrics({f"val_{name}": value for name, value in binary_metrics(y_val, val_predictions).items()})
    mlflow.log_metric("val_auc", roc_auc_score(y_val, val_probabilities))

    RocCurveDisplay.from_predictions(y_val, val_probabilities)
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "validation_roc.png")
    plt.close()

    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = test_probabilities >= threshold
    mlflow.log_metrics(
        {f"test_{name}": value for name, value in binary_metrics(y_test, test_predictions).items()}
    )
    mlflow.set_tag("test_evaluated", "true")

    ConfusionMatrixDisplay.from_predictions(y_test, test_predictions)
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "test_confusion_matrix.png")
    plt.close()

    mlflow.sklearn.log_model(model, name="model", input_example=X_train.head(5))
