"""
Evaluation metrics for binary classification.

All metrics required by Q1 AI journals for imbalanced classification:
  - Accuracy
  - Precision / Recall / F1 (macro and weighted)
  - AUC-ROC
  - Matthews Correlation Coefficient (MCC) — essential for imbalanced targets
  - Log-loss
  - Confusion matrix
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    log_loss,
    confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict:
    """
    Compute the full set of classification metrics.

    Args:
        y_true  : true binary labels (0/1)
        y_pred  : predicted binary labels (0/1)
        y_proba : predicted probabilities for class 1, shape (n,) or (n, 2)

    Returns:
        dict with all metrics as floats, plus the confusion matrix.
    """
    if y_proba is not None and y_proba.ndim == 2:
        y_proba_pos = y_proba[:, 1]
    else:
        y_proba_pos = y_proba

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    metrics = {
        "accuracy":         round(accuracy_score(y_true, y_pred), 6),
        "precision_macro":  round(precision_score(y_true, y_pred, average="macro",   zero_division=0), 6),
        "precision_weighted": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 6),
        "recall_macro":     round(recall_score(y_true, y_pred, average="macro",    zero_division=0), 6),
        "recall_weighted":  round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 6),
        "f1_macro":         round(f1_score(y_true, y_pred, average="macro",    zero_division=0), 6),
        "f1_weighted":      round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 6),
        "mcc":              round(matthews_corrcoef(y_true, y_pred), 6),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n_test": int(len(y_true)),
        "class_1_ratio": round(float(y_true.mean()), 4),
    }

    if y_proba_pos is not None:
        try:
            metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba_pos), 6)
            metrics["log_loss"] = round(log_loss(y_true, y_proba_pos), 6)
        except ValueError:
            metrics["roc_auc"] = float("nan")
            metrics["log_loss"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
        metrics["log_loss"] = float("nan")

    return metrics


def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """
    Aggregate metrics across walk-forward folds.
    Returns mean, std, and raw per-fold values for each metric.
    """
    scalar_keys = [
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "f1_weighted", "roc_auc", "mcc", "log_loss",
    ]
    aggregated = {}
    for key in scalar_keys:
        values = np.array([m[key] for m in fold_metrics if not np.isnan(m.get(key, float("nan")))])
        if len(values) > 0:
            aggregated[f"{key}_mean"] = round(float(values.mean()), 6)
            aggregated[f"{key}_std"]  = round(float(values.std(ddof=1)), 6)
            aggregated[f"{key}_values"] = values.tolist()
        else:
            aggregated[f"{key}_mean"] = float("nan")
            aggregated[f"{key}_std"]  = float("nan")
            aggregated[f"{key}_values"] = []

    return aggregated
