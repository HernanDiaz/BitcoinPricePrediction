"""
Bootstrap confidence intervals for classification metrics.
Resamples the test predictions n_iterations times to estimate
the sampling distribution of each metric.
"""

import numpy as np
from .metrics import compute_metrics


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    n_iterations: int = 1000,
    ci: float = 0.95,
    random_seed: int = 42,
) -> dict:
    """
    Compute bootstrap confidence intervals for all metrics.

    Returns:
        dict mapping metric_name → {"mean", "lower", "upper", "std", "values"}
    """
    rng = np.random.default_rng(random_seed)
    n = len(y_true)

    scalar_keys = [
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "f1_weighted", "roc_auc", "mcc",
    ]

    bootstrap_values: dict[str, list] = {k: [] for k in scalar_keys}

    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        ypr = y_proba[idx] if y_proba is not None else None

        # Skip iterations where only one class is present (roc_auc undefined)
        if len(np.unique(yt)) < 2:
            continue

        m = compute_metrics(yt, yp, ypr)
        for key in scalar_keys:
            val = m.get(key, float("nan"))
            if not np.isnan(val):
                bootstrap_values[key].append(val)

    alpha = 1.0 - ci
    results = {}
    for key in scalar_keys:
        vals = np.array(bootstrap_values[key])
        if len(vals) == 0:
            results[key] = {"mean": float("nan"), "lower": float("nan"), "upper": float("nan"), "std": float("nan")}
            continue
        results[key] = {
            "mean":   round(float(vals.mean()), 6),
            "lower":  round(float(np.percentile(vals, alpha / 2 * 100)), 6),
            "upper":  round(float(np.percentile(vals, (1 - alpha / 2) * 100)), 6),
            "std":    round(float(vals.std(ddof=1)), 6),
            "n_valid_iterations": len(vals),
        }

    return results
