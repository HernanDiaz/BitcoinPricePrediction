"""
SHAP-based feature importance analysis.

Computes SHAP values for tree-based models (RandomForest, XGBoost)
using TreeExplainer and for CNN-LSTM using DeepExplainer.

The DualEncoder uses its own attention weights mechanism (see dual_encoder.py)
and does not require SHAP — the cross-attention weights are directly interpretable.
"""

import logging
from pathlib import Path

import numpy as np
import shap

logger = logging.getLogger(__name__)


def compute_shap_tree(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
) -> dict:
    """
    Compute SHAP values for RandomForest or XGBoost using TreeExplainer.

    Args:
        model         : fitted sklearn or XGBoost model (with .predict_proba)
        X_background  : background dataset for SHAP (typically training set, subsampled)
        X_explain     : samples to explain (typically test set)
        feature_names : list of feature column names

    Returns:
        {
          "shap_values"       : np.ndarray (n_samples, n_features) — values for class 1
          "base_value"        : float
          "feature_names"     : list[str]
          "mean_abs_shap"     : np.ndarray (n_features,) — mean |SHAP| per feature
          "feature_importance": list of (feature, importance) sorted descending
        }
    """
    # Subsample background to speed up computation (max 500 samples)
    if len(X_background) > 500:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_background), 500, replace=False)
        X_bg = X_background[idx]
    else:
        X_bg = X_background

    clf = model.clf  # unwrap from our wrapper
    explainer = shap.TreeExplainer(clf, data=X_bg, feature_perturbation="interventional")
    shap_values = explainer(X_explain)

    # For binary classification, shap_values.values has shape (n, features, 2)
    # We take class 1 (positive class)
    if shap_values.values.ndim == 3:
        sv = shap_values.values[:, :, 1]
        base_value = float(shap_values.base_values[0, 1])
    else:
        sv = shap_values.values
        base_value = float(shap_values.base_values[0]) if hasattr(shap_values.base_values, '__len__') else float(shap_values.base_values)

    mean_abs = np.abs(sv).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    feature_importance = [(feature_names[i], round(float(mean_abs[i]), 6)) for i in sorted_idx]

    return {
        "shap_values": sv,
        "base_value": base_value,
        "feature_names": feature_names,
        "mean_abs_shap": mean_abs,
        "feature_importance": feature_importance,
    }


def compute_shap_deep(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
) -> dict:
    """
    Compute SHAP values for CNN-LSTM using DeepExplainer.
    X arrays must be (n_samples, seq_len, n_features).
    Importance is averaged over the sequence dimension.
    """
    import torch

    device = model._device
    net = model._net
    net.eval()

    # Subsample background
    if len(X_background) > 100:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_background), 100, replace=False)
        X_bg = X_background[idx]
    else:
        X_bg = X_background

    bg_tensor = torch.tensor(X_bg, dtype=torch.float32).to(device)
    ex_tensor = torch.tensor(X_explain[:200], dtype=torch.float32).to(device)  # limit for speed

    # Wrap model to output scalar (sigmoid probability for class 1)
    class _Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return torch.sigmoid(self.inner(x))

    wrapper = _Wrapper(net).to(device)
    explainer = shap.DeepExplainer(wrapper, bg_tensor)

    try:
        sv = explainer.shap_values(ex_tensor)
        # sv shape: (n_samples, seq_len, n_features)
        sv = np.array(sv)
        # Average importance over sequence dimension
        sv_flat = sv.mean(axis=1)  # (n_samples, n_features)
    except Exception as e:
        logger.warning(f"DeepExplainer failed ({e}). Falling back to zero SHAP values.")
        sv_flat = np.zeros((len(X_explain[:200]), len(feature_names)))

    mean_abs = np.abs(sv_flat).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    feature_importance = [(feature_names[i], round(float(mean_abs[i]), 6)) for i in sorted_idx]

    return {
        "shap_values": sv_flat,
        "base_value": float(explainer.expected_value[0]) if hasattr(explainer, "expected_value") else 0.0,
        "feature_names": feature_names,
        "mean_abs_shap": mean_abs,
        "feature_importance": feature_importance,
    }


def save_shap_results(shap_result: dict, output_dir: Path, experiment_id: str) -> None:
    """Persist SHAP values and feature importance to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{experiment_id}_shap_values.npy", shap_result["shap_values"])
    np.save(output_dir / f"{experiment_id}_mean_abs_shap.npy", shap_result["mean_abs_shap"])

    import json
    meta = {
        "feature_names": shap_result["feature_names"],
        "feature_importance": shap_result["feature_importance"],
        "base_value": shap_result["base_value"],
        "experiment_id": experiment_id,
    }
    with open(output_dir / f"{experiment_id}_shap_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"SHAP results saved → {output_dir / experiment_id}*")
