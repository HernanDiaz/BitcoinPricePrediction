"""
Statistical significance tests for model comparison.

Tests applied:
  - McNemar's test       : pairwise comparison of two classifiers on the same test set
  - Wilcoxon signed-rank : pairwise comparison of metric distributions across folds
  - Bonferroni correction: adjusts p-values for multiple comparisons

References:
  Dietterich (1998). Approximate Statistical Tests for Comparing Supervised
  Classification Learning Algorithms. Neural Computation.
"""

import itertools
import logging

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

logger = logging.getLogger(__name__)


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> dict:
    """
    McNemar's test comparing two classifiers on the same test set.

    Null hypothesis: both classifiers have the same error rate.

    Returns:
        {"statistic": float, "pvalue": float, "b": int, "c": int}
        where b = A correct, B wrong; c = A wrong, B correct
    """
    b = int(((y_pred_a == y_true) & (y_pred_b != y_true)).sum())
    c = int(((y_pred_a != y_true) & (y_pred_b == y_true)).sum())

    contingency = np.array([[0, b], [c, 0]])
    # Use exact binomial test when b+c < 25
    exact = (b + c) < 25
    result = mcnemar(contingency, exact=exact, correction=not exact)

    return {
        "statistic": round(float(result.statistic), 6),
        "pvalue":    round(float(result.pvalue), 6),
        "b": b,
        "c": c,
        "exact": exact,
    }


def wilcoxon_test(
    values_a: list[float],
    values_b: list[float],
    metric_name: str = "metric",
) -> dict:
    """
    Wilcoxon signed-rank test comparing metric distributions across folds.

    Null hypothesis: the median difference between paired samples is zero.
    Requires at least 5 fold pairs.

    Returns:
        {"statistic": float, "pvalue": float, "n_pairs": int}
    """
    a = np.array(values_a)
    b = np.array(values_b)

    if len(a) != len(b):
        raise ValueError("Both arrays must have the same length (one value per fold).")

    if len(a) < 5:
        logger.warning(
            f"Wilcoxon test for '{metric_name}': only {len(a)} pairs — "
            f"result may be unreliable (recommended >= 5)."
        )

    diff = a - b
    if np.all(diff == 0):
        return {"statistic": 0.0, "pvalue": 1.0, "n_pairs": len(a), "note": "all_equal"}

    stat, pval = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return {
        "statistic": round(float(stat), 6),
        "pvalue":    round(float(pval), 6),
        "n_pairs":   len(a),
        "mean_diff": round(float(diff.mean()), 6),
        "std_diff":  round(float(diff.std(ddof=1)), 6),
    }


def bonferroni_correction(pvalues: list[float], alpha: float = 0.05) -> list[float]:
    """
    Apply Bonferroni correction to a list of p-values.
    Returns adjusted p-values capped at 1.0.
    """
    n = len(pvalues)
    return [min(p * n, 1.0) for p in pvalues]


def run_pairwise_comparisons(
    results: dict,
    metric: str = "f1_macro",
    alpha: float = 0.05,
) -> dict:
    """
    Run all pairwise Wilcoxon tests between model×group combinations.

    Args:
        results: {experiment_id: {"fold_metrics": [metric_dict, ...]}}
        metric: the metric to compare (key in each metric_dict)
        alpha: significance level (before Bonferroni correction)

    Returns:
        {
          "pairs": [{"a": id_a, "b": id_b, "raw_pvalue": ..., "adj_pvalue": ..., "significant": bool}],
          "matrix": nested dict for heatmap
        }
    """
    experiment_ids = list(results.keys())
    pairs = list(itertools.combinations(experiment_ids, 2))

    raw_pvalues = []
    pair_results = []

    for id_a, id_b in pairs:
        vals_a = [m[metric] for m in results[id_a]["fold_metrics"]]
        vals_b = [m[metric] for m in results[id_b]["fold_metrics"]]
        test = wilcoxon_test(vals_a, vals_b, metric_name=metric)
        pair_results.append({"a": id_a, "b": id_b, **test})
        raw_pvalues.append(test["pvalue"])

    # Bonferroni-corrected p-values
    adj_pvalues = bonferroni_correction(raw_pvalues, alpha)

    output_pairs = []
    for i, pr in enumerate(pair_results):
        pr["raw_pvalue"] = raw_pvalues[i]
        pr["adj_pvalue"] = round(adj_pvalues[i], 6)
        pr["significant"] = adj_pvalues[i] < alpha
        output_pairs.append(pr)

    # Build symmetric matrix for easy heatmap plotting
    matrix: dict[str, dict[str, float]] = {eid: {} for eid in experiment_ids}
    for pr in output_pairs:
        matrix[pr["a"]][pr["b"]] = pr["adj_pvalue"]
        matrix[pr["b"]][pr["a"]] = pr["adj_pvalue"]
    for eid in experiment_ids:
        matrix[eid][eid] = 1.0

    return {"pairs": output_pairs, "matrix": matrix, "metric": metric, "alpha": alpha}
