"""
Statistical significance analysis runner.

Loads fold-level metrics from the ablation results and applies:
  1. Wilcoxon signed-rank test (pairwise, all experiment combinations)
  2. Bonferroni correction for multiple comparisons
  3. McNemar's test on concatenated test predictions
  4. Generates significance matrix table and plots

Run AFTER run_ablation.py has completed.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_config
from src.evaluation.statistical_tests import (
    run_pairwise_comparisons,
    mcnemar_test,
)
from src.visualization.tables import build_significance_table
from src.visualization.plots import _save

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_stats")


def load_all_results(results_dir: Path) -> dict:
    """Load all fold metrics and predictions from the ablation run."""
    metrics_dir = results_dir / "metrics"
    results = {}

    for csv_path in sorted(metrics_dir.glob("*_fold_metrics.csv")):
        exp_id = csv_path.stem.replace("_fold_metrics", "")
        fold_df = pd.read_csv(csv_path)
        fold_metrics = fold_df.to_dict("records")

        y_true_path  = metrics_dir / f"{exp_id}_y_true.npy"
        y_pred_path  = metrics_dir / f"{exp_id}_y_pred.npy"
        y_proba_path = metrics_dir / f"{exp_id}_y_proba.npy"

        results[exp_id] = {
            "fold_metrics": fold_metrics,
            "y_true_all":  np.load(y_true_path)  if y_true_path.exists()  else None,
            "y_pred_all":  np.load(y_pred_path)  if y_pred_path.exists()  else None,
            "y_proba_all": np.load(y_proba_path) if y_proba_path.exists() else None,
        }
        logger.info(f"Loaded: {exp_id} ({len(fold_metrics)} folds)")

    return results


def plot_pvalue_heatmap(pairwise_results: dict, output_dir: Path) -> None:
    matrix = pairwise_results["matrix"]
    ids = sorted(matrix.keys())
    alpha = pairwise_results.get("alpha", 0.05)

    data = np.array([[matrix[a].get(b, float("nan")) for b in ids] for a in ids])
    mask = np.eye(len(ids), dtype=bool)

    fig, ax = plt.subplots(figsize=(max(8, len(ids) * 0.9), max(6, len(ids) * 0.75)))
    sns.heatmap(
        data,
        ax=ax,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        vmin=0.0,
        vmax=0.2,
        xticklabels=ids,
        yticklabels=ids,
        linewidths=0.4,
        cbar_kws={"label": f"Adjusted p-value (Bonferroni, α={alpha})"},
    )
    ax.set_title(f"Pairwise Wilcoxon Test — F1 Macro (Bonferroni-corrected)\n"
                 f"Bold cells: p < {alpha} (significant)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save(fig, output_dir, "significance_pvalue_heatmap")


def main():
    config_path = ROOT / "config.yaml"
    config = load_config(config_path)
    results_dir = ROOT / config["paths"]["results"]
    stats_dir   = results_dir / "statistical_tests"
    plots_dir   = results_dir / "plots"
    alpha = config["evaluation"]["significance_level"]

    logger.info("Loading ablation results...")
    all_results = load_all_results(results_dir)

    if len(all_results) < 2:
        logger.error("Need at least 2 experiments to run statistical tests.")
        return

    # ── 1. Wilcoxon pairwise comparisons ─────────────────────────────────
    logger.info("\nRunning pairwise Wilcoxon tests (F1 Macro)...")
    pairwise_f1 = run_pairwise_comparisons(all_results, metric="f1_macro", alpha=alpha)

    logger.info("\nRunning pairwise Wilcoxon tests (MCC)...")
    pairwise_mcc = run_pairwise_comparisons(all_results, metric="mcc", alpha=alpha)

    # Save results
    stats_dir.mkdir(parents=True, exist_ok=True)
    with open(stats_dir / "wilcoxon_f1_macro.json", "w") as f:
        json.dump(pairwise_f1, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else x)
    with open(stats_dir / "wilcoxon_mcc.json", "w") as f:
        json.dump(pairwise_mcc, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else x)

    build_significance_table(pairwise_f1, stats_dir)
    plot_pvalue_heatmap(pairwise_f1, plots_dir)

    # ── 2. McNemar's test (best model pairs) ──────────────────────────────
    logger.info("\nRunning McNemar pairwise tests...")
    mcnemar_results = []
    exp_ids = list(all_results.keys())
    for i in range(len(exp_ids)):
        for j in range(i + 1, len(exp_ids)):
            id_a, id_b = exp_ids[i], exp_ids[j]
            if all_results[id_a]["y_true_all"] is None or all_results[id_b]["y_true_all"] is None:
                continue
            # Only compare on matching test sets (same length)
            ya = all_results[id_a]["y_pred_all"]
            yb = all_results[id_b]["y_pred_all"]
            yt = all_results[id_a]["y_true_all"]
            if len(ya) != len(yb) or len(ya) != len(yt):
                continue
            result = mcnemar_test(yt, ya, yb)
            mcnemar_results.append({"a": id_a, "b": id_b, **result})
            if result["pvalue"] < alpha:
                logger.info(f"  SIGNIFICANT: {id_a} vs {id_b} | p={result['pvalue']:.4f}")

    pd.DataFrame(mcnemar_results).to_csv(stats_dir / "mcnemar_results.csv", index=False)

    # ── 3. Summary log ────────────────────────────────────────────────────
    sig_pairs = [p for p in pairwise_f1["pairs"] if p.get("significant")]
    logger.info(f"\n{'='*60}")
    logger.info(f"STATISTICAL SUMMARY")
    logger.info(f"  Experiments compared : {len(all_results)}")
    logger.info(f"  Total pairs tested   : {len(pairwise_f1['pairs'])}")
    logger.info(f"  Significant pairs    : {len(sig_pairs)} (F1 Macro, α={alpha} Bonferroni)")
    logger.info(f"  Results saved to     : {stats_dir}")
    logger.info(f"{'='*60}")

    for p in sig_pairs:
        logger.info(
            f"  {p['a']} vs {p['b']} | "
            f"Δ={p.get('mean_diff', 0):+.4f} | p_adj={p['adj_pvalue']:.4f}"
        )

    logger.info("\n✓ Statistical analysis complete.")


if __name__ == "__main__":
    main()
