#!/usr/bin/env python3
"""
run_paper_plots.py
==================
Generates all final figures for the paper from Optuna JSON results.

Figures produced:
  Fig A — Model comparison bar chart (Acc / MCC / AUC, folds 1-6)
  Fig B — Per-fold MCC chart (all 5 models across 7 folds)
  Fig C — ROC curves (all 5 models, folds 1-6 concatenated)
  Fig D — Wilcoxon significance heatmap (loaded from stats output)

Run AFTER: run_paper_stats.py and run_paper_shap.py
"""

import sys, json, logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, auc

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "onchain_direction"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("paper_plots")

RESULTS_DIR = ROOT / "results" / "optuna"
STATS_DIR   = ROOT / "results" / "statistical_tests"
PLOTS_DIR   = ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Paleta y etiquetas ────────────────────────────────────────────────────────
MODEL_META = {
    "XGBoost":         {"label": "XGBoost G3",           "color": "#E15759", "marker": "o"},
    "LightGBM":        {"label": "LightGBM G3",          "color": "#F28E2B", "marker": "s"},
    "MLP_Simple":      {"label": "MLP Simple",           "color": "#76B7B2", "marker": "^"},
    "MLP_Dual_NoAttn": {"label": "MLP Dual (no attn.)",  "color": "#59A14F", "marker": "D"},
    "MLP_Dual_V2":     {"label": "MLP Dual Encoder V2",  "color": "#4E79A7", "marker": "*"},
}

FOLD_LABELS = ["2019", "2020", "2021", "2022", "2023", "2024", "2025"]


def save_fig(fig, name: str):
    fig.savefig(PLOTS_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(PLOTS_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {name}.pdf/.png")


# ── Load data ─────────────────────────────────────────────────────────────────
def load_model_data():
    data = {}
    mapping = {
        "XGBoost":         "xgboost_g3_optuna_final.json",
        "LightGBM":        "lightgbm_g3_optuna_final.json",
        "MLP_Simple":      "mlp_simple_optuna.json",
        "MLP_Dual_NoAttn": "mlp_dual_noattn_optuna.json",
        "MLP_Dual_V2":     "mlp_dual_v2_optuna.json",
    }
    for key, fname in mapping.items():
        d = json.load(open(RESULTS_DIR / fname))
        # Per-fold MCC: from fold_metrics if available, else from summary values
        if "fold_metrics" in d:
            fold_mccs = [fm["mcc"] for fm in d["fold_metrics"]]
            fold_accs = [fm["accuracy"] for fm in d["fold_metrics"]]
            fold_aucs = [fm["roc_auc"] for fm in d["fold_metrics"]]
        else:
            # XGBoost / LightGBM: load from stats output (recomputed)
            stats_path = STATS_DIR / "statistical_tests_full.json"
            if stats_path.exists():
                stats = json.load(open(stats_path))
                summary = next((r for r in stats["model_summary"] if r["Model"] == key), None)
                if summary:
                    fold_mccs = summary["MCC_values"] + [None]  # 6 stable + placeholder
                else:
                    fold_mccs = [None] * 7
            else:
                fold_mccs = [None] * 7
            fold_accs = d["summary_stable"].get("accuracy_mean", None)
            fold_aucs = d["summary_stable"].get("roc_auc_mean", None)

        data[key] = {
            "fold_mccs": fold_mccs,
            "fold_accs": fold_accs if isinstance(fold_accs, list) else None,
            "fold_aucs": fold_aucs if isinstance(fold_aucs, list) else None,
            "summary_stable": d["summary_stable"],
            "y_true":  np.array(d["y_true_all"]),
            "y_proba": np.array(d["y_proba_all"]),
        }
    return data


data = load_model_data()

# Fold boundary sizes (approximate: cumulative test samples)
# Each test fold ≈ 365 samples. 7 folds × ~364 = 2548 total
fold_sizes = []
cumsum = 0
for i in range(7):
    fold_sizes.append(cumsum)
    cumsum += 364
fold_sizes.append(cumsum)


# ─────────────────────────────────────────────────────────────────────────────
# Fig A — Model comparison bar chart (folds 1-6)
# ─────────────────────────────────────────────────────────────────────────────
log.info("\n[Fig A] Model comparison bar chart")

metrics_to_plot = ["accuracy_mean", "mcc_mean", "roc_auc_mean"]
metric_labels   = ["Accuracy", "MCC", "AUC-ROC"]
metric_stds     = ["accuracy_std", "mcc_std", "roc_auc_std"]

model_order = ["XGBoost", "LightGBM", "MLP_Simple", "MLP_Dual_NoAttn", "MLP_Dual_V2"]
n_models    = len(model_order)
n_metrics   = len(metrics_to_plot)
x           = np.arange(n_metrics)
width       = 0.15

fig, ax = plt.subplots(figsize=(11, 6))

for i, key in enumerate(model_order):
    vals = [data[key]["summary_stable"].get(m, 0) for m in metrics_to_plot]
    errs = [data[key]["summary_stable"].get(s, 0) for s in metric_stds]
    offset = (i - n_models / 2 + 0.5) * width
    bars = ax.bar(x + offset, vals, width, label=MODEL_META[key]["label"],
                  color=MODEL_META[key]["color"], alpha=0.88,
                  yerr=errs, capsize=3, error_kw={"linewidth": 1})

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_title("Model Comparison — Walk-Forward CV (Folds 1–6, 2019–2024)", fontsize=13)
ax.legend(loc="upper left", fontsize=9, ncol=2)
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)

# Add value labels on top of bars (only BarContainer, skip ErrorbarContainer)
import matplotlib.container as mcontainer
for bar_group in ax.containers:
    if isinstance(bar_group, mcontainer.BarContainer):
        ax.bar_label(bar_group, fmt="%.3f", fontsize=6.5, padding=2, rotation=90)

plt.tight_layout()
save_fig(fig, "fig_model_comparison_bar")


# ─────────────────────────────────────────────────────────────────────────────
# Fig B — Per-fold MCC chart (all 7 folds)
# ─────────────────────────────────────────────────────────────────────────────
log.info("[Fig B] Per-fold MCC chart")

fig, ax = plt.subplots(figsize=(12, 5))

for key in model_order:
    mccs = data[key]["fold_mccs"]
    valid = [m for m in mccs if m is not None]
    x_vals = list(range(len(valid)))
    meta = MODEL_META[key]
    ax.plot(x_vals, valid, marker=meta["marker"], color=meta["color"],
            label=meta["label"], linewidth=2, markersize=7)

ax.axvspan(5.5, 6.5, alpha=0.12, color="red", label="2025 (regime change)")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.set_xticks(range(7))
ax.set_xticklabels(FOLD_LABELS, fontsize=11)
ax.set_xlabel("Test Year (Fold)", fontsize=12)
ax.set_ylabel("MCC", fontsize=12)
ax.set_title("Per-Fold MCC — Walk-Forward Expanding Window CV", fontsize=13)
ax.legend(loc="lower left", fontsize=9, ncol=2)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
save_fig(fig, "fig_per_fold_mcc")


# ─────────────────────────────────────────────────────────────────────────────
# Fig C — ROC curves (folds 1-6 concatenated)
# ─────────────────────────────────────────────────────────────────────────────
log.info("[Fig C] ROC curves (folds 1-6)")

# folds 1-6: first 6*~364 = ~2184 samples
n_stable = 6 * 364   # approximate; use all but last fold

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)

for key in model_order:
    y_true  = data[key]["y_true"][:n_stable]
    y_proba = data[key]["y_proba"][:n_stable]
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc_val = auc(fpr, tpr)
    meta = MODEL_META[key]
    ax.plot(fpr, tpr, color=meta["color"], linewidth=2,
            label=f"{meta['label']} (AUC={roc_auc_val:.3f})")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — Folds 1–6 (2019–2024)", fontsize=13)
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
save_fig(fig, "fig_roc_curves")


# ─────────────────────────────────────────────────────────────────────────────
# Fig D — Wilcoxon p-value heatmap
# ─────────────────────────────────────────────────────────────────────────────
log.info("[Fig D] Wilcoxon heatmap")

stats_path = STATS_DIR / "wilcoxon_mcc_folds1_6.csv"
if stats_path.exists():
    wilcoxon_df = pd.read_csv(stats_path)

    labels_map = {k: MODEL_META[k]["label"] for k in model_order}
    model_labels = [labels_map[k] for k in model_order]
    n = len(model_order)
    pval_matrix = np.full((n, n), np.nan)

    for _, row in wilcoxon_df.iterrows():
        if row["model_A"] in model_order and row["model_B"] in model_order:
            i = model_order.index(row["model_A"])
            j = model_order.index(row["model_B"])
            pval_matrix[i, j] = row["pvalue_bonferroni"]
            pval_matrix[j, i] = row["pvalue_bonferroni"]

    # MCC mean matrix for annotations
    mcc_means = {k: np.mean([m for m in data[k]["fold_mccs"][:6] if m is not None])
                 for k in model_order}

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.eye(n, dtype=bool)
    annot = np.full((n, n), "", dtype=object)
    for i in range(n):
        for j in range(n):
            if i != j and not np.isnan(pval_matrix[i, j]):
                p = pval_matrix[i, j]
                annot[i, j] = f"{p:.3f}" + ("*" if p < 0.05 else "")

    sns.heatmap(pval_matrix, ax=ax, mask=mask, annot=annot, fmt="",
                cmap="RdYlGn_r", vmin=0, vmax=0.2,
                xticklabels=model_labels, yticklabels=model_labels,
                linewidths=0.5,
                cbar_kws={"label": "Bonferroni-adjusted p-value"})
    ax.set_title("Pairwise Wilcoxon Signed-Rank Test (MCC, folds 1–6)\n* = significant at α=0.05",
                 fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    save_fig(fig, "fig_wilcoxon_heatmap")
else:
    log.warning("  Wilcoxon CSV not found — run run_paper_stats.py first")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table (LaTeX-ready)
# ─────────────────────────────────────────────────────────────────────────────
log.info("\n[Table] Generating LaTeX-ready results table")

table_rows = []
for key in model_order:
    s = data[key]["summary_stable"]
    mccs = [m for m in data[key]["fold_mccs"][:6] if m is not None]
    table_rows.append({
        "Model":    MODEL_META[key]["label"],
        "Acc":      f"{s['accuracy_mean']:.3f} ± {s['accuracy_std']:.3f}",
        "MCC":      f"{s['mcc_mean']:.3f} ± {s['mcc_std']:.3f}",
        "AUC":      f"{s['roc_auc_mean']:.3f} ± {s['roc_auc_std']:.3f}",
        "MCC_num":  s["mcc_mean"],
    })

table_df = pd.DataFrame(table_rows).sort_values("MCC_num", ascending=False).drop("MCC_num", axis=1)
table_df.to_csv(PLOTS_DIR / "results_table.csv", index=False)

log.info("\nResults table (folds 1-6):")
log.info(table_df.to_string(index=False))

log.info(f"\nAll figures saved to {PLOTS_DIR}")
log.info("✓ Plots complete.")
