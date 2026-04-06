#!/usr/bin/env python3
"""
run_paper_shap.py
=================
SHAP feature importance analysis for the paper using XGBoost G3 + Optuna.

Uses SHAP TreeExplainer on the best XGBoost model trained on all stable folds
(folds 1-6 combined as training, fold 6 test set as reference).

Generates:
  - SHAP bar chart (mean |SHAP|) — feature importance ranking
  - SHAP beeswarm plot
  - CSV with SHAP values per feature
"""

import sys, json, logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Elsevier figure style ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":        9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "legend.framealpha": 0.85,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})
COL1 = 3.54   # single column 90 mm
COL2 = 7.48   # double column 190 mm

import shap

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "onchain_direction"))

import yaml
from src.data.loader import load_dataset, impute_missing
from src.data.feature_groups import FEATURE_GROUPS
from src.data.preprocessor import FoldPreprocessor
from src.validation.walk_forward import WalkForwardCV

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("paper_shap")

# ── Config ───────────────────────────────────────────────────────────────────
with open(ROOT / "onchain_direction" / "config.yaml") as f:
    config = yaml.safe_load(f)
config["paths"]["dataset"] = "data/bitcoin_onchain_2013_2025.csv"

df         = impute_missing(load_dataset(config))
cv         = WalkForwardCV(config)
target_col = config["data"]["target_column"]
seed       = config["project"]["random_seed"]
feat_cols  = [c for c in FEATURE_GROUPS["G3"].features if c in df.columns]

RESULTS_DIR  = ROOT / "results" / "optuna"
SHAP_DIR     = ROOT / "results" / "shap"
PLOTS_DIR    = ROOT / "results" / "plots"
PAPER_FIGS   = ROOT / "paper" / "EAAI" / "figures"
SHAP_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

# ── Load best XGBoost params ──────────────────────────────────────────────────
xgb_json    = json.load(open(RESULTS_DIR / "xgboost_g3_optuna_final.json"))
best_params = dict(xgb_json["best_params"])
best_params.update({"random_state": seed, "eval_metric": "logloss",
                    "verbosity": 0, "device": "cuda"})

# ── Train on each fold, collect SHAP values ───────────────────────────────────
import xgboost as xgb

all_shap_values = []
all_X_test      = []

folds_stable = list(cv)[:-1]   # folds 1-6

log.info(f"Computing SHAP values across {len(folds_stable)} stable folds...")

for fold in folds_stable:
    train_df, test_df = fold.split(df)
    prep = FoldPreprocessor(sequence_length=1)
    X_tr, _, y_tr, _ = prep.fit_transform(train_df, feat_cols, target_col)
    X_te, _, y_te, _ = prep.transform(test_df, feat_cols, target_col)

    spw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    params = dict(best_params)
    params["scale_pos_weight"] = spw

    clf = xgb.XGBClassifier(**params)
    clf.fit(X_tr, y_tr)

    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_te)   # shape: (n_test, n_features)

    all_shap_values.append(shap_values)
    all_X_test.append(X_te)
    log.info(f"  [{fold.label}] SHAP computed — test samples: {len(X_te)}")

# Concatenate across folds
shap_matrix = np.vstack(all_shap_values)   # (total_test_samples, n_features)
X_all       = np.vstack(all_X_test)

log.info(f"\nTotal SHAP matrix: {shap_matrix.shape}")

# ── Feature importance (mean |SHAP|) ─────────────────────────────────────────
mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
importance_df = pd.DataFrame({
    "feature":        feat_cols,
    "mean_abs_shap":  mean_abs_shap,
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

importance_df.to_csv(SHAP_DIR / "xgboost_shap_importance.csv", index=False)
log.info(f"\nTop 10 features by SHAP:")
for _, row in importance_df.head(10).iterrows():
    log.info(f"  {row['feature']:40s}  {row['mean_abs_shap']:.4f}")

# ── Plot 1: Bar chart (top 10 features) ──────────────────────────────────────
top_n = 10
top_df = importance_df.head(top_n)

# Color by domain (technical vs on-chain)
from src.data.feature_groups import get_dual_encoder_splits
tech_cols, onchain_cols = get_dual_encoder_splits(df)

colors = []
for feat in top_df["feature"]:
    if feat in tech_cols:
        colors.append("#7EB6D4")   # muted blue = technical
    else:
        colors.append("#E8A87C")   # muted orange = on-chain

fig, ax = plt.subplots(figsize=(COL1 * 0.8, top_n * 0.22 + 0.6))
bars = ax.barh(range(top_n), top_df["mean_abs_shap"].values[::-1],
               color=colors[::-1], edgecolor="white", linewidth=0.5)
# Shorten feature names: underscores → spaces, max 22 chars
def shorten(name, maxlen=22):
    s = name.replace("_", " ")
    return s if len(s) <= maxlen else s[:maxlen-1] + "…"

short_labels = [shorten(f) for f in top_df["feature"].values[::-1]]
ax.set_yticks(range(top_n))
ax.set_yticklabels(short_labels, fontsize=7)
ax.set_xlabel("Mean |SHAP value|", fontsize=8)
ax.tick_params(axis="x", labelsize=7)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#E8A87C", label="On-chain"),
    Patch(facecolor="#7EB6D4", label="Technical"),
]
ax.legend(handles=legend_elements, loc="lower right")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "shap_bar_top20.pdf", bbox_inches="tight")
fig.savefig(PLOTS_DIR / "shap_bar_top20.png", dpi=300, bbox_inches="tight")
import shutil; shutil.copy(PLOTS_DIR / "shap_bar_top20.pdf", PAPER_FIGS / "shap_importance_bar.pdf")
plt.close(fig)
log.info(f"Saved: shap_bar_top20.pdf/.png → paper/EAAI/figures/shap_importance_bar.pdf")

# ── Plot 2: Beeswarm (SHAP summary plot) ─────────────────────────────────────
# Use only top-20 features for readability
top_idx = [feat_cols.index(f) for f in top_df["feature"]]
shap_top = shap_matrix[:, top_idx]
X_top    = X_all[:, top_idx]
feat_top = top_df["feature"].tolist()

fig, ax = plt.subplots(figsize=(COL2, 4.2))
shap.summary_plot(
    shap_top, X_top,
    feature_names=feat_top,
    show=False, plot_size=None,
    max_display=top_n,
)
plt.tight_layout()
fig = plt.gcf()
fig.savefig(PLOTS_DIR / "shap_beeswarm_top20.pdf", bbox_inches="tight")
fig.savefig(PLOTS_DIR / "shap_beeswarm_top20.png", dpi=300, bbox_inches="tight")
plt.close("all")
log.info(f"Saved: shap_beeswarm_top20.pdf/.png")

# ── Save full SHAP matrix ─────────────────────────────────────────────────────
np.save(SHAP_DIR / "xgboost_shap_matrix.npy", shap_matrix)
np.save(SHAP_DIR / "xgboost_X_test.npy",      X_all)

log.info(f"\nSHAP results saved to {SHAP_DIR}")
log.info("✓ SHAP analysis complete.")
