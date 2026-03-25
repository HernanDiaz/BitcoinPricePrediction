#!/usr/bin/env python3
"""
run_paper_stats.py
==================
Statistical significance tests for the paper.

Tests performed:
  1. Wilcoxon signed-rank test (pairwise, MCC per fold) + Bonferroni correction
  2. McNemar test on concatenated test-set predictions

Models compared (folds 1-6, stable period):
  - XGBoost_G3_Optuna
  - LightGBM_G3_Optuna
  - MLPSimple_Optuna
  - MLPDualNoAttn_Optuna
  - MLPDualV2_Optuna

For XGBoost and LightGBM the per-fold MCC is recomputed from best_params
(the JSON files omitted fold-level values at save time).
"""

import sys, json, logging
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "onchain_direction"))

import yaml
from src.data.loader import load_dataset, impute_missing
from src.data.feature_groups import FEATURE_GROUPS
from src.data.preprocessor import FoldPreprocessor
from src.validation.walk_forward import WalkForwardCV
from src.evaluation.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("paper_stats")

# ── Config ───────────────────────────────────────────────────────────────────
with open(ROOT / "onchain_direction" / "config.yaml") as f:
    config = yaml.safe_load(f)
config["paths"]["dataset"] = "dataset_COMPLETO_con_OHLCV_20251221_014211.csv"

df         = impute_missing(load_dataset(config))
cv         = WalkForwardCV(config)
target_col = config["data"]["target_column"]
seed       = config["project"]["random_seed"]
feat_cols  = [c for c in FEATURE_GROUPS["G3"].features if c in df.columns]

RESULTS_DIR = ROOT / "results" / "optuna"
OUT_DIR     = ROOT / "results" / "statistical_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STABLE_FOLDS = list(cv)[:-1]   # folds 1-6 (exclude 2025)
ALL_FOLDS    = list(cv)         # all 7 folds


# ── Helper: run 7-fold eval for tree models ───────────────────────────────────
def eval_xgboost(best_params: dict):
    import xgboost as xgb
    fold_mccs, y_true_all, y_pred_all, y_proba_all = [], [], [], []
    for fold in ALL_FOLDS:
        train_df, test_df = fold.split(df)
        prep = FoldPreprocessor(sequence_length=1)
        X_tr, _, y_tr, _ = prep.fit_transform(train_df, feat_cols, target_col)
        X_te, _, y_te, _ = prep.transform(test_df, feat_cols, target_col)
        spw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
        params = dict(best_params)
        params.update({"scale_pos_weight": spw, "random_state": seed,
                       "eval_metric": "logloss", "verbosity": 0, "device": "cuda"})
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_tr, y_tr)
        preds  = clf.predict(X_te)
        probas = clf.predict_proba(X_te)[:, 1]
        m = compute_metrics(y_te, preds, probas)
        fold_mccs.append(m["mcc"])
        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(preds.tolist())
        y_proba_all.extend(probas.tolist())
        log.info(f"  [{fold.label}] Acc={m['accuracy']:.4f} MCC={m['mcc']:.4f}")
    return fold_mccs, np.array(y_true_all), np.array(y_pred_all), np.array(y_proba_all)


def eval_lightgbm(best_params: dict):
    import lightgbm as lgb
    fold_mccs, y_true_all, y_pred_all, y_proba_all = [], [], [], []
    for fold in ALL_FOLDS:
        train_df, test_df = fold.split(df)
        prep = FoldPreprocessor(sequence_length=1)
        X_tr, _, y_tr, _ = prep.fit_transform(train_df, feat_cols, target_col)
        X_te, _, y_te, _ = prep.transform(test_df, feat_cols, target_col)
        pos = y_tr.sum(); neg = len(y_tr) - pos
        params = dict(best_params)
        params.update({"scale_pos_weight": neg / max(pos, 1), "random_state": seed,
                       "verbosity": -1, "n_jobs": -1})
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_tr, y_tr)
        preds  = clf.predict(X_te)
        probas = clf.predict_proba(X_te)[:, 1]
        m = compute_metrics(y_te, preds, probas)
        fold_mccs.append(m["mcc"])
        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(preds.tolist())
        y_proba_all.extend(probas.tolist())
        log.info(f"  [{fold.label}] Acc={m['accuracy']:.4f} MCC={m['mcc']:.4f}")
    return fold_mccs, np.array(y_true_all), np.array(y_pred_all), np.array(y_proba_all)


# ── Load all models ───────────────────────────────────────────────────────────
models = {}

# XGBoost
log.info("\n[XGBoost] Re-evaluating 7 folds with best params...")
xgb_json = json.load(open(RESULTS_DIR / "xgboost_g3_optuna_final.json"))
mccs, yt, yp, ypr = eval_xgboost(xgb_json["best_params"])
models["XGBoost"] = {"fold_mccs": mccs, "y_true": yt, "y_pred": yp, "y_proba": ypr}

# LightGBM
log.info("\n[LightGBM] Re-evaluating 7 folds with best params...")
lgbm_json = json.load(open(RESULTS_DIR / "lightgbm_g3_optuna_final.json"))
mccs, yt, yp, ypr = eval_lightgbm(lgbm_json["best_params"])
models["LightGBM"] = {"fold_mccs": mccs, "y_true": yt, "y_pred": yp, "y_proba": ypr}

# MLP models (fold_metrics already saved)
for fname, label in [
    ("mlp_simple_optuna.json",      "MLP_Simple"),
    ("mlp_dual_noattn_optuna.json", "MLP_Dual_NoAttn"),
    ("mlp_dual_v2_optuna.json",     "MLP_Dual_V2"),
]:
    d = json.load(open(RESULTS_DIR / fname))
    fold_mccs = [fm["mcc"] for fm in d["fold_metrics"]]
    y_true  = np.array(d["y_true_all"])
    y_proba = np.array(d["y_proba_all"])
    y_pred  = (y_proba >= 0.5).astype(int)
    models[label] = {"fold_mccs": fold_mccs, "y_true": y_true, "y_pred": y_pred, "y_proba": y_proba}
    log.info(f"Loaded {label}: {len(fold_mccs)} folds, mean MCC (1-6)={np.mean(fold_mccs[:6]):.4f}")


# ── 1. Wilcoxon signed-rank test (folds 1-6) ────────────────────────────────
log.info("\n" + "="*60)
log.info("WILCOXON SIGNED-RANK TEST (folds 1-6, MCC)")
log.info("="*60)

model_names = list(models.keys())
n_pairs     = len(model_names) * (len(model_names) - 1) // 2
alpha       = 0.05
alpha_bonf  = alpha / n_pairs

wilcoxon_results = []
for a, b in combinations(model_names, 2):
    mccs_a = np.array(models[a]["fold_mccs"][:6])
    mccs_b = np.array(models[b]["fold_mccs"][:6])
    diff = mccs_a - mccs_b
    if np.all(diff == 0):
        stat, pval = np.nan, 1.0
    else:
        stat, pval = wilcoxon(mccs_a, mccs_b, alternative="two-sided")
    pval_adj  = min(pval * n_pairs, 1.0)
    sig       = pval_adj < alpha
    mean_a    = mccs_a.mean()
    mean_b    = mccs_b.mean()
    wilcoxon_results.append({
        "model_A": a, "model_B": b,
        "mean_mcc_A": round(mean_a, 4), "mean_mcc_B": round(mean_b, 4),
        "delta_mcc": round(mean_a - mean_b, 4),
        "statistic": round(float(stat), 4) if not np.isnan(stat) else None,
        "pvalue": round(float(pval), 4),
        "pvalue_bonferroni": round(float(pval_adj), 4),
        "significant": bool(sig),
    })
    marker = "*** SIGNIFICANT" if sig else ""
    log.info(f"  {a} vs {b}: Δ={mean_a-mean_b:+.4f}  p={pval:.4f}  p_adj={pval_adj:.4f}  {marker}")

wilcoxon_df = pd.DataFrame(wilcoxon_results)
wilcoxon_df.to_csv(OUT_DIR / "wilcoxon_mcc_folds1_6.csv", index=False)
log.info(f"\nBonferroni-corrected α = {alpha_bonf:.4f} ({n_pairs} pairs)")
log.info(f"Significant pairs: {wilcoxon_df['significant'].sum()}/{len(wilcoxon_df)}")


# ── 2. McNemar test (all 7 folds concatenated) ──────────────────────────────
log.info("\n" + "="*60)
log.info("McNEMAR TEST (all 7 folds concatenated)")
log.info("="*60)

mcnemar_results = []
for a, b in combinations(model_names, 2):
    yt_a = models[a]["y_true"]
    yp_a = models[a]["y_pred"]
    yt_b = models[b]["y_true"]
    yp_b = models[b]["y_pred"]

    if not np.array_equal(yt_a, yt_b):
        log.warning(f"  Skipping {a} vs {b}: y_true arrays differ")
        continue

    correct_a = (yp_a == yt_a)
    correct_b = (yp_b == yt_b)

    n00 = int(((~correct_a) & (~correct_b)).sum())   # both wrong
    n01 = int(((~correct_a) & correct_b).sum())      # A wrong, B right
    n10 = int((correct_a & (~correct_b)).sum())      # A right, B wrong
    n11 = int((correct_a & correct_b).sum())         # both right

    table = np.array([[n00, n01], [n10, n11]])
    result = mcnemar(table, exact=False, correction=True)
    pval     = float(result.pvalue)
    pval_adj = min(pval * n_pairs, 1.0)
    sig      = pval_adj < alpha

    mcnemar_results.append({
        "model_A": a, "model_B": b,
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "statistic": round(float(result.statistic), 4),
        "pvalue": round(pval, 4),
        "pvalue_bonferroni": round(pval_adj, 4),
        "significant": bool(sig),
    })
    marker = "*** SIGNIFICANT" if sig else ""
    log.info(f"  {a} vs {b}: χ²={result.statistic:.2f}  p={pval:.4f}  p_adj={pval_adj:.4f}  {marker}")

mcnemar_df = pd.DataFrame(mcnemar_results)
mcnemar_df.to_csv(OUT_DIR / "mcnemar_results.csv", index=False)


# ── 3. Summary table ─────────────────────────────────────────────────────────
log.info("\n" + "="*60)
log.info("MODEL PERFORMANCE SUMMARY (folds 1-6)")
log.info("="*60)

summary_rows = []
for name, data in models.items():
    mccs = np.array(data["fold_mccs"][:6])
    summary_rows.append({
        "Model": name,
        "MCC_mean": round(mccs.mean(), 4),
        "MCC_std":  round(mccs.std(), 4),
        "MCC_values": [round(v, 4) for v in mccs.tolist()],
    })
    log.info(f"  {name}: MCC = {mccs.mean():.4f} ± {mccs.std():.4f}")

summary_df = pd.DataFrame(summary_rows).sort_values("MCC_mean", ascending=False)
summary_df.to_csv(OUT_DIR / "model_summary_stats.csv", index=False)

# Save full results as JSON
all_out = {
    "wilcoxon": wilcoxon_results,
    "mcnemar": mcnemar_results,
    "model_summary": summary_rows,
    "n_bonferroni_pairs": n_pairs,
    "alpha": alpha,
    "alpha_bonferroni": alpha_bonf,
}
with open(OUT_DIR / "statistical_tests_full.json", "w") as f:
    json.dump(all_out, f, indent=2)

log.info(f"\nResultados guardados en {OUT_DIR}")
log.info("✓ Statistical analysis complete.")
