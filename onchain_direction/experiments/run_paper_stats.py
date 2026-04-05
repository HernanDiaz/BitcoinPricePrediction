#!/usr/bin/env python3
"""
run_paper_stats.py
==================
Statistical significance tests for the paper (folds 1-6 only, 2019-2024).

Tests performed:
  1. Wilcoxon signed-rank test (pairwise, MCC per fold) + Bonferroni correction
  2. McNemar test on folds 1-6 concatenated predictions

Models compared:
  - XGBoost_G3_Optuna
  - LightGBM_G3_Optuna
  - SVM_G3_Optuna
  - CNN_LSTM_G3_Optuna
  - MLP_Simple_Optuna
  - MLP_Dual_NoAttn_Optuna
  - MLP_Dual_V2_Optuna  (DECA)

Note: CNN-LSTM uses seq_len=5, dropping seq_len-1=4 samples per fold.
McNemar pairs involving CNN-LSTM are skipped if y_true arrays do not align.
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

# ── Config ────────────────────────────────────────────────────────────────────
with open(ROOT / "onchain_direction" / "config.yaml") as f:
    config = yaml.safe_load(f)
config["paths"]["dataset"] = "dataset_COMPLETO_con_OHLCV_20251221_014211.csv"

df         = impute_missing(load_dataset(config))
cv         = WalkForwardCV(config)
target_col = config["data"]["target_column"]
seed       = config["project"]["random_seed"]
feat_cols  = [c for c in FEATURE_GROUPS["G3"].features if c in df.columns]

RESULTS_DIR  = ROOT / "results" / "optuna"
OUT_DIR      = ROOT / "results" / "statistical_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_FOLDS    = list(cv)
STABLE_FOLDS = ALL_FOLDS[:-1]   # folds 1-6 (excluye 2025)

# Calcular tamaño exacto del fold 7 (2025) para slicing de arrays JSON
_, fold7_test = ALL_FOLDS[-1].split(df)
N_FOLD7 = len(fold7_test.dropna(subset=[target_col]))
log.info(f"Fold 7 (2025) test size: {N_FOLD7} samples — will be excluded from all analyses")


# ── Helpers: re-evaluar modelos sobre folds 1-6 ───────────────────────────────
def eval_xgboost(best_params: dict):
    import xgboost as xgb
    fold_mccs, y_true_all, y_pred_all, y_proba_all = [], [], [], []
    for fold in STABLE_FOLDS:
        train_df, test_df = fold.split(df)
        prep = FoldPreprocessor(sequence_length=1)
        X_tr, _, y_tr, _ = prep.fit_transform(train_df, feat_cols, target_col)
        X_te, _, y_te, _ = prep.transform(test_df, feat_cols, target_col)
        spw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
        params = dict(best_params)
        params.update({"scale_pos_weight": spw, "random_state": seed,
                       "eval_metric": "logloss", "verbosity": 0})
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
    for fold in STABLE_FOLDS:
        train_df, test_df = fold.split(df)
        prep = FoldPreprocessor(sequence_length=1)
        X_tr, _, y_tr, _ = prep.fit_transform(train_df, feat_cols, target_col)
        X_te, _, y_te, _ = prep.transform(test_df, feat_cols, target_col)
        pos = y_tr.sum(); neg = len(y_tr) - pos
        params = dict(best_params)
        params.update({"scale_pos_weight": neg / max(pos, 1),
                       "random_state": seed, "verbosity": -1, "n_jobs": -1})
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


def eval_svm(best_params: dict):
    """SVM re-evaluado sobre folds 1-6 con StandardScaler (requerimiento del modelo)."""
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    fold_mccs, y_true_all, y_pred_all, y_proba_all = [], [], [], []
    for fold in STABLE_FOLDS:
        train_df, test_df = fold.split(df)
        prep = FoldPreprocessor(sequence_length=1)
        X_tr, _, y_tr, _ = prep.fit_transform(train_df, feat_cols, target_col)
        X_te, _, y_te, _ = prep.transform(test_df, feat_cols, target_col)
        # StandardScaler adicional requerido por SVM
        ss = StandardScaler()
        X_tr = ss.fit_transform(X_tr)
        X_te = ss.transform(X_te)
        params = dict(best_params)
        params.update({"random_state": seed, "probability": True,
                       "class_weight": "balanced", "max_iter": 10000})
        clf = SVC(**params)
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


def load_from_json(json_path: Path, n_fold7: int):
    """Carga fold_metrics + predicciones de JSON, excluyendo fold 7."""
    d = json.load(open(json_path))
    fold_mccs  = [fm["mcc"] for fm in d["fold_metrics"][:6]]
    y_true_all = np.array(d["y_true_all"])
    y_proba    = np.array(d["y_proba_all"])
    # Excluir fold 7 (ultimos n_fold7 muestras)
    if n_fold7 > 0 and len(y_true_all) > n_fold7:
        y_true_all = y_true_all[:-n_fold7]
        y_proba    = y_proba[:-n_fold7]
    y_pred = (y_proba >= 0.5).astype(int)
    return fold_mccs, y_true_all, y_pred, y_proba


# ── Cargar todos los modelos ──────────────────────────────────────────────────
models = {}

log.info("\n[XGBoost] Re-evaluando folds 1-6 con best params...")
xgb_json = json.load(open(RESULTS_DIR / "xgboost_g3_optuna_final.json"))
mccs, yt, yp, ypr = eval_xgboost(xgb_json["best_params"])
models["XGBoost"] = {"fold_mccs": mccs, "y_true": yt, "y_pred": yp, "y_proba": ypr}

log.info("\n[LightGBM] Re-evaluando folds 1-6 con best params...")
lgbm_json = json.load(open(RESULTS_DIR / "lightgbm_g3_optuna_final.json"))
mccs, yt, yp, ypr = eval_lightgbm(lgbm_json["best_params"])
models["LightGBM"] = {"fold_mccs": mccs, "y_true": yt, "y_pred": yp, "y_proba": ypr}

log.info("\n[SVM] Re-evaluando folds 1-6 con best params...")
svm_json = json.load(open(RESULTS_DIR / "svm_g3_optuna_final.json"))
# Extraer solo los params del clasificador (sin seq_len, etc.)
svm_clf_params = {k: v for k, v in svm_json["best_params"].items()
                  if k in ["C", "gamma", "kernel", "degree", "coef0"]}
mccs, yt, yp, ypr = eval_svm(svm_clf_params)
models["SVM"] = {"fold_mccs": mccs, "y_true": yt, "y_pred": yp, "y_proba": ypr}

# MLP models desde JSON (seq_len=1, mismas muestras que XGB/LGB/SVM)
for fname, label in [
    ("mlp_simple_optuna.json",        "MLP_Simple"),
    ("mlp_dual_noattn_optuna.json",   "MLP_Dual_NoAttn"),
    ("mlp_dual_v2_optuna.json",       "DECA"),
]:
    mccs, yt, yp, ypr = load_from_json(RESULTS_DIR / fname, N_FOLD7)
    models[label] = {"fold_mccs": mccs, "y_true": yt, "y_pred": yp, "y_proba": ypr}
    log.info(f"Loaded {label}: {len(mccs)} folds, MCC_mean(1-6)={np.mean(mccs):.4f}, "
             f"n_samples={len(yt)}")

# CNN-LSTM desde JSON (seq_len=5: 4 muestras menos por fold = ~24 menos en total)
log.info("\n[CNN-LSTM] Cargando desde JSON (seq_len=5)...")
cnn_json = json.load(open(RESULTS_DIR / "cnn_lstm_g3_optuna_final.json"))
cnn_fold_mccs  = [fm["mcc"] for fm in cnn_json["fold_metrics"][:6]]
cnn_y_true_all = np.array(cnn_json["y_true_all"])
cnn_y_proba    = np.array(cnn_json["y_proba_all"])
# El JSON incluye fold 7 al final
cnn_n_fold7    = len(cnn_json["fold_metrics"][-1].get("y_true", [])) if False else N_FOLD7 - 4
# Aproximar: fold 7 de CNN tiene (N_FOLD7 - seq_len + 1) muestras
cnn_seq_len    = cnn_json.get("best_seq_len", 5)
cnn_n_fold7_adj = max(N_FOLD7 - cnn_seq_len + 1, 0)
if cnn_n_fold7_adj > 0 and len(cnn_y_true_all) > cnn_n_fold7_adj:
    cnn_y_true_all = cnn_y_true_all[:-cnn_n_fold7_adj]
    cnn_y_proba    = cnn_y_proba[:-cnn_n_fold7_adj]
cnn_y_pred = (cnn_y_proba >= 0.5).astype(int)
models["CNN_LSTM"] = {
    "fold_mccs": cnn_fold_mccs,
    "y_true": cnn_y_true_all,
    "y_pred": cnn_y_pred,
    "y_proba": cnn_y_proba,
}
log.info(f"Loaded CNN_LSTM: {len(cnn_fold_mccs)} folds, MCC_mean(1-6)={np.mean(cnn_fold_mccs):.4f}, "
         f"n_samples={len(cnn_y_true_all)} (seq_len={cnn_seq_len})")


# ── 1. Wilcoxon signed-rank test (folds 1-6) ─────────────────────────────────
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
    pval_adj = min(pval * n_pairs, 1.0)
    sig      = pval_adj < alpha
    mean_a   = mccs_a.mean()
    mean_b   = mccs_b.mean()
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


# ── 2. McNemar test (folds 1-6 concatenados) ─────────────────────────────────
log.info("\n" + "="*60)
log.info("McNEMAR TEST (folds 1-6 concatenados)")
log.info("="*60)

mcnemar_results = []
for a, b in combinations(model_names, 2):
    yt_a = models[a]["y_true"]
    yp_a = models[a]["y_pred"]
    yt_b = models[b]["y_true"]
    yp_b = models[b]["y_pred"]

    if len(yt_a) != len(yt_b) or not np.array_equal(yt_a, yt_b):
        log.warning(f"  Skipping {a} vs {b}: y_true no alineados "
                    f"(n_a={len(yt_a)}, n_b={len(yt_b)}) — "
                    f"CNN-LSTM usa seq_len={cnn_seq_len}")
        continue

    correct_a = (yp_a == yt_a)
    correct_b = (yp_b == yt_b)
    n00 = int(((~correct_a) & (~correct_b)).sum())
    n01 = int(((~correct_a) &   correct_b ).sum())
    n10 = int((  correct_a  & (~correct_b)).sum())
    n11 = int((  correct_a  &   correct_b ).sum())

    table  = np.array([[n00, n01], [n10, n11]])
    result = mcnemar(table, exact=False, correction=True)
    pval     = float(result.pvalue)
    pval_adj = min(pval * n_pairs, 1.0)
    sig      = pval_adj < alpha

    mcnemar_results.append({
        "model_A": a, "model_B": b,
        "n_samples": int(len(yt_a)),
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "statistic": round(float(result.statistic), 4),
        "pvalue": round(pval, 6),
        "pvalue_bonferroni": round(pval_adj, 6),
        "significant": bool(sig),
    })
    marker = "*** SIGNIFICANT" if sig else ""
    log.info(f"  {a} vs {b}: χ²={result.statistic:.2f}  p={pval:.4f}  "
             f"p_adj={pval_adj:.4f}  {marker}")

mcnemar_df = pd.DataFrame(mcnemar_results)
mcnemar_df.to_csv(OUT_DIR / "mcnemar_results.csv", index=False)


# ── 3. Tabla resumen ──────────────────────────────────────────────────────────
log.info("\n" + "="*60)
log.info("RESUMEN DE RENDIMIENTO (folds 1-6)")
log.info("="*60)

summary_rows = []
for name, data in models.items():
    mccs = np.array(data["fold_mccs"][:6])
    summary_rows.append({
        "Model":      name,
        "MCC_mean":   round(mccs.mean(), 4),
        "MCC_std":    round(mccs.std(), 4),
        "MCC_values": [round(v, 4) for v in mccs.tolist()],
    })
    log.info(f"  {name}: MCC = {mccs.mean():.4f} ± {mccs.std():.4f}")

summary_df = pd.DataFrame(summary_rows).sort_values("MCC_mean", ascending=False)
summary_df.to_csv(OUT_DIR / "model_summary_stats.csv", index=False)

all_out = {
    "wilcoxon":             wilcoxon_results,
    "mcnemar":              mcnemar_results,
    "model_summary":        summary_rows,
    "n_bonferroni_pairs":   n_pairs,
    "alpha":                alpha,
    "alpha_bonferroni":     alpha_bonf,
    "folds_used":           "1-6 (2019-2024)",
    "n_fold7_excluded":     int(N_FOLD7),
}
with open(OUT_DIR / "statistical_tests_full.json", "w") as f:
    json.dump(all_out, f, indent=2)

log.info(f"\nResultados guardados en {OUT_DIR}")
log.info("✓ Statistical analysis complete.")
