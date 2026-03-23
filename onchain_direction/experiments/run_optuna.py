#!/usr/bin/env python3
"""
run_optuna.py
=============
Bayesian hyperparameter optimisation for XGBoost on G3 (full feature set).
Uses Optuna with TPE sampler, 100 trials, walk-forward CV inner loop.

Results are saved to results/optuna/ and the best model is re-trained on
all available data (all 7 folds) for final evaluation.
"""

import sys, json, logging
from pathlib import Path

import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "onchain_direction"))

from src.data.loader import load_dataset
from src.data.feature_groups import FEATURE_GROUPS
from src.data.preprocessor import FoldPreprocessor
from src.validation.walk_forward import WalkForwardCV
from src.evaluation.metrics import compute_metrics, aggregate_fold_metrics
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("run_optuna")

# ── Load config & data ────────────────────────────────────────────────────────
with open(ROOT / "onchain_direction" / "config.yaml") as f:
    config = yaml.safe_load(f)

df = load_dataset(config)
cv = WalkForwardCV(config)

group   = FEATURE_GROUPS["G3"]
feat_cols = [c for c in group.features if c in df.columns]
seed    = config["project"]["random_seed"]

results_dir = ROOT / "results" / "optuna"
results_dir.mkdir(parents=True, exist_ok=True)

log.info(f"Optuna XGBoost G3 | {len(feat_cols)} features | {len(cv.folds)} folds | 100 trials")


# ── Objective ────────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    import xgboost as xgb
    from sklearn.metrics import matthews_corrcoef

    params = {
        "n_estimators"     : trial.suggest_int("n_estimators", 200, 1000, step=100),
        "max_depth"        : trial.suggest_int("max_depth", 3, 10),
        "learning_rate"    : trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight" : trial.suggest_int("min_child_weight", 1, 10),
        "gamma"            : trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state"     : seed,
        "eval_metric"      : "logloss",
        "verbosity"        : 0,
        "device"           : "cuda",
    }

    mccs = []
    for fold in cv:
        train_df, test_df = fold.split(df)
        prep = FoldPreprocessor(sequence_length=1)
        target_col = config["data"]["target_column"]
        X_tr, _, y_tr, _ = prep.fit_transform(train_df, feat_cols, target_col)
        X_te, _, y_te, _ = prep.transform(test_df, feat_cols, target_col)

        pos = y_tr.sum(); neg = len(y_tr) - pos
        spw = neg / max(pos, 1)

        clf = xgb.XGBClassifier(scale_pos_weight=spw, **params)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        mccs.append(matthews_corrcoef(y_te, preds))

    return float(np.mean(mccs))


# ── Run study ─────────────────────────────────────────────────────────────────
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=seed),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3),
)

log.info("Starting 100-trial Optuna search ...")
study.optimize(objective, n_trials=100, show_progress_bar=True)

best = study.best_trial
log.info(f"\nBest trial: MCC={best.value:.4f}")
log.info(f"Best params: {best.params}")


# ── Final evaluation with best params ─────────────────────────────────────────
import xgboost as xgb
from src.models.xgboost_model import XGBoostModel

log.info("\nFinal evaluation with best params (7 folds) ...")
fold_metrics = []
y_true_all, y_proba_all = [], []

for fold in cv:
    train_df, test_df = fold.split(df)
    prep = FoldPreprocessor(sequence_length=1)
    target_col = config["data"]["target_column"]
    X_tr, _, y_tr, _ = prep.fit_transform(train_df, feat_cols, target_col)
    X_te, _, y_te, _ = prep.transform(test_df, feat_cols, target_col)

    pos = y_tr.sum(); neg = len(y_tr) - pos
    spw = neg / max(pos, 1)

    params = dict(best.params)
    params.update({"scale_pos_weight": spw, "random_state": seed,
                   "eval_metric": "logloss", "verbosity": 0, "device": "cuda"})

    clf = xgb.XGBClassifier(**params)
    clf.fit(X_tr, y_tr)

    preds  = clf.predict(X_te)
    probas = clf.predict_proba(X_te)[:, 1]
    m = compute_metrics(y_te, preds, probas)
    fold_metrics.append(m)
    y_true_all.extend(y_te.tolist())
    y_proba_all.extend(probas.tolist())

    log.info(f"  [{fold.test_label}] Acc={m['accuracy']:.4f} MCC={m['mcc']:.4f} AUC={m['roc_auc']:.4f}")

summary = aggregate_fold_metrics(fold_metrics)
log.info(f"\nRESULT (Optuna XGBoost G3):")
log.info(f"  Acc  = {summary['accuracy']['mean']:.4f} +- {summary['accuracy']['std']:.4f}")
log.info(f"  MCC  = {summary['mcc']['mean']:.4f} +- {summary['mcc']['std']:.4f}")
log.info(f"  AUC  = {summary['roc_auc']['mean']:.4f} +- {summary['roc_auc']['std']:.4f}")

# Save
out = {
    "experiment_id"  : "XGBoost_G3_Optuna",
    "best_params"    : best.params,
    "best_trial_mcc" : best.value,
    "summary"        : {k: {"mean": v["mean"], "std": v["std"]} for k, v in summary.items() if k != "confusion_matrix"},
    "y_true_all"     : y_true_all,
    "y_proba_all"    : y_proba_all,
}
with open(results_dir / "xgboost_g3_optuna.json", "w") as f:
    json.dump(out, f, indent=2)

# Save study
study.trials_dataframe().to_csv(results_dir / "optuna_trials.csv", index=False)
log.info(f"\nResults saved -> {results_dir}")
log.info("Optuna optimisation complete.")
