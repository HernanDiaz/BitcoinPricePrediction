#!/usr/bin/env python3
"""
run_optuna_lgbm.py
==================
LightGBM exhaustive Optuna optimisation — 300 trials, hybrid TPE -> CMA-ES.
Runs in parallel with run_optuna_xgb.py.
"""

import sys, json, logging
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "onchain_direction"))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.data.loader import load_dataset
from src.data.feature_groups import FEATURE_GROUPS
from src.data.preprocessor import FoldPreprocessor
from src.validation.walk_forward import WalkForwardCV
from src.evaluation.metrics import compute_metrics, aggregate_fold_metrics
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("optuna_lgbm")

with open(ROOT / "onchain_direction" / "config.yaml") as f:
    config = yaml.safe_load(f)

df        = load_dataset(config)
cv        = WalkForwardCV(config)
target_col = config["data"]["target_column"]
seed      = config["project"]["random_seed"]
group     = FEATURE_GROUPS["G3"]
feat_cols  = [c for c in group.features if c in df.columns]

search_folds = list(cv)[:-1]   # Folds 1-6 para búsqueda

results_dir = ROOT / "results" / "optuna"
results_dir.mkdir(parents=True, exist_ok=True)

log.info(f"LightGBM Optuna exhaustivo | {len(feat_cols)} features | {len(search_folds)} folds (2019-2024) | 300 trials")


def objective(trial: optuna.Trial) -> float:
    import lightgbm as lgb
    from sklearn.metrics import matthews_corrcoef

    params = {
        "n_estimators"      : trial.suggest_int("n_estimators", 300, 2000, step=100),
        "max_depth"         : trial.suggest_int("max_depth", 3, 12),
        "learning_rate"     : trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "num_leaves"        : trial.suggest_int("num_leaves", 15, 255),
        "min_child_samples" : trial.suggest_int("min_child_samples", 5, 100),
        "subsample"         : trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree"  : trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha"         : trial.suggest_float("reg_alpha", 1e-8, 20.0, log=True),
        "reg_lambda"        : trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),
        "min_split_gain"    : trial.suggest_float("min_split_gain", 0.0, 5.0),
        "random_state"      : seed,
        "verbosity"         : -1,
        "device"            : "gpu",
    }

    mccs = []
    for fold in search_folds:
        train_df, test_df = fold.split(df)
        prep = FoldPreprocessor(sequence_length=1)
        X_tr, _, y_tr, _ = prep.fit_transform(train_df, feat_cols, target_col)
        X_te, _, y_te, _ = prep.transform(test_df, feat_cols, target_col)
        pos = y_tr.sum(); neg = len(y_tr) - pos
        clf = lgb.LGBMClassifier(is_unbalance=False,
                                  scale_pos_weight=neg/max(pos,1), **params)
        clf.fit(X_tr, y_tr)
        mccs.append(matthews_corrcoef(y_te, clf.predict(X_te)))

    return float(np.mean(mccs))


sampler = optuna.samplers.TPESampler(seed=seed)
study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
    pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=3),
)

log.info("Fase 1: TPE (50 trials) ...")
study.optimize(objective, n_trials=50, show_progress_bar=True)
log.info(f"  Mejor MCC tras TPE: {study.best_value:.4f}")

log.info("Fase 2: CMA-ES (250 trials) ...")
study.sampler = optuna.samplers.CmaEsSampler(
    seed=seed,
    x0=study.best_params,
    sigma0=0.3,
)
study.optimize(objective, n_trials=250, show_progress_bar=True)

best = study.best_trial
log.info(f"\nMejor trial: MCC={best.value:.4f}")
log.info(f"Mejores params: {best.params}")

# Evaluacion final 7 folds
log.info("\nEvaluacion final (7 folds) ...")
import lightgbm as lgb

fold_metrics, y_true_all, y_proba_all = [], [], []
for fold in cv:
    train_df, test_df = fold.split(df)
    prep = FoldPreprocessor(sequence_length=1)
    X_tr, _, y_tr, _ = prep.fit_transform(train_df, feat_cols, target_col)
    X_te, _, y_te, _ = prep.transform(test_df, feat_cols, target_col)
    pos = y_tr.sum(); neg = len(y_tr) - pos
    params = dict(best.params)
    params.update({"scale_pos_weight": neg/max(pos,1), "random_state": seed,
                   "verbosity": -1, "device": "gpu"})
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_tr, y_tr)
    preds  = clf.predict(X_te)
    probas = clf.predict_proba(X_te)[:, 1]
    m = compute_metrics(y_te, preds, probas)
    fold_metrics.append(m)
    y_true_all.extend(y_te.tolist())
    y_proba_all.extend(probas.tolist())
    log.info(f"  [{fold.label}] Acc={m['accuracy']:.4f} MCC={m['mcc']:.4f} AUC={m['roc_auc']:.4f}")

summary        = aggregate_fold_metrics(fold_metrics)
summary_stable = aggregate_fold_metrics(fold_metrics[:-1])
log.info(f"\nRESULTADO LightGBM_G3_Optuna (7 folds):")
log.info(f"  Acc={summary['accuracy_mean']:.4f}+-{summary['accuracy_std']:.4f}  MCC={summary['mcc_mean']:.4f}+-{summary['mcc_std']:.4f}  AUC={summary['roc_auc_mean']:.4f}+-{summary['roc_auc_std']:.4f}")
log.info(f"RESULTADO (folds 1-6, 2019-2024):")
log.info(f"  Acc={summary_stable['accuracy_mean']:.4f}+-{summary_stable['accuracy_std']:.4f}  MCC={summary_stable['mcc_mean']:.4f}+-{summary_stable['mcc_std']:.4f}  AUC={summary_stable['roc_auc_mean']:.4f}+-{summary_stable['roc_auc_std']:.4f}")

out = {
    "experiment_id"  : "LightGBM_G3_Optuna",
    "n_trials"       : 300,
    "sampler"        : "TPE(50)+CmaES(250)",
    "search_folds"   : "Fold1-Fold6 (2019-2024)",
    "best_trial_mcc" : best.value,
    "best_params"    : best.params,
    "summary_7folds" : summary,
    "summary_stable" : summary_stable,
    "y_true_all"     : y_true_all,
    "y_proba_all"    : y_proba_all,
    "fold_metrics"   : [{k:v for k,v in m.items() if k != "confusion_matrix"} for m in fold_metrics],
}
with open(results_dir / "lightgbm_g3_optuna.json", "w") as f:
    json.dump(out, f, indent=2)

study.trials_dataframe().to_csv(results_dir / "optuna_lgbm_300trials.csv", index=False)
log.info(f"\nGuardado en {results_dir}")
