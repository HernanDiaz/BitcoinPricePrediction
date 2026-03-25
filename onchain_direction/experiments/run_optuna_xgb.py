#!/usr/bin/env python3
"""
run_optuna_xgb.py — Optuna para XGBoost G3.
Configuracion identica a run_optuna_lgbm.py y run_optuna_mlp.py:
  50 trials, TPE sampler, folds 1-6 para busqueda, dataset original 33 features.
"""
import sys, json, logging
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "onchain_direction"))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.data.loader import load_dataset, impute_missing
from src.data.feature_groups import FEATURE_GROUPS
from src.data.preprocessor import FoldPreprocessor
from src.validation.walk_forward import WalkForwardCV
from src.evaluation.metrics import compute_metrics, aggregate_fold_metrics
import yaml, xgboost as xgb
from sklearn.metrics import matthews_corrcoef

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("optuna_xgb")

with open(ROOT / "onchain_direction" / "config.yaml") as f:
    config = yaml.safe_load(f)
config["paths"]["dataset"] = "dataset_COMPLETO_con_OHLCV_20251221_014211.csv"

df         = impute_missing(load_dataset(config))
cv         = WalkForwardCV(config)
target_col = config["data"]["target_column"]
seed       = config["project"]["random_seed"]
group      = FEATURE_GROUPS["G3"]
feat_cols  = [c for c in group.features if c in df.columns]
search_folds = list(cv)[:-1]
results_dir  = ROOT / "results" / "optuna"
results_dir.mkdir(parents=True, exist_ok=True)

log.info(f"XGBoost Optuna | {len(feat_cols)} features | {len(search_folds)} search folds | 50 trials")

def objective(trial):
    params = {
        "n_estimators"     : trial.suggest_int("n_estimators", 200, 2000, step=100),
        "max_depth"        : trial.suggest_int("max_depth", 3, 12),
        "learning_rate"    : trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight" : trial.suggest_int("min_child_weight", 1, 10),
        "gamma"            : trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": seed, "eval_metric": "logloss", "verbosity": 0, "device": "cuda",
    }
    mccs = []
    for fold in search_folds:
        train_df, test_df = fold.split(df)
        prep = FoldPreprocessor(sequence_length=1)
        X_tr, _, y_tr, _ = prep.fit_transform(train_df, feat_cols, target_col)
        X_te, _, y_te, _ = prep.transform(test_df, feat_cols, target_col)
        spw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
        clf = xgb.XGBClassifier(scale_pos_weight=spw, **params)
        clf.fit(X_tr, y_tr)
        mccs.append(matthews_corrcoef(y_te, clf.predict(X_te)))
    return float(np.mean(mccs))

study = optuna.create_study(direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=seed),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10))
log.info("Iniciando 50 trials ...")
study.optimize(objective, n_trials=50, show_progress_bar=True)
best = study.best_trial
log.info(f"\nMejor MCC={best.value:.4f}  params={best.params}")

log.info("\nEvaluacion final (7 folds) ...")
fold_metrics, y_true_all, y_proba_all = [], [], []
for fold in cv:
    train_df, test_df = fold.split(df)
    prep = FoldPreprocessor(sequence_length=1)
    X_tr, _, y_tr, _ = prep.fit_transform(train_df, feat_cols, target_col)
    X_te, _, y_te, _ = prep.transform(test_df, feat_cols, target_col)
    spw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    params = dict(best.params)
    params.update({"scale_pos_weight": spw, "random_state": seed,
                   "eval_metric": "logloss", "verbosity": 0, "device": "cuda"})
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    probas = clf.predict_proba(X_te)[:, 1]
    m = compute_metrics(y_te, preds, probas)
    fold_metrics.append(m)
    y_true_all.extend(y_te.tolist())
    y_proba_all.extend(probas.tolist())
    log.info(f"  [{fold.label}] Acc={m['accuracy']:.4f} MCC={m['mcc']:.4f} AUC={m['roc_auc']:.4f}")

summary = aggregate_fold_metrics(fold_metrics)
summary_stable = aggregate_fold_metrics(fold_metrics[:-1])
log.info(f"\nRESULTADO XGBoost (7 folds):  Acc={summary['accuracy_mean']:.4f}  MCC={summary['mcc_mean']:.4f}  AUC={summary['roc_auc_mean']:.4f}")
log.info(f"RESULTADO (folds 1-6):         Acc={summary_stable['accuracy_mean']:.4f}  MCC={summary_stable['mcc_mean']:.4f}  AUC={summary_stable['roc_auc_mean']:.4f}")

out = {"experiment_id": "XGBoost_G3_Optuna_final", "dataset": "original_33features",
       "n_trials": 50, "best_params": best.params, "best_trial_mcc": best.value,
       "summary_7folds": {k: v for k, v in summary.items() if not k.endswith("_values")},
       "summary_stable": {k: v for k, v in summary_stable.items() if not k.endswith("_values")},
       "y_true_all": y_true_all, "y_proba_all": y_proba_all}
with open(results_dir / "xgboost_g3_optuna_final.json", "w") as f:
    json.dump(out, f, indent=2)
study.trials_dataframe().to_csv(results_dir / "optuna_xgb_50trials.csv", index=False)
log.info(f"Guardado en {results_dir}")
