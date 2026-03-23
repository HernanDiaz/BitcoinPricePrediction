#!/usr/bin/env python3
"""
Evaluación final con los mejores hiperparámetros encontrados por Optuna.
"""
import sys, json, logging
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "onchain_direction"))

from src.data.loader import load_dataset
from src.data.feature_groups import FEATURE_GROUPS
from src.data.preprocessor import FoldPreprocessor
from src.validation.walk_forward import WalkForwardCV
from src.evaluation.metrics import compute_metrics, aggregate_fold_metrics
import yaml, xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("optuna_eval")

with open(ROOT / "onchain_direction" / "config.yaml") as f:
    config = yaml.safe_load(f)

df  = load_dataset(config)
cv  = WalkForwardCV(config)
target_col = config["data"]["target_column"]
seed = config["project"]["random_seed"]

group     = FEATURE_GROUPS["G3"]
feat_cols = [c for c in group.features if c in df.columns]

# Best params from Optuna (100 trials, best MCC=0.4632)
best_params = {
    "n_estimators"     : 1000,
    "max_depth"        : 5,
    "learning_rate"    : 0.03461105864859672,
    "subsample"        : 0.5993512265902075,
    "colsample_bytree" : 0.5244834429593548,
    "min_child_weight" : 3,
    "gamma"            : 3.8940524421166822,
    "reg_alpha"        : 0.000630323305565728,
    "reg_lambda"       : 0.042337001176192476,
    "eval_metric"      : "logloss",
    "verbosity"        : 0,
    "device"           : "cuda",
    "random_state"     : seed,
}

log.info(f"Evaluacion final XGBoost_G3_Optuna | {len(feat_cols)} features | {len(cv.folds)} folds")
log.info(f"Params: {best_params}")

fold_metrics = []
y_true_all, y_proba_all = [], []

for fold in cv:
    train_df, test_df = fold.split(df)
    prep = FoldPreprocessor(sequence_length=1)
    X_tr, _, y_tr, _ = prep.fit_transform(train_df, feat_cols, target_col)
    X_te, _, y_te, _ = prep.transform(test_df, feat_cols, target_col)

    pos = y_tr.sum(); neg = len(y_tr) - pos
    params = dict(best_params)
    params["scale_pos_weight"] = neg / max(pos, 1)

    clf = xgb.XGBClassifier(**params)
    clf.fit(X_tr, y_tr)

    preds  = clf.predict(X_te)
    probas = clf.predict_proba(X_te)[:, 1]
    m = compute_metrics(y_te, preds, probas)
    fold_metrics.append(m)
    y_true_all.extend(y_te.tolist())
    y_proba_all.extend(probas.tolist())

    log.info(f"  [{fold.label}] Acc={m['accuracy']:.4f} MCC={m['mcc']:.4f} AUC={m['roc_auc']:.4f}")

summary = aggregate_fold_metrics(fold_metrics)
log.info(f"\nRESULTADO FINAL (XGBoost_G3_Optuna):")
log.info(f"  Acc  = {summary['accuracy_mean']:.4f} +- {summary['accuracy_std']:.4f}")
log.info(f"  F1   = {summary['f1_macro_mean']:.4f} +- {summary['f1_macro_std']:.4f}")
log.info(f"  MCC  = {summary['mcc_mean']:.4f} +- {summary['mcc_std']:.4f}")
log.info(f"  AUC  = {summary['roc_auc_mean']:.4f} +- {summary['roc_auc_std']:.4f}")

results_dir = ROOT / "results" / "optuna"
results_dir.mkdir(parents=True, exist_ok=True)

out = {
    "experiment_id" : "XGBoost_G3_Optuna",
    "best_params"   : best_params,
    "summary"       : summary,
    "y_true_all"    : y_true_all,
    "y_proba_all"   : y_proba_all,
    "fold_metrics"  : [{k: v for k, v in m.items() if k != "confusion_matrix"}
                       for m in fold_metrics],
}
with open(results_dir / "xgboost_g3_optuna.json", "w") as f:
    json.dump(out, f, indent=2)

log.info(f"\nGuardado en: {results_dir / 'xgboost_g3_optuna.json'}")
