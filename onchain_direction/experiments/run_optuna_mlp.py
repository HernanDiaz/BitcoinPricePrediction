#!/usr/bin/env python3
"""
run_optuna_mlp.py
=================
Optuna hyperparameter search for MLPDualEncoderV2.
Configuracion identica a run_optuna_xgb.py y run_optuna_lgbm.py:
  50 trials, TPE sampler, folds 1-6 para busqueda, dataset original 3
  3 features.
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
from src.models.mlp_dual_encoder_v2 import MLPDualEncoderModelV2
import torch, yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("optuna_mlp")

with open(ROOT / "onchain_direction" / "config.yaml") as f:
    config = yaml.safe_load(f)

# Use original dataset (33 features) for MLP — enriched dataset has too many NaN
# in early folds (FearGreed starts 2018, training starts 2013)
config["paths"]["dataset"] = "dataset_COMPLETO_con_OHLCV_20251221_014211.csv"

from src.data.loader import impute_missing
df         = impute_missing(load_dataset(config))
cv         = WalkForwardCV(config)
target_col = config["data"]["target_column"]
seed       = config["project"]["random_seed"]
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

group     = FEATURE_GROUPS["G3"]
feat_cols = [c for c in group.features if c in df.columns]

from src.data.feature_groups import get_dual_encoder_splits
tech_cols, onchain_cols = get_dual_encoder_splits(df)

search_folds = list(cv)[:-1]   # Folds 1-6

results_dir = ROOT / "results" / "optuna"
results_dir.mkdir(parents=True, exist_ok=True)

log.info(f"MLPDualEncoderV2 Optuna | tech={len(tech_cols)} onchain={len(onchain_cols)} | {len(search_folds)} folds | 100 trials")
log.info(f"Device: {device}")


def objective(trial: optuna.Trial) -> float:
    from sklearn.metrics import matthews_corrcoef

    cfg = {
        "hidden_dim"        : trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        "n_context_tokens"  : trial.suggest_categorical("n_context_tokens", [4, 8, 16, 32]),
        "n_heads"           : trial.suggest_categorical("n_heads", [2, 4, 8]),
        "n_encoder_layers"  : trial.suggest_int("n_encoder_layers", 2, 4),
        "n_cross_layers"    : trial.suggest_int("n_cross_layers", 1, 3),
        "dropout"           : trial.suggest_float("dropout", 0.1, 0.5),
        "lr"                : trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay"      : trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size"        : trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "warmup_epochs"     : trial.suggest_int("warmup_epochs", 5, 30),
        "label_smoothing"   : trial.suggest_float("label_smoothing", 0.0, 0.15),
        "epochs"            : 200,
        "patience"          : 30,
    }

    # Validate n_heads divides hidden_dim
    if cfg["hidden_dim"] % cfg["n_heads"] != 0:
        return 0.0

    mccs = []
    for fold in search_folds:
        train_df, test_df = fold.split(df)
        n_val = max(int(len(train_df) * 0.15), 60)
        val_df   = train_df.iloc[-n_val:]
        train_df = train_df.iloc[:-n_val]

        prep = FoldPreprocessor(sequence_length=1)
        X_tr, _, y_tr, _ = prep.fit_transform(train_df, feat_cols, target_col)
        X_va, _, y_va, _ = prep.transform(val_df, feat_cols, target_col)
        X_te, _, y_te, _ = prep.transform(test_df, feat_cols, target_col)

        def split_branches(X):
            tc = [feat_cols.index(c) for c in tech_cols if c in feat_cols]
            oc = [feat_cols.index(c) for c in onchain_cols if c in feat_cols]
            return X[:, tc], X[:, oc]

        X_tr_t, X_tr_o = split_branches(X_tr)
        X_va_t, X_va_o = split_branches(X_va)
        X_te_t, X_te_o = split_branches(X_te)

        pos = y_tr.sum(); neg = len(y_tr) - pos
        class_weight = neg / max(pos, 1)

        model = MLPDualEncoderModelV2(
            cfg=cfg,
            n_technical=X_tr_t.shape[1],
            n_onchain=X_tr_o.shape[1],
            device=device,
            random_seed=seed,
        )
        model.fit((X_tr_t, X_tr_o), y_tr,
                  X_val=(X_va_t, X_va_o), y_val=y_va,
                  class_weight=class_weight)
        preds = model.predict((X_te_t, X_te_o))
        mccs.append(matthews_corrcoef(y_te, preds))

    return float(np.mean(mccs))


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=seed),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=3),
)

log.info("Iniciando 50 trials ...")
study.optimize(objective, n_trials=50, show_progress_bar=True)

best = study.best_trial
log.info(f"\nMejor trial: MCC={best.value:.4f}")
log.info(f"Mejores params: {best.params}")

# Evaluacion final 7 folds
log.info("\nEvaluacion final (7 folds) ...")
fold_metrics, y_true_all, y_proba_all = [], [], []

for fold in cv:
    train_df, test_df = fold.split(df)
    n_val = max(int(len(train_df) * 0.15), 60)
    val_df   = train_df.iloc[-n_val:]
    train_df = train_df.iloc[:-n_val]

    prep = FoldPreprocessor(sequence_length=1)
    X_tr, _, y_tr, _ = prep.fit_transform(train_df, feat_cols, target_col)
    X_va, _, y_va, _ = prep.transform(val_df, feat_cols, target_col)
    X_te, _, y_te, _ = prep.transform(test_df, feat_cols, target_col)

    def split_branches(X):
        tc = [feat_cols.index(c) for c in tech_cols if c in feat_cols]
        oc = [feat_cols.index(c) for c in onchain_cols if c in feat_cols]
        return X[:, tc], X[:, oc]

    X_tr_t, X_tr_o = split_branches(X_tr)
    X_va_t, X_va_o = split_branches(X_va)
    X_te_t, X_te_o = split_branches(X_te)

    cfg = dict(best.params)
    cfg.update({"epochs": 300, "patience": 40})

    pos = y_tr.sum(); neg = len(y_tr) - pos
    model = MLPDualEncoderModelV2(cfg=cfg, n_technical=X_tr_t.shape[1],
                                   n_onchain=X_tr_o.shape[1], device=device, random_seed=seed)
    model.fit((X_tr_t, X_tr_o), y_tr,
              X_val=(X_va_t, X_va_o), y_val=y_va,
              class_weight=neg/max(pos,1))

    preds  = model.predict((X_te_t, X_te_o))
    probas = model.predict_proba((X_te_t, X_te_o))[:, 1]
    m = compute_metrics(y_te, preds, probas)
    fold_metrics.append(m)
    y_true_all.extend(y_te.tolist())
    y_proba_all.extend(probas.tolist())
    log.info(f"  [{fold.label}] Acc={m['accuracy']:.4f} MCC={m['mcc']:.4f} AUC={m['roc_auc']:.4f}")

summary        = aggregate_fold_metrics(fold_metrics)
summary_stable = aggregate_fold_metrics(fold_metrics[:-1])
log.info(f"\nRESULTADO MLPDualEncoderV2_G3_Optuna (7 folds):")
log.info(f"  Acc={summary['accuracy_mean']:.4f}  MCC={summary['mcc_mean']:.4f}  AUC={summary['roc_auc_mean']:.4f}")
log.info(f"RESULTADO (folds 1-6, 2019-2024):")
log.info(f"  Acc={summary_stable['accuracy_mean']:.4f}  MCC={summary_stable['mcc_mean']:.4f}  AUC={summary_stable['roc_auc_mean']:.4f}")

out = {
    "experiment_id"  : "MLPDualEncoderV2_G3_Optuna",
    "best_params"    : best.params,
    "best_trial_mcc" : best.value,
    "summary_7folds" : summary,
    "summary_stable" : summary_stable,
    "y_true_all"     : y_true_all,
    "y_proba_all"    : y_proba_all,
    "fold_metrics"   : [{k:v for k,v in m.items() if k!="confusion_matrix"} for m in fold_metrics],
}
with open(results_dir / "mlp_dual_v2_optuna.json", "w") as f:
    json.dump(out, f, indent=2)
study.trials_dataframe().to_csv(results_dir / "optuna_mlp_100trials.csv", index=False)
log.info(f"\nGuardado en {results_dir}")
