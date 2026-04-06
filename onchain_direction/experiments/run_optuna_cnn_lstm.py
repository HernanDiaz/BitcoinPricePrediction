#!/usr/bin/env python3
"""
run_optuna_cnn_lstm.py — Optuna para CNN-LSTM G3.
Configuracion identica al resto de modelos:
  50 trials, TPE sampler, folds 1-6 para busqueda, dataset original 33 features.

CNN-LSTM es el mejor modelo en Omole & Enke (2024) Financial Innovation
con MCC=0.649 sobre un split simple 80/20. Este experimento lo evalua
bajo walk-forward CV con Optuna para busqueda de hiperparametros optima,
incluyendo la longitud de la ventana temporal (seq_len).
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
from src.models.cnn_lstm import CNNLSTMModel
import torch, yaml
from sklearn.metrics import matthews_corrcoef

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("optuna_cnn_lstm")

with open(ROOT / "onchain_direction" / "config.yaml") as f:
    config = yaml.safe_load(f)
config["paths"]["dataset"] = "data/bitcoin_onchain_2013_2025.csv"

df         = impute_missing(load_dataset(config))
cv         = WalkForwardCV(config)
target_col = config["data"]["target_column"]
seed       = config["project"]["random_seed"]
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

group     = FEATURE_GROUPS["G3"]
feat_cols = [c for c in group.features if c in df.columns]

all_folds    = list(cv)
search_folds = all_folds[:-1]   # Folds 1-6
results_dir  = ROOT / "results" / "optuna"
results_dir.mkdir(parents=True, exist_ok=True)

log.info(f"CNN-LSTM Optuna | {len(feat_cols)} features | {len(search_folds)} search folds | 50 trials")
log.info(f"Device: {device}")


def build_cfg(trial: optuna.Trial) -> tuple[dict, int]:
    """Devuelve (model_cfg, seq_len) para un trial dado."""
    seq_len      = trial.suggest_categorical("seq_len", [5, 10, 20, 30])
    conv_filters = trial.suggest_categorical("conv_filters", [32, 64, 128])
    lstm_hidden  = trial.suggest_categorical("lstm_hidden", [32, 64, 128, 256])
    lstm_layers  = trial.suggest_int("lstm_layers", 1, 3)
    dropout      = trial.suggest_float("dropout", 0.1, 0.5)
    lr           = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [32, 64, 128])

    model_cfg = {
        "conv_filters"           : conv_filters,
        "conv_kernel_size"       : 3,      # fijo — kernel 3 es estandar para series temporales
        "lstm_hidden"            : lstm_hidden,
        "lstm_layers"            : lstm_layers,
        "dropout"                : dropout,
        "learning_rate"          : lr,
        "weight_decay"           : weight_decay,
        "batch_size"             : batch_size,
        "epochs"                 : 150,
        "early_stopping_patience": 20,
        "lr_scheduler_patience"  : 10,
        "lr_scheduler_factor"    : 0.5,
        "lr_min"                 : 1e-6,
    }
    return model_cfg, seq_len


def objective(trial: optuna.Trial) -> float:
    model_cfg, seq_len = build_cfg(trial)

    # Inyectamos los params en config para que CNNLSTMModel los lea
    trial_config = {**config, "models": {**config.get("models", {}),
                                          "cnn_lstm": model_cfg}}
    trial_config["hardware"] = {"device": str(device)}

    mccs = []
    for fold in search_folds:
        train_df, test_df = fold.split(df)

        # Validation split (15% del training, minimo 60 dias)
        n_val    = max(int(len(train_df) * 0.15), 60)
        val_df   = train_df.iloc[-n_val:]
        train_df = train_df.iloc[:-n_val]

        prep = FoldPreprocessor(sequence_length=seq_len)
        _, X_tr_seq, _, y_tr = prep.fit_transform(train_df, feat_cols, target_col)
        _, X_va_seq, _, y_va = prep.transform(val_df,   feat_cols, target_col)
        _, X_te_seq, _, y_te = prep.transform(test_df,  feat_cols, target_col)

        class_weight = prep.compute_class_weight(y_tr)

        model = CNNLSTMModel(trial_config, n_features=len(feat_cols),
                             random_seed=seed)
        model.fit(X_tr_seq, y_tr, X_val=X_va_seq, y_val=y_va,
                  class_weight=class_weight)
        preds = model.predict(X_te_seq)
        mccs.append(matthews_corrcoef(y_te, preds))

    return float(np.mean(mccs))


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=seed),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
)
log.info("Iniciando 50 trials ...")
study.optimize(objective, n_trials=50, show_progress_bar=True)

best = study.best_trial
log.info(f"\nMejor MCC={best.value:.6f}  params={best.params}")

# ── Evaluacion final con mejores params ──────────────────────────────────────
log.info("\nEvaluacion final (todos los folds) ...")
best_model_cfg, best_seq_len = build_cfg(best)

fold_metrics, y_true_all, y_proba_all = [], [], []
final_config = {**config, "models": {**config.get("models", {}),
                                      "cnn_lstm": best_model_cfg}}
final_config["hardware"] = {"device": str(device)}

for fold in all_folds:
    train_df, test_df = fold.split(df)

    prep = FoldPreprocessor(sequence_length=best_seq_len)
    _, X_tr_seq, _, y_tr = prep.fit_transform(train_df, feat_cols, target_col)
    _, X_te_seq, _, y_te = prep.transform(test_df,  feat_cols, target_col)

    class_weight = prep.compute_class_weight(y_tr)

    model = CNNLSTMModel(final_config, n_features=len(feat_cols),
                         random_seed=seed)
    model.fit(X_tr_seq, y_tr, class_weight=class_weight)
    preds  = model.predict(X_te_seq)
    probas = model.predict_proba(X_te_seq)[:, 1]

    m = compute_metrics(y_te, preds, probas)
    fold_metrics.append(m)
    y_true_all.extend(y_te.tolist())
    y_proba_all.extend(probas.tolist())
    log.info(f"  [{fold.label}] Acc={m['accuracy']:.4f} MCC={m['mcc']:.4f} AUC={m['roc_auc']:.4f}")

summary_stable = aggregate_fold_metrics(fold_metrics[:-1])
summary_all    = aggregate_fold_metrics(fold_metrics)

log.info(f"\nRESULTADO CNN-LSTM (folds 1-6): "
         f"Acc={summary_stable['accuracy_mean']:.4f} "
         f"MCC={summary_stable['mcc_mean']:.4f} "
         f"AUC={summary_stable['roc_auc_mean']:.4f}")
log.info(f"RESULTADO CNN-LSTM (todos folds): "
         f"Acc={summary_all['accuracy_mean']:.4f} "
         f"MCC={summary_all['mcc_mean']:.4f} "
         f"AUC={summary_all['roc_auc_mean']:.4f}")

fold_metrics_serializable = [
    {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in m.items()}
    for m in fold_metrics
]

out = {
    "experiment_id" : "CNN_LSTM_G3_Optuna_final",
    "dataset"       : "original_33features",
    "n_trials"      : 50,
    "best_params"   : best.params,
    "best_trial_mcc": best.value,
    "best_seq_len"  : best_seq_len,
    "summary_stable": {k: v for k, v in summary_stable.items() if not k.endswith("_values")},
    "summary_7folds": {k: v for k, v in summary_all.items()    if not k.endswith("_values")},
    "fold_metrics"  : fold_metrics_serializable,
    "y_true_all"    : y_true_all,
    "y_proba_all"   : y_proba_all,
}
out_path = results_dir / "cnn_lstm_g3_optuna_final.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

study.trials_dataframe().to_csv(results_dir / "optuna_cnn_lstm_50trials.csv", index=False)
log.info(f"\nGuardado en {out_path}")
