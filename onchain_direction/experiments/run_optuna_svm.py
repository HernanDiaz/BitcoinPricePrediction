#!/usr/bin/env python3
"""
run_optuna_svm.py — Optuna para SVM (RBF kernel) G3.
Configuracion identica al resto de modelos:
  50 trials, TPE sampler, folds 1-6 para busqueda, dataset original 33 features.

SVM es el mejor modelo reportado en Omole & Enke (2025) EAAI con 83% acc
sobre un split simple 80/20. Este experimento lo evalua bajo walk-forward CV.
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
import yaml
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("optuna_svm")

with open(ROOT / "onchain_direction" / "config.yaml") as f:
    config = yaml.safe_load(f)
config["paths"]["dataset"] = "dataset_COMPLETO_con_OHLCV_20251221_014211.csv"

df         = impute_missing(load_dataset(config))
cv         = WalkForwardCV(config)
target_col = config["data"]["target_column"]
seed       = config["project"]["random_seed"]
group      = FEATURE_GROUPS["G3"]
feat_cols  = [c for c in group.features if c in df.columns]

# Folds 1-6 para busqueda (mismo protocolo que XGBoost/LightGBM/MLP)
all_folds    = list(cv)
search_folds = all_folds[:-1]   # excluye fold 7 (2025, fuera del paper)
results_dir  = ROOT / "results" / "optuna"
results_dir.mkdir(parents=True, exist_ok=True)

log.info(f"SVM Optuna | {len(feat_cols)} features | {len(search_folds)} search folds | 50 trials")


def objective(trial):
    # Espacio de hiperparametros: kernel RBF con C y gamma
    # Incluimos tambien 'poly' y 'sigmoid' para que Optuna explore
    kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])
    C      = trial.suggest_float("C", 1e-2, 1e3, log=True)
    gamma  = trial.suggest_float("gamma", 1e-4, 10.0, log=True)

    params = dict(
        kernel=kernel, C=C, gamma=gamma,
        class_weight="balanced",   # equivalente a scale_pos_weight en XGBoost
        probability=True,          # necesario para AUC / predict_proba
        random_state=seed,
        cache_size=500,            # MB de cache para acelerar
        max_iter=20000,
    )
    if kernel == "poly":
        params["degree"] = trial.suggest_int("degree", 2, 4)
        params["coef0"]  = trial.suggest_float("coef0", 0.0, 5.0)

    mccs = []
    for fold in search_folds:
        train_df, test_df = fold.split(df)
        prep = FoldPreprocessor(sequence_length=1)
        X_tr, _, y_tr, _ = prep.fit_transform(train_df, feat_cols, target_col)
        X_te, _, y_te, _ = prep.transform(test_df, feat_cols, target_col)
        # SVM requiere StandardScaler (media=0, var=1) ademas de RobustScaler
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        clf = SVC(**params)
        clf.fit(X_tr, y_tr)
        mccs.append(matthews_corrcoef(y_te, clf.predict(X_te)))
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

# ── Evaluacion final con los mejores parametros ──────────────────────────────
log.info("\nEvaluacion final (todos los folds) ...")
fold_metrics, y_true_all, y_proba_all = [], [], []

for fold in all_folds:
    train_df, test_df = fold.split(df)
    prep = FoldPreprocessor(sequence_length=1)
    X_tr, _, y_tr, _ = prep.fit_transform(train_df, feat_cols, target_col)
    X_te, _, y_te, _ = prep.transform(test_df, feat_cols, target_col)

    # Reconstruir params limpios
    bp = dict(best.params)
    clf = SVC(
        kernel   = bp["kernel"],
        C        = bp["C"],
        gamma    = bp["gamma"],
        degree   = bp.get("degree", 3),
        coef0    = bp.get("coef0", 0.0),
        class_weight = "balanced",
        probability  = True,
        random_state = seed,
        cache_size   = 500,
        max_iter     = 20000,
    )
    # Mismo escalado que en la busqueda
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    clf.fit(X_tr, y_tr)
    preds  = clf.predict(X_te)
    probas = clf.predict_proba(X_te)[:, 1]
    m = compute_metrics(y_te, preds, probas)
    fold_metrics.append(m)
    y_true_all.extend(y_te.tolist())
    y_proba_all.extend(probas.tolist())
    log.info(f"  [{fold.label}] Acc={m['accuracy']:.4f} MCC={m['mcc']:.4f} AUC={m['roc_auc']:.4f}")

# Metricas folds 1-6 (periodo estable, reportado en el paper)
summary_stable = aggregate_fold_metrics(fold_metrics[:-1])
summary_all    = aggregate_fold_metrics(fold_metrics)

log.info(f"\nRESULTADO SVM (folds 1-6): "
         f"Acc={summary_stable['accuracy_mean']:.4f} "
         f"MCC={summary_stable['mcc_mean']:.4f} "
         f"AUC={summary_stable['roc_auc_mean']:.4f}")
log.info(f"RESULTADO SVM (todos folds): "
         f"Acc={summary_all['accuracy_mean']:.4f} "
         f"MCC={summary_all['mcc_mean']:.4f} "
         f"AUC={summary_all['roc_auc_mean']:.4f}")

# Guardar fold_metrics completos para run_paper_stats.py
fold_metrics_serializable = [
    {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in m.items()}
    for m in fold_metrics
]

out = {
    "experiment_id" : "SVM_G3_Optuna_final",
    "dataset"       : "original_33features",
    "n_trials"      : 50,
    "best_params"   : best.params,
    "best_trial_mcc": best.value,
    "summary_stable": {k: v for k, v in summary_stable.items() if not k.endswith("_values")},
    "summary_7folds": {k: v for k, v in summary_all.items()    if not k.endswith("_values")},
    "fold_metrics"  : fold_metrics_serializable,
    "y_true_all"    : y_true_all,
    "y_proba_all"   : y_proba_all,
}
out_path = results_dir / "svm_g3_optuna_final.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

study.trials_dataframe().to_csv(results_dir / "optuna_svm_50trials.csv", index=False)
log.info(f"\nGuardado en {out_path}")
