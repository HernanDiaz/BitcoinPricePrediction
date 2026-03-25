#!/usr/bin/env python3
"""
run_ablation_mlp.py
===================
Architectural ablation study for the MLP Dual Encoder V2.

Three variants evaluated with identical hyperparameters (best params from Optuna):
  1. MLP Simple       — single MLP on all 33 features concatenated (no dual encoder)
  2. MLP Dual NoAttn  — dual encoder with concatenation fusion (no cross-attention)
  3. MLP Dual V2      — full architecture with bidirectional cross-attention

Purpose: isolate the contribution of (a) domain separation and (b) cross-attention.
All variants use the same best_params from mlp_dual_v2_optuna.json to ensure fair comparison.
Evaluated on 7-fold walk-forward CV (2019-2025).
"""

import sys, json, logging
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "onchain_direction"))

from src.data.loader import load_dataset, impute_missing
from src.data.feature_groups import get_dual_encoder_splits
from src.data.preprocessor import FoldPreprocessor
from src.validation.walk_forward import WalkForwardCV
from src.evaluation.metrics import compute_metrics, aggregate_fold_metrics
from src.models.mlp_ablation_models import MLPSimpleModel, MLPDualNoAttnModel
from src.models.mlp_dual_encoder_v2 import MLPDualEncoderModelV2
import torch, yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ablation_mlp")

# ── Config ─────────────────────────────────────────────────────────────────
with open(ROOT / "onchain_direction" / "config.yaml") as f:
    config = yaml.safe_load(f)

config["paths"]["dataset"] = "dataset_COMPLETO_con_OHLCV_20251221_014211.csv"

df         = impute_missing(load_dataset(config))
cv         = WalkForwardCV(config)
target_col = config["data"]["target_column"]
seed       = config["project"]["random_seed"]
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tech_cols, onchain_cols = get_dual_encoder_splits(df)
feat_cols = tech_cols + onchain_cols

# ── Load best hyperparameters from V2 Optuna ───────────────────────────────
optuna_results = ROOT / "results" / "optuna" / "mlp_dual_v2_optuna.json"
with open(optuna_results) as f:
    optuna_data = json.load(f)

best_params = dict(optuna_data["best_params"])
best_params.update({"epochs": 300, "patience": 40})   # final eval: more epochs

log.info("=" * 60)
log.info("MLP ARCHITECTURAL ABLATION STUDY")
log.info("=" * 60)
log.info(f"Device : {device}")
log.info(f"Features: tech={len(tech_cols)}, onchain={len(onchain_cols)}, total={len(feat_cols)}")
log.info(f"Best params (from V2 Optuna): {best_params}")
log.info("")

results_dir = ROOT / "results" / "ablation"
results_dir.mkdir(parents=True, exist_ok=True)

# ── Helper ─────────────────────────────────────────────────────────────────

def split_branches(X, feat_cols, tech_cols, onchain_cols):
    tc = [feat_cols.index(c) for c in tech_cols if c in feat_cols]
    oc = [feat_cols.index(c) for c in onchain_cols if c in feat_cols]
    return X[:, tc], X[:, oc]


def evaluate_model(model_cls, model_name):
    log.info(f"\n{'─'*60}")
    log.info(f"Evaluating: {model_name}")
    log.info(f"{'─'*60}")

    fold_metrics, y_true_all, y_proba_all = [], [], []

    for fold in cv:
        train_df, test_df = fold.split(df)
        n_val    = max(int(len(train_df) * 0.15), 60)
        val_df   = train_df.iloc[-n_val:]
        train_df = train_df.iloc[:-n_val]

        prep = FoldPreprocessor(sequence_length=1)
        X_tr, _, y_tr, _ = prep.fit_transform(train_df, feat_cols, target_col)
        X_va, _, y_va, _ = prep.transform(val_df,   feat_cols, target_col)
        X_te, _, y_te, _ = prep.transform(test_df,  feat_cols, target_col)

        X_tr_t, X_tr_o = split_branches(X_tr, feat_cols, tech_cols, onchain_cols)
        X_va_t, X_va_o = split_branches(X_va, feat_cols, tech_cols, onchain_cols)
        X_te_t, X_te_o = split_branches(X_te, feat_cols, tech_cols, onchain_cols)

        pos = y_tr.sum(); neg = len(y_tr) - pos
        class_weight = neg / max(pos, 1)

        model = model_cls(
            cfg=best_params,
            n_technical=X_tr_t.shape[1],
            n_onchain=X_tr_o.shape[1],
            device=device,
            random_seed=seed,
        )
        model.fit(
            (X_tr_t, X_tr_o), y_tr,
            X_val=(X_va_t, X_va_o), y_val=y_va,
            class_weight=class_weight,
        )

        preds  = model.predict((X_te_t, X_te_o))
        probas = model.predict_proba((X_te_t, X_te_o))[:, 1]
        m = compute_metrics(y_te, preds, probas)
        fold_metrics.append(m)
        y_true_all.extend(y_te.tolist())
        y_proba_all.extend(probas.tolist())

        log.info(f"  [{fold.label}] Acc={m['accuracy']:.4f}  MCC={m['mcc']:.4f}  AUC={m['roc_auc']:.4f}")

    summary        = aggregate_fold_metrics(fold_metrics)
    summary_stable = aggregate_fold_metrics(fold_metrics[:-1])   # folds 1-6 (2019-2024)

    log.info(f"\n  RESULTADO {model_name} (7 folds):")
    log.info(f"    Acc={summary['accuracy_mean']:.4f}  MCC={summary['mcc_mean']:.4f}  AUC={summary['roc_auc_mean']:.4f}")
    log.info(f"  RESULTADO {model_name} (folds 1-6, 2019-2024):")
    log.info(f"    Acc={summary_stable['accuracy_mean']:.4f}  MCC={summary_stable['mcc_mean']:.4f}  AUC={summary_stable['roc_auc_mean']:.4f}")

    return {
        "experiment_id"  : model_name,
        "summary_7folds" : summary,
        "summary_stable" : summary_stable,
        "fold_metrics"   : [{k: v for k, v in m.items() if k != "confusion_matrix"}
                            for m in fold_metrics],
        "y_true_all"     : y_true_all,
        "y_proba_all"    : y_proba_all,
    }


# ── Run all three variants ──────────────────────────────────────────────────
variants = [
    (MLPSimpleModel,        "MLP_Simple"),
    (MLPDualNoAttnModel,    "MLP_Dual_NoAttn"),
    (MLPDualEncoderModelV2, "MLP_Dual_V2"),
]

all_results = {}
for model_cls, model_name in variants:
    result = evaluate_model(model_cls, model_name)
    all_results[model_name] = result

# ── Summary table ───────────────────────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("ABLATION SUMMARY — Folds 1-6 (2019-2024, stable period)")
log.info("=" * 70)
log.info(f"{'Model':<22} {'Accuracy':>10} {'MCC':>10} {'AUC':>10}")
log.info("-" * 54)
for name, res in all_results.items():
    s = res["summary_stable"]
    log.info(f"{name:<22} {s['accuracy_mean']:>10.4f} {s['mcc_mean']:>10.4f} {s['roc_auc_mean']:>10.4f}")
log.info("=" * 70)

log.info("\nABLATION SUMMARY — All 7 folds (2019-2025)")
log.info("=" * 70)
log.info(f"{'Model':<22} {'Accuracy':>10} {'MCC':>10} {'AUC':>10}")
log.info("-" * 54)
for name, res in all_results.items():
    s = res["summary_7folds"]
    log.info(f"{name:<22} {s['accuracy_mean']:>10.4f} {s['mcc_mean']:>10.4f} {s['roc_auc_mean']:>10.4f}")
log.info("=" * 70)

# ── Save ────────────────────────────────────────────────────────────────────
out_path = results_dir / "mlp_ablation_results.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)

log.info(f"\nResultados guardados en: {out_path}")
