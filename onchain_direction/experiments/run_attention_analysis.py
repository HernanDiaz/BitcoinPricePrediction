#!/usr/bin/env python3
"""
run_attention_analysis.py
=========================
Extrae y guarda los pesos de cross-attention de DECA (MLPDualEncoderV2)
para cada fold (1-6, excluyendo 2025).

Por cada muestra del conjunto de test guarda:
  - Pesos de cross-attention tech→onchain (K×K matriz)
  - Pesos de cross-attention onchain→tech (K×K matriz)
  - Etiqueta real (y_true)
  - Prediccion del modelo (y_pred)
  - Probabilidad de subida (p_up)
  - Fold al que pertenece

Ademas calcula por muestra:
  - Entropía de atención (concentración): cuánto de focalizada está la atención
  - Concentración normalizada: 1 - H/log(K), rango [0,1]

Todo se guarda en results/attention/ antes de generar graficas.
"""

import sys, json, logging
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "onchain_direction"))

from src.data.loader import load_dataset, impute_missing
from src.data.feature_groups import FEATURE_GROUPS, get_dual_encoder_splits
from src.data.preprocessor import FoldPreprocessor
from src.validation.walk_forward import WalkForwardCV
from src.evaluation.metrics import compute_metrics
from src.models.mlp_dual_encoder_v2 import MLPDualEncoderModelV2
import torch, yaml
from sklearn.metrics import matthews_corrcoef

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("attention_analysis")

# ── Configuracion ─────────────────────────────────────────────────────────────
with open(ROOT / "onchain_direction" / "config.yaml") as f:
    config = yaml.safe_load(f)
config["paths"]["dataset"] = "dataset_COMPLETO_con_OHLCV_20251221_014211.csv"

df         = impute_missing(load_dataset(config))
cv         = WalkForwardCV(config)
target_col = config["data"]["target_column"]
seed       = config["project"]["random_seed"]
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

group     = FEATURE_GROUPS["G3"]
feat_cols = [c for c in group.features if c in df.columns]
tech_cols, onchain_cols = get_dual_encoder_splits(df)

all_folds    = list(cv)
search_folds = all_folds[:-1]   # Folds 1-6 (excluye 2025)

# ── Mejores hiperparametros (Optuna ya ejecutado) ─────────────────────────────
with open(ROOT / "results" / "optuna" / "mlp_dual_v2_optuna.json") as f:
    optuna_results = json.load(f)
best_params = optuna_results["best_params"]
best_params["epochs"]  = 200
best_params["patience"] = 30

log.info(f"DECA best_params: {best_params}")
log.info(f"n_context_tokens = {best_params['n_context_tokens']}  "
         f"(attention matrices = {best_params['n_context_tokens']}x{best_params['n_context_tokens']})")

out_dir = ROOT / "results" / "attention"
out_dir.mkdir(parents=True, exist_ok=True)


def split_branches(X, feat_cols, tech_cols, onchain_cols):
    tc = [feat_cols.index(c) for c in tech_cols if c in feat_cols]
    oc = [feat_cols.index(c) for c in onchain_cols if c in feat_cols]
    return X[:, tc], X[:, oc]


def attention_entropy(weights: np.ndarray) -> np.ndarray:
    """
    Calcula la entropía de cada fila de la matriz de atención.
    weights: (B, K_q, K_k) — B muestras, K_q queries, K_k keys
    Devuelve: (B,) — entropía media por muestra (promedio sobre las K_q filas)
    """
    eps = 1e-9
    # Entropia por fila: -sum(w * log(w))  shape (B, K_q)
    H_rows = -np.sum(weights * np.log(weights + eps), axis=-1)
    # Media sobre queries -> (B,)
    return H_rows.mean(axis=-1)


def attention_concentration(weights: np.ndarray, K: int) -> np.ndarray:
    """
    Concentracion normalizada: 1 - H / log(K)
    0 = atención completamente difusa (uniforme)
    1 = atención completamente focalizada en un token
    """
    H = attention_entropy(weights)
    log_K = np.log(K) if K > 1 else 1.0
    return 1.0 - H / log_K


# ── Extraccion por fold ───────────────────────────────────────────────────────
all_records = []   # lista de dicts, uno por muestra
fold_summaries = []

for fold_idx, fold in enumerate(search_folds):
    fold_year = 2019 + fold_idx
    log.info(f"\n{'='*60}")
    log.info(f"Fold {fold_idx+1} — Test {fold_year}")

    train_df, test_df = fold.split(df)

    prep = FoldPreprocessor(sequence_length=1)
    X_tr, _, y_tr, _ = prep.fit_transform(train_df, feat_cols, target_col)
    X_te, _, y_te, _ = prep.transform(test_df,  feat_cols, target_col)

    X_tr_t, X_tr_o = split_branches(X_tr, feat_cols, tech_cols, onchain_cols)
    X_te_t, X_te_o = split_branches(X_te, feat_cols, tech_cols, onchain_cols)

    class_weight = prep.compute_class_weight(y_tr)

    model = MLPDualEncoderModelV2(
        cfg=best_params,
        n_technical=X_tr_t.shape[1],
        n_onchain=X_tr_o.shape[1],
        device=device,
        random_seed=seed,
    )
    model.fit((X_tr_t, X_tr_o), y_tr, class_weight=class_weight)

    probas = model.predict_proba((X_te_t, X_te_o))
    preds  = (probas[:, 1] >= 0.5).astype(int)
    tw, ow = model.get_attention_weights()   # (N, K, K) each

    K = tw.shape[-1]   # n_context_tokens

    # Concentracion por muestra
    conc_to  = attention_concentration(tw, K)   # tech→onchain
    conc_ot  = attention_concentration(ow, K)   # onchain→tech

    mcc = matthews_corrcoef(y_te, preds)
    log.info(f"  MCC={mcc:.4f}  N_test={len(y_te)}")
    log.info(f"  Mean concentration tech→onchain : {conc_to.mean():.4f} ± {conc_to.std():.4f}")
    log.info(f"  Mean concentration onchain→tech : {conc_ot.mean():.4f} ± {conc_ot.std():.4f}")

    # Guardar matrices de atención medias del fold (K×K)
    mean_tw_fold = tw.mean(axis=0).tolist()   # (K, K)
    mean_ow_fold = ow.mean(axis=0).tolist()

    fold_summaries.append({
        "fold"             : fold_idx + 1,
        "year"             : fold_year,
        "mcc"              : float(mcc),
        "n_test"           : int(len(y_te)),
        "K"                : int(K),
        "mean_attn_to_matrix" : mean_tw_fold,
        "mean_attn_ot_matrix" : mean_ow_fold,
        "mean_conc_to"     : float(conc_to.mean()),
        "std_conc_to"      : float(conc_to.std()),
        "mean_conc_ot"     : float(conc_ot.mean()),
        "std_conc_ot"      : float(conc_ot.std()),
    })

    # Guardar registros por muestra
    for i in range(len(y_te)):
        all_records.append({
            "fold"         : fold_idx + 1,
            "year"         : fold_year,
            "y_true"       : int(y_te[i]),
            "y_pred"       : int(preds[i]),
            "p_up"         : float(probas[i, 1]),
            "correct"      : int(y_te[i] == preds[i]),
            "conc_to"      : float(conc_to[i]),
            "conc_ot"      : float(conc_ot[i]),
            # Matrices K×K aplanadas para reconstruccion posterior
            "attn_to_flat" : tw[i].tolist(),
            "attn_ot_flat" : ow[i].tolist(),
        })

# ── Guardar resultados ────────────────────────────────────────────────────────
fold_path   = out_dir / "attention_fold_summaries.json"
sample_path = out_dir / "attention_sample_records.json"

with open(fold_path, "w") as f:
    json.dump({"fold_summaries": fold_summaries,
               "K": int(K),
               "tech_cols": tech_cols,
               "onchain_cols": onchain_cols}, f, indent=2)

with open(sample_path, "w") as f:
    json.dump(all_records, f, indent=2)

log.info(f"\n{'='*60}")
log.info(f"Guardados {len(all_records)} registros de muestras → {sample_path}")
log.info(f"Guardados {len(fold_summaries)} resúmenes de fold  → {fold_path}")

# ── Resumen global ────────────────────────────────────────────────────────────
records = all_records
conc_to_all = np.array([r["conc_to"] for r in records])
conc_ot_all = np.array([r["conc_ot"] for r in records])
correct     = np.array([r["correct"] for r in records], dtype=bool)
pred_up     = np.array([r["y_pred"]  for r in records]) == 1

log.info("\n=== RESUMEN GLOBAL ===")
log.info(f"  Concentracion media tech→onchain : {conc_to_all.mean():.4f}")
log.info(f"  Concentracion media onchain→tech : {conc_ot_all.mean():.4f}")
log.info(f"\n  Correcto  — conc tech→onchain : {conc_to_all[correct].mean():.4f}")
log.info(f"  Incorrecto — conc tech→onchain : {conc_to_all[~correct].mean():.4f}")
log.info(f"\n  Pred UP   — conc tech→onchain : {conc_to_all[pred_up].mean():.4f}")
log.info(f"  Pred DOWN — conc tech→onchain : {conc_to_all[~pred_up].mean():.4f}")
