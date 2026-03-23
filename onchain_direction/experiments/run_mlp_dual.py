"""
Runs only the MLPDualEncoder experiment on G3 and appends results
to the existing ablation output. Run after run_ablation.py.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_config, load_dataset, impute_missing
from src.data.feature_groups import FEATURE_GROUPS, get_dual_encoder_splits
from src.data.preprocessor import FoldPreprocessor, split_dual_encoder_features
from src.validation.walk_forward import WalkForwardCV
from src.models.mlp_dual_encoder import MLPDualEncoderModel
from src.evaluation.metrics import compute_metrics, aggregate_fold_metrics
from src.evaluation.bootstrap_ci import bootstrap_ci
from src.visualization.tables import build_main_results_table, save_results_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_mlp_dual")


def main():
    config = load_config(ROOT / "config.yaml")
    results_dir = ROOT / config["paths"]["results"]
    metrics_dir = results_dir / "metrics"
    models_dir  = results_dir / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(config)
    df = impute_missing(df)
    cv = WalkForwardCV(config)
    seed = config["project"]["random_seed"]

    tech_cols, onchain_cols = get_dual_encoder_splits(list(df.columns))
    g3_cols = [c for c in FEATURE_GROUPS["G3"].features if c in df.columns]

    # Use flat features (no sequence) — split G3 into technical and on-chain branches
    tech_g3 = [c for c in tech_cols if c in g3_cols]
    onchain_g3 = [c for c in onchain_cols if c in g3_cols]

    exp_id = "MLPDualEncoder_G3"
    logger.info(f"\n{'='*60}")
    logger.info(f"  EXPERIMENT: {exp_id}  ({len(cv)} folds)")
    logger.info(f"  Tech features  : {len(tech_g3)}")
    logger.info(f"  OnChain features: {len(onchain_g3)}")
    logger.info(f"{'='*60}")

    fold_metrics = []
    all_y_true, all_y_pred, all_y_proba = [], [], []

    for fold in cv:
        df_train, df_test = fold.split(df)

        # Use FoldPreprocessor with seq_len=1 to get flat (no sequence) arrays
        prep = FoldPreprocessor(sequence_length=1)
        Xf_tr, _, yf_tr, _ = prep.fit_transform(df_train, g3_cols, "y")
        Xf_te, _, yf_te, _ = prep.transform(df_test, g3_cols, "y")
        cw = prep.compute_class_weight(yf_tr)

        # Split flat arrays into technical and on-chain branches
        tech_idx    = [g3_cols.index(c) for c in tech_g3]
        onchain_idx = [g3_cols.index(c) for c in onchain_g3]
        Xt_tr, Xo_tr = Xf_tr[:, tech_idx], Xf_tr[:, onchain_idx]
        Xt_te, Xo_te = Xf_te[:, tech_idx], Xf_te[:, onchain_idx]

        # Validation split (last 15% of training)
        val_size = max(30, int(len(Xt_tr) * 0.15))
        Xt_val, Xo_val, y_val = Xt_tr[-val_size:], Xo_tr[-val_size:], yf_tr[-val_size:]
        Xt_tr2, Xo_tr2, y_tr2 = Xt_tr[:-val_size], Xo_tr[:-val_size], yf_tr[:-val_size]

        model = MLPDualEncoderModel(
            config,
            n_technical=Xt_tr.shape[-1],
            n_onchain=Xo_tr.shape[-1],
            random_seed=seed,
        )
        history = model.fit(
            (Xt_tr2, Xo_tr2), y_tr2,
            (Xt_val, Xo_val), y_val,
            class_weight=cw,
        )

        y_pred  = model.predict((Xt_te, Xo_te))
        y_proba = model.predict_proba((Xt_te, Xo_te))

        m = compute_metrics(yf_te, y_pred, y_proba)
        m.update({
            "fold": fold.fold, "fold_label": fold.label,
            "model": "MLPDualEncoder", "group": "G3",
            "n_train": len(df_train), "n_test": len(df_test),
            "epochs_trained": history["epochs_trained"],
        })

        logger.info(
            f"  [{fold.label}] {exp_id} | "
            f"Acc={m['accuracy']:.4f} F1={m['f1_macro']:.4f} "
            f"MCC={m['mcc']:.4f} AUC={m['roc_auc']:.4f} "
            f"Epochs={history['epochs_trained']}"
        )

        fold_metrics.append(m)
        all_y_true.append(yf_te)
        all_y_pred.append(y_pred)
        all_y_proba.append(y_proba[:, 1])

        model.save(models_dir / f"{exp_id}_{fold.label}.pt")

    # Aggregate and save
    agg = aggregate_fold_metrics(fold_metrics)
    y_true_all  = np.concatenate(all_y_true)
    y_pred_all  = np.concatenate(all_y_pred)
    y_proba_all = np.concatenate(all_y_proba)

    ci = bootstrap_ci(
        y_true_all, y_pred_all,
        np.column_stack([1 - y_proba_all, y_proba_all]),
        n_iterations=config["evaluation"]["bootstrap_n_iterations"],
        ci=config["evaluation"]["bootstrap_ci"],
        random_seed=seed,
    )

    pd.DataFrame(fold_metrics).to_csv(metrics_dir / f"{exp_id}_fold_metrics.csv", index=False)
    np.save(metrics_dir / f"{exp_id}_y_true.npy",  y_true_all)
    np.save(metrics_dir / f"{exp_id}_y_pred.npy",  y_pred_all)
    np.save(metrics_dir / f"{exp_id}_y_proba.npy", y_proba_all)

    summary = {
        "experiment_id": exp_id, "model": "MLPDualEncoder", "group": "G3",
        "n_features": len(g3_cols), **agg,
        **{f"{m}_ci_lower": ci[m]["lower"] for m in ci},
        **{f"{m}_ci_upper": ci[m]["upper"] for m in ci},
    }
    with open(metrics_dir / f"{exp_id}_summary.json", "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, np.floating) else x)

    logger.info(
        f"\n  RESULT: Acc={agg['accuracy_mean']:.4f}+-{agg['accuracy_std']:.4f} "
        f"F1={agg['f1_macro_mean']:.4f}+-{agg['f1_macro_std']:.4f} "
        f"MCC={agg['mcc_mean']:.4f}+-{agg['mcc_std']:.4f}"
    )

    # Rebuild complete summary table with all 14 experiments
    all_summaries = []
    for path in sorted((metrics_dir).glob("*_summary.json")):
        with open(path) as fp:
            all_summaries.append(json.load(fp))

    df_summary = build_main_results_table(all_summaries)
    save_results_table(df_summary, results_dir / "tables", "main_results_v2")
    logger.info(f"\n  Updated table saved -> {results_dir}/tables/main_results_v2.csv")
    logger.info("\nMLPDualEncoder experiment complete.")


if __name__ == "__main__":
    main()
