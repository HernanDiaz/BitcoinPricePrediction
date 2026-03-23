"""
SHAP analysis runner.

Loads the best-performing model per feature group from the last ablation run,
computes SHAP values on the full test set (all folds concatenated), and
generates figures and tables.

Run AFTER run_ablation.py has completed.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_config, load_dataset, impute_missing
from src.data.feature_groups import FEATURE_GROUPS
from src.data.preprocessor import FoldPreprocessor
from src.validation.walk_forward import WalkForwardCV
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.cnn_lstm import CNNLSTMModel
from src.models.dual_encoder import DualEncoderModel
from src.explainability.shap_analysis import (
    compute_shap_tree, compute_shap_deep, save_shap_results,
)
from src.visualization.plots import plot_shap_summary, plot_shap_beeswarm, plot_attention_weights
from src.visualization.tables import build_feature_importance_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_shap")


def load_best_models_per_group(results_dir: Path) -> dict:
    """
    Identify the best model per feature group based on saved summary JSONs.
    Best = highest mcc_mean.
    """
    metrics_dir = results_dir / "metrics"
    summaries = {}
    for f in metrics_dir.glob("*_summary.json"):
        with open(f) as fp:
            s = json.load(fp)
        summaries[s["experiment_id"]] = s

    best_per_group = {}
    for group in ["G0", "G1", "G2", "G3"]:
        candidates = {eid: s for eid, s in summaries.items() if s["group"] == group}
        if not candidates:
            continue
        best_eid = max(candidates, key=lambda k: candidates[k].get("mcc_mean", -1))
        best_per_group[group] = candidates[best_eid]
        logger.info(f"Best model for {group}: {best_eid} (MCC={candidates[best_eid].get('mcc_mean', 'N/A'):.4f})")

    return best_per_group


def main():
    config_path = ROOT / "config.yaml"
    config = load_config(config_path)
    results_dir = ROOT / config["paths"]["results"]
    shap_dir    = results_dir / "shap"
    plots_dir   = results_dir / "plots"
    tables_dir  = results_dir / "tables"

    df = load_dataset(config)
    df = impute_missing(df)
    cv = WalkForwardCV(config)
    seed = config["project"]["random_seed"]

    best_per_group = load_best_models_per_group(results_dir)

    for group_name, summary in best_per_group.items():
        model_name = summary["model"]
        exp_id = summary["experiment_id"]

        logger.info(f"\n{'─'*50}")
        logger.info(f"SHAP analysis: {exp_id}")
        logger.info(f"{'─'*50}")

        group = FEATURE_GROUPS[group_name]
        feature_cols = [c for c in group.features if c in df.columns]
        seq_len = config["models"]["cnn_lstm"]["sequence_length"]

        # Collect all train and test data across folds (last fold = most data)
        last_fold = cv.folds[-1]
        df_train, df_test = last_fold.split(df)
        prep = FoldPreprocessor(sequence_length=seq_len)
        X_train_flat, X_train_seq, y_train_flat, y_train_seq = prep.fit_transform(df_train, feature_cols, "y")
        X_test_flat,  X_test_seq,  y_test_flat,  y_test_seq  = prep.transform(df_test, feature_cols, "y")

        # Load saved model (last fold)
        model_path_pkl = results_dir / "models" / f"{exp_id}_{last_fold.label}.pkl"
        model_path_pt  = results_dir / "models" / f"{exp_id}_{last_fold.label}.pt"

        shap_result = None

        if model_name in ("RandomForest", "XGBoost"):
            if model_name == "RandomForest":
                model = RandomForestModel(config, random_seed=seed)
            else:
                model = XGBoostModel(config, random_seed=seed)
            model.load(model_path_pkl)
            shap_result = compute_shap_tree(model, X_train_flat, X_test_flat, feature_cols)

        elif model_name == "CNN-LSTM":
            model = CNNLSTMModel(config, n_features=len(feature_cols), random_seed=seed)
            model.load(model_path_pt)
            shap_result = compute_shap_deep(model, X_train_seq, X_test_seq, feature_cols)

        elif model_name == "DualEncoder":
            from src.data.feature_groups import get_dual_encoder_splits
            from src.data.preprocessor import split_dual_encoder_features
            tech_cols, onchain_cols = get_dual_encoder_splits(list(df.columns))
            X_te_tech, X_te_on = split_dual_encoder_features(X_test_seq, feature_cols, tech_cols, onchain_cols)
            X_tr_tech, X_tr_on = split_dual_encoder_features(X_train_seq, feature_cols, tech_cols, onchain_cols)

            model = DualEncoderModel(
                config,
                n_technical=X_tr_tech.shape[-1],
                n_onchain=X_tr_on.shape[-1],
                random_seed=seed,
            )
            model.load(model_path_pt)

            # For DualEncoder, use attention weights instead of SHAP
            _ = model.predict_proba((X_te_tech, X_te_on))
            tw, ow = model.get_attention_weights()
            np.save(shap_dir / f"{exp_id}_tech_attention.npy", tw)
            np.save(shap_dir / f"{exp_id}_onchain_attention.npy", ow)
            plot_attention_weights(tw, ow, seq_len=seq_len, output_dir=plots_dir)
            logger.info(f"  Attention weights saved and plotted for {exp_id}")
            continue  # no SHAP for DualEncoder

        if shap_result is not None:
            save_shap_results(shap_result, shap_dir, exp_id)
            plot_shap_summary(
                shap_result["feature_importance"], model_name, group_name, plots_dir
            )
            plot_shap_beeswarm(
                shap_result["shap_values"], X_test_flat if model_name != "CNN-LSTM" else X_test_seq[:, -1, :],
                feature_cols, plots_dir,
            )
            build_feature_importance_table(
                shap_result["feature_importance"], model_name, group_name, tables_dir
            )
            logger.info(f"  SHAP done for {exp_id} | Top feature: {shap_result['feature_importance'][0]}")

    logger.info("\n✓ SHAP analysis complete.")


if __name__ == "__main__":
    main()
