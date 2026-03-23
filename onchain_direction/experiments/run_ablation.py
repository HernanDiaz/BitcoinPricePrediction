"""
Main ablation experiment runner.

Executes all 13 experiment combinations:
  - 4 feature groups × 3 models (RandomForest, XGBoost, CNN-LSTM)  = 12
  - DualEncoder on G3                                                =  1
                                                                    ──────
  Total                                                              = 13

For each combination, runs 7 walk-forward folds and saves:
  - Per-fold metrics (CSV)
  - Aggregated metrics (CSV)
  - Bootstrap confidence intervals (JSON)
  - Saved model per fold (for later SHAP analysis)
  - Concatenated test predictions (for ROC curves)
  - Training history for PyTorch models (for learning curve plots)
  - Experiment log (JSON)
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_config, load_dataset, impute_missing
from src.data.feature_groups import FEATURE_GROUPS, get_dual_encoder_splits
from src.data.preprocessor import FoldPreprocessor, split_dual_encoder_features
from src.validation.walk_forward import WalkForwardCV
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.cnn_lstm import CNNLSTMModel
from src.models.dual_encoder import DualEncoderModel
from src.evaluation.metrics import compute_metrics, aggregate_fold_metrics
from src.evaluation.bootstrap_ci import bootstrap_ci

# ── logging setup ─────────────────────────────────────────────────────────────
def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / f"ablation_{timestamp}.log"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=handlers,
    )


logger = logging.getLogger("run_ablation")


# ── helpers ───────────────────────────────────────────────────────────────────
def log_environment(config: dict, results_dir: Path) -> None:
    """Save library versions and config for reproducibility."""
    import sklearn, xgboost, shap
    env = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "shap": shap.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "random_seed": config["project"]["random_seed"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "environment.json", "w") as f:
        json.dump(env, f, indent=2)
    logger.info(f"Environment: PyTorch {env['torch']} | CUDA: {env['cuda_device']}")


def get_experiment_id(model_name: str, group_name: str) -> str:
    return f"{model_name}_{group_name}"


# ── per-fold runner ───────────────────────────────────────────────────────────
def run_fold(
    fold,
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    group_name: str,
    config: dict,
    models_dir: Path,
    tech_cols: list[str] | None = None,
    onchain_cols: list[str] | None = None,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train and evaluate one model on one fold.

    Returns:
        metrics      : dict with all classification metrics
        y_true       : ground truth labels
        y_pred       : predicted labels
        y_proba_pos  : predicted probabilities for class 1
    """
    seed = config["project"]["random_seed"]
    seq_len = config["models"].get("cnn_lstm", {}).get("sequence_length", 30)
    is_deep = model_name in ("CNN-LSTM", "DualEncoder")

    df_train, df_test = fold.split(df)
    prep = FoldPreprocessor(sequence_length=seq_len)

    X_train_flat, X_train_seq, y_train_flat, y_train_seq = prep.fit_transform(df_train, feature_cols, "y")
    X_test_flat,  X_test_seq,  y_test_flat,  y_test_seq  = prep.transform(df_test, feature_cols, "y")

    class_weight = prep.compute_class_weight(y_train_flat)

    # Split validation from end of training (last 15% of training sequences)
    val_size = max(30, int(len(X_train_seq) * 0.15))
    if is_deep:
        X_val_seq = X_train_seq[-val_size:]
        y_val_seq = y_train_seq[-val_size:]
        X_tr_seq  = X_train_seq[:-val_size]
        y_tr_seq  = y_train_seq[:-val_size]
    else:
        X_val_flat = X_train_flat[-val_size:]
        y_val_flat = y_train_flat[-val_size:]
        X_tr_flat  = X_train_flat[:-val_size]
        y_tr_flat  = y_train_flat[:-val_size]

    # ── instantiate and train model ──────────────────────────────────────────
    exp_id = get_experiment_id(model_name, group_name)
    model_path = models_dir / f"{exp_id}_{fold.label}.pkl"

    if model_name == "RandomForest":
        model = RandomForestModel(config, random_seed=seed)
        history = model.fit(X_tr_flat, y_tr_flat, class_weight=class_weight)
        y_pred = model.predict(X_test_flat)
        y_proba = model.predict_proba(X_test_flat)
        y_true = y_test_flat

    elif model_name == "XGBoost":
        model = XGBoostModel(config, random_seed=seed)
        history = model.fit(X_tr_flat, y_tr_flat, X_val_flat, y_val_flat, class_weight=class_weight)
        y_pred = model.predict(X_test_flat)
        y_proba = model.predict_proba(X_test_flat)
        y_true = y_test_flat

    elif model_name == "CNN-LSTM":
        model = CNNLSTMModel(config, n_features=len(feature_cols), random_seed=seed)
        history = model.fit(X_tr_seq, y_tr_seq, X_val_seq, y_val_seq, class_weight=class_weight)
        y_pred = model.predict(X_test_seq)
        y_proba = model.predict_proba(X_test_seq)
        y_true = y_test_seq
        model_path = model_path.with_suffix(".pt")

    elif model_name == "DualEncoder":
        X_tr_tech,  X_tr_on  = split_dual_encoder_features(X_tr_seq,  feature_cols, tech_cols, onchain_cols)
        X_val_tech, X_val_on = split_dual_encoder_features(X_val_seq, feature_cols, tech_cols, onchain_cols)
        X_te_tech,  X_te_on  = split_dual_encoder_features(X_test_seq, feature_cols, tech_cols, onchain_cols)

        model = DualEncoderModel(
            config,
            n_technical=X_tr_tech.shape[-1],
            n_onchain=X_tr_on.shape[-1],
            random_seed=seed,
        )
        history = model.fit(
            (X_tr_tech, X_tr_on), y_tr_seq,
            (X_val_tech, X_val_on), y_val_seq,
            class_weight=class_weight,
        )
        y_pred = model.predict((X_te_tech, X_te_on))
        y_proba = model.predict_proba((X_te_tech, X_te_on))
        y_true = y_test_seq
        model_path = model_path.with_suffix(".pt")

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.save(model_path)

    metrics = compute_metrics(y_true, y_pred, y_proba)
    metrics["fold"] = fold.fold
    metrics["fold_label"] = fold.label
    metrics["model"] = model_name
    metrics["group"] = group_name
    metrics["n_train"] = len(df_train)
    metrics["n_test"] = len(df_test)

    logger.info(
        f"  [{fold.label}] {exp_id} | "
        f"Acc={metrics['accuracy']:.4f} F1={metrics['f1_macro']:.4f} "
        f"MCC={metrics['mcc']:.4f} AUC={metrics['roc_auc']:.4f}"
    )

    return metrics, y_true, y_pred, y_proba[:, 1]


# ── main experiment loop ──────────────────────────────────────────────────────
def run_experiment(
    model_name: str,
    group_name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    cv: WalkForwardCV,
    config: dict,
    results_dir: Path,
    tech_cols: list[str] | None = None,
    onchain_cols: list[str] | None = None,
) -> dict:
    exp_id = get_experiment_id(model_name, group_name)
    logger.info(f"\n{'='*60}")
    logger.info(f"  EXPERIMENT: {exp_id}  ({len(cv)} folds)")
    logger.info(f"  Features  : {len(feature_cols)}")
    logger.info(f"{'='*60}")

    metrics_dir = results_dir / "metrics"
    models_dir  = results_dir / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    all_y_true, all_y_pred, all_y_proba = [], [], []

    for fold in cv:
        m, yt, yp, ypr = run_fold(
            fold, df, feature_cols, model_name, group_name,
            config, models_dir, tech_cols, onchain_cols,
        )
        fold_metrics.append(m)
        all_y_true.append(yt)
        all_y_pred.append(yp)
        all_y_proba.append(ypr)

    # Aggregate across folds
    agg = aggregate_fold_metrics(fold_metrics)

    # Bootstrap CI on concatenated test predictions
    y_true_all  = np.concatenate(all_y_true)
    y_pred_all  = np.concatenate(all_y_pred)
    y_proba_all = np.concatenate(all_y_proba)
    ci = bootstrap_ci(
        y_true_all, y_pred_all,
        np.column_stack([1 - y_proba_all, y_proba_all]),
        n_iterations=config["evaluation"]["bootstrap_n_iterations"],
        ci=config["evaluation"]["bootstrap_ci"],
        random_seed=config["project"]["random_seed"],
    )

    # ── persist results ──────────────────────────────────────────────────────
    pd.DataFrame(fold_metrics).to_csv(metrics_dir / f"{exp_id}_fold_metrics.csv", index=False)

    summary = {"experiment_id": exp_id, "model": model_name, "group": group_name,
               "n_features": len(feature_cols), **agg}
    for metric, ci_data in ci.items():
        summary[f"{metric}_ci_lower"] = ci_data["lower"]
        summary[f"{metric}_ci_upper"] = ci_data["upper"]

    with open(metrics_dir / f"{exp_id}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    np.save(metrics_dir / f"{exp_id}_y_true.npy",  y_true_all)
    np.save(metrics_dir / f"{exp_id}_y_pred.npy",  y_pred_all)
    np.save(metrics_dir / f"{exp_id}_y_proba.npy", y_proba_all)

    logger.info(
        f"  RESULT: Acc={agg['accuracy_mean']:.4f}±{agg['accuracy_std']:.4f} "
        f"F1={agg['f1_macro_mean']:.4f}±{agg['f1_macro_std']:.4f} "
        f"MCC={agg['mcc_mean']:.4f}±{agg['mcc_std']:.4f}"
    )

    return {"experiment_id": exp_id, "fold_metrics": fold_metrics, "summary": summary,
            "y_true_all": y_true_all, "y_pred_all": y_pred_all, "y_proba_all": y_proba_all}


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    config_path = ROOT / "config.yaml"
    config = load_config(config_path)

    results_dir = ROOT / config["paths"]["results"]
    setup_logging(results_dir / "logs")
    log_environment(config, results_dir)

    # Load and clean dataset
    df = load_dataset(config)
    df = impute_missing(df)

    cv = WalkForwardCV(config)
    logger.info(f"\n{cv.summary()}")

    seed = config["project"]["random_seed"]
    tech_cols, onchain_cols = get_dual_encoder_splits(list(df.columns))

    all_results = {}

    # ── Standard experiments: G0-G3 × RF + XGBoost (skip CNN-LSTM: known failure) ──
    for group_name in ["G0", "G1", "G2", "G3"]:
        group = FEATURE_GROUPS[group_name]
        feature_cols = [c for c in group.features if c in df.columns]

        for model_name in ["RandomForest", "XGBoost"]:
            result = run_experiment(
                model_name, group_name, df, feature_cols, cv, config, results_dir
            )
            all_results[result["experiment_id"]] = result

    # ── Enriched experiments: G4-G5 × RF + XGBoost ────────────────────────
    for group_name in ["G4", "G5"]:
        if group_name not in FEATURE_GROUPS:
            continue
        group = FEATURE_GROUPS[group_name]
        feature_cols = [c for c in group.features if c in df.columns]

        for model_name in ["RandomForest", "XGBoost"]:
            result = run_experiment(
                model_name, group_name, df, feature_cols, cv, config, results_dir
            )
            all_results[result["experiment_id"]] = result

    # ── DualEncoder on G3 ─────────────────────────────────────────────────
    g3_cols = [c for c in FEATURE_GROUPS["G3"].features if c in df.columns]
    result = run_experiment(
        "DualEncoder", "G3", df, g3_cols, cv, config, results_dir,
        tech_cols=tech_cols, onchain_cols=onchain_cols,
    )
    all_results[result["experiment_id"]] = result

    # ── build summary table ────────────────────────────────────────────────
    from src.visualization.tables import build_main_results_table, save_results_table
    records = [r["summary"] for r in all_results.values()]
    summary_df = build_main_results_table(records)
    save_results_table(summary_df, results_dir / "tables", "main_results")

    # ── ROC curves ────────────────────────────────────────────────────────
    from src.visualization.plots import plot_roc_curves
    roc_data = {
        eid: {
            "y_true_all": r["y_true_all"],
            "y_proba_all": r["y_proba_all"],
            "model": r["summary"]["model"],
            "group": r["summary"]["group"],
        }
        for eid, r in all_results.items()
    }
    plot_roc_curves(roc_data, results_dir / "plots")

    logger.info("\n✓ Ablation experiment complete. Results in: " + str(results_dir))
    return all_results


if __name__ == "__main__":
    main()
