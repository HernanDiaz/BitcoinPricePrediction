#!/usr/bin/env python3
"""
run_paper_plots.py
==================
Genera todas las figuras finales del paper (folds 1-6, 2019-2024).

Figuras producidas:
  Fig A — Model comparison bar chart (Acc / MCC / AUC, folds 1-6 ± std)
  Fig B — Per-fold MCC chart (7 modelos, 6 folds)
  Fig C — ROC curves (todos los modelos, folds 1-6 concatenados)
  Fig D — Wilcoxon significance heatmap

Ejecutar DESPUES de: run_paper_stats.py y run_paper_shap.py
"""

import sys, json, logging, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Elsevier figure style ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":        9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "legend.framealpha": 0.85,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})
COL1 = 3.54   # single column 90 mm
COL2 = 7.48   # double column 190 mm

import matplotlib.container as mcontainer
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, auc

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "onchain_direction"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("paper_plots")

RESULTS_DIR = ROOT / "results" / "optuna"
STATS_DIR   = ROOT / "results" / "statistical_tests"
PLOTS_DIR   = ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PAPER_FIGS_DIR = ROOT / "paper" / "EAAI" / "figures"
PAPER_FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Metadatos de modelos (orden = importancia descendente) ────────────────────
MODEL_META = {
    "DECA":      {"label": "DECA (ours)", "color": "#4E79A7", "marker": "*"},
    "SVM":       {"label": "SVM G3",      "color": "#B07AA1", "marker": "P"},
    "XGBoost":   {"label": "XGBoost G3",  "color": "#E15759", "marker": "o"},
    "LightGBM":  {"label": "LightGBM G3", "color": "#F28E2B", "marker": "s"},
    "CNN_LSTM":  {"label": "CNN-LSTM G3", "color": "#9C755F", "marker": "v"},
}

FOLD_LABELS = ["2019", "2020", "2021", "2022", "2023", "2024"]
N_STABLE_FOLDS = 6


def save_fig(fig, name: str):
    fig.savefig(PLOTS_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(PLOTS_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {name}.pdf/.png")


# ── Cargar datos ──────────────────────────────────────────────────────────────
def load_model_data():
    """
    Carga resultados de todos los modelos (solo folds 1-6).
    Usa el stats JSON para XGBoost/LightGBM/SVM (recomputados alli),
    y los JSON de Optuna para los modelos con fold_metrics.
    """
    data = {}

    # Calcular n_fold7 para slicing usando el stats JSON
    stats_path = STATS_DIR / "statistical_tests_full.json"
    n_fold7 = 0
    if stats_path.exists():
        stats_full = json.load(open(stats_path))
        n_fold7 = stats_full.get("n_fold7_excluded", 0)
    log.info(f"  n_fold7 para slicing = {n_fold7}")

    # Modelos con fold_metrics en JSON de Optuna
    json_models = {
        "DECA":     "mlp_dual_v2_optuna.json",
        "CNN_LSTM": "cnn_lstm_g3_optuna_final.json",
        "SVM":      "svm_g3_optuna_final.json",
        "XGBoost":  "xgboost_g3_optuna_final.json",
        "LightGBM": "lightgbm_g3_optuna_final.json",
    }

    # Precalcular tamaños de fold desde SVM (siempre tiene fold_metrics)
    _svm_path = RESULTS_DIR / "svm_g3_optuna_final.json"
    _fold_sizes = None
    if _svm_path.exists():
        _svm_d = json.load(open(_svm_path))
        if "fold_metrics" in _svm_d:
            _fold_sizes = [fm["n_test"] for fm in _svm_d["fold_metrics"][:N_STABLE_FOLDS]]
            log.info(f"  Fold sizes (from SVM): {_fold_sizes}")

    for key, fname in json_models.items():
        fpath = RESULTS_DIR / fname
        if not fpath.exists():
            log.warning(f"  Archivo no encontrado: {fname} — saltando {key}")
            continue

        d = json.load(open(fpath))

        # ── Per-fold metrics (solo folds 1-6) ──
        if "fold_metrics" in d:
            fold_mccs = [fm["mcc"]      for fm in d["fold_metrics"][:N_STABLE_FOLDS]]
            fold_accs = [fm["accuracy"] for fm in d["fold_metrics"][:N_STABLE_FOLDS]]
            fold_aucs = [fm["roc_auc"]  for fm in d["fold_metrics"][:N_STABLE_FOLDS]]
        elif _fold_sizes is not None and "y_true_all" in d and "y_proba_all" in d:
            # Reconstruir fold MCC a partir de predicciones y tamaños de fold
            from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score
            yt_all = np.array(d["y_true_all"])
            yp_all = np.array(d["y_proba_all"])
            # Tomar solo folds 1-6 (descartar fold 7)
            n_stable = sum(_fold_sizes)
            yt_all = yt_all[:n_stable]
            yp_all = yp_all[:n_stable]
            fold_mccs, fold_accs, fold_aucs = [], [], []
            idx = 0
            for n in _fold_sizes:
                yt = yt_all[idx:idx+n]
                yp_prob = yp_all[idx:idx+n]
                yp_pred = (yp_prob > 0.5).astype(int)
                fold_mccs.append(float(matthews_corrcoef(yt, yp_pred)))
                fold_accs.append(float(accuracy_score(yt, yp_pred)))
                fold_aucs.append(float(roc_auc_score(yt, yp_prob)))
                idx += n
            log.info(f"  {key}: fold MCC reconstruido desde y_proba_all")
        else:
            fold_mccs = [None] * N_STABLE_FOLDS
            fold_accs = [None] * N_STABLE_FOLDS
            fold_aucs = [None] * N_STABLE_FOLDS

        # ── Summary stable (folds 1-6) ──
        if "summary_stable" in d:
            summary = d["summary_stable"]
        else:
            # Calcular desde fold_metrics si disponible
            if all(v is not None for v in fold_mccs):
                summary = {
                    "accuracy_mean": float(np.mean(fold_accs)),
                    "accuracy_std":  float(np.std(fold_accs)),
                    "mcc_mean":      float(np.mean(fold_mccs)),
                    "mcc_std":       float(np.std(fold_mccs)),
                    "roc_auc_mean":  float(np.mean(fold_aucs)),
                    "roc_auc_std":   float(np.std(fold_aucs)),
                }
            else:
                summary = {}

        # ── Predicciones (folds 1-6, excluir fold 7) ──
        y_true  = np.array(d.get("y_true_all",  []))
        y_proba = np.array(d.get("y_proba_all", []))

        # Ajuste de n_fold7 para CNN-LSTM (seq_len reduce muestras por fold)
        if key == "CNN_LSTM":
            seq_len = d.get("best_seq_len", 5)
            n_fold7_adj = max(n_fold7 - seq_len + 1, 0)
        else:
            n_fold7_adj = n_fold7

        if n_fold7_adj > 0 and len(y_true) > n_fold7_adj:
            y_true  = y_true[:-n_fold7_adj]
            y_proba = y_proba[:-n_fold7_adj]

        data[key] = {
            "fold_mccs":      fold_mccs,
            "fold_accs":      fold_accs,
            "fold_aucs":      fold_aucs,
            "summary_stable": summary,
            "y_true":         y_true,
            "y_proba":        y_proba,
        }
        log.info(f"  {key}: MCC={np.mean([m for m in fold_mccs if m]):.4f} | "
                 f"n_pred={len(y_true)}")

    return data


log.info("Cargando datos de modelos...")
data = load_model_data()

# Orden para los graficos (de mejor a peor MCC)
model_order = [k for k in MODEL_META if k in data]


# ─────────────────────────────────────────────────────────────────────────────
# Fig A — Model comparison bar chart (folds 1-6)
# ─────────────────────────────────────────────────────────────────────────────
log.info("\n[Fig A] Model comparison bar chart")

metrics_to_plot = ["accuracy_mean", "mcc_mean", "roc_auc_mean"]
metric_labels   = ["Accuracy", "MCC", "AUC-ROC"]
metric_stds     = ["accuracy_std",  "mcc_std",  "roc_auc_std"]

n_models  = len(model_order)
x         = np.arange(len(metrics_to_plot))
width     = 0.14
fig, ax   = plt.subplots(figsize=(COL2, 3.2))

for i, key in enumerate(model_order):
    s      = data[key]["summary_stable"]
    vals   = [s.get(m, 0) for m in metrics_to_plot]
    errs   = [s.get(e, 0) for e in metric_stds]
    offset = (i - n_models / 2 + 0.5) * width
    ax.bar(x + offset, vals, width,
           label=MODEL_META[key]["label"],
           color=MODEL_META[key]["color"], alpha=0.88,
           yerr=errs, capsize=3, error_kw={"linewidth": 1})

ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.10)
ax.legend(loc="upper left", ncol=2)
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)

for bar_group in ax.containers:
    if isinstance(bar_group, mcontainer.BarContainer):
        ax.bar_label(bar_group, fmt="%.3f", fontsize=6, padding=2, rotation=90)

plt.tight_layout()
save_fig(fig, "fig_model_comparison_bar")
shutil.copy(PLOTS_DIR / "fig_model_comparison_bar.pdf", PAPER_FIGS_DIR / "model_comparison.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig B — Per-fold MCC (folds 1-6 solamente)
# ─────────────────────────────────────────────────────────────────────────────
log.info("[Fig B] Per-fold MCC chart (folds 1-6)")

fig, ax = plt.subplots(figsize=(COL2, 2.8))

for key in model_order:
    mccs  = [m for m in data[key]["fold_mccs"] if m is not None]
    x_vals = list(range(len(mccs)))
    meta   = MODEL_META[key]
    ax.plot(x_vals, mccs, marker=meta["marker"], color=meta["color"],
            label=meta["label"], linewidth=1.0, markersize=4)

ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.set_xticks(range(N_STABLE_FOLDS))
ax.set_xticklabels(FOLD_LABELS)
ax.set_xlabel("Test Year (Fold)")
ax.set_ylabel("MCC")
ax.legend(loc="lower left", ncol=2)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
save_fig(fig, "fig_per_fold_mcc")
shutil.copy(PLOTS_DIR / "fig_per_fold_mcc.pdf", PAPER_FIGS_DIR / "perfold_mcc.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig C — ROC curves (folds 1-6 concatenados)
# ─────────────────────────────────────────────────────────────────────────────
log.info("[Fig C] ROC curves (folds 1-6)")

fig, ax = plt.subplots(figsize=(2.36, 2.36))  # 60 x 60 mm
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)

for key in model_order:
    y_true  = data[key]["y_true"]
    y_proba = data[key]["y_proba"]
    if len(y_true) == 0:
        continue
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_val     = auc(fpr, tpr)
    meta        = MODEL_META[key]
    ax.plot(fpr, tpr, color=meta["color"], linewidth=1.0,
            label=meta['label'])

ax.set_xlabel("False Positive Rate", fontsize=8)
ax.set_ylabel("True Positive Rate", fontsize=8)
ax.tick_params(axis="both", labelsize=7)
ax.legend(loc="lower right", fontsize=7)
ax.grid(alpha=0.3)
plt.tight_layout()
save_fig(fig, "fig_roc_curves")
shutil.copy(PLOTS_DIR / "fig_roc_curves.pdf", PAPER_FIGS_DIR / "roc_curves.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig D — Wilcoxon p-value heatmap
# ─────────────────────────────────────────────────────────────────────────────
log.info("[Fig D] Wilcoxon heatmap")

stats_path = STATS_DIR / "wilcoxon_mcc_folds1_6.csv"
if stats_path.exists():
    wilcoxon_df  = pd.read_csv(stats_path)
    labels_map   = {k: MODEL_META[k]["label"] for k in model_order}
    model_labels = [labels_map[k] for k in model_order]
    n = len(model_order)
    pval_matrix  = np.full((n, n), np.nan)

    for _, row in wilcoxon_df.iterrows():
        if row["model_A"] in model_order and row["model_B"] in model_order:
            i = model_order.index(row["model_A"])
            j = model_order.index(row["model_B"])
            pval_matrix[i, j] = row["pvalue_bonferroni"]
            pval_matrix[j, i] = row["pvalue_bonferroni"]

    fig, ax = plt.subplots(figsize=(COL2*0.8, COL2*0.8))
    mask    = np.eye(n, dtype=bool)
    annot   = np.full((n, n), "", dtype=object)
    for i in range(n):
        for j in range(n):
            if i != j and not np.isnan(pval_matrix[i, j]):
                p = pval_matrix[i, j]
                annot[i, j] = f"{p:.3f}" + ("*" if p < 0.05 else "")

    sns.heatmap(pval_matrix, ax=ax, mask=mask, annot=annot, fmt="",
                cmap="RdYlGn_r", vmin=0, vmax=0.2,
                xticklabels=model_labels, yticklabels=model_labels,
                linewidths=0.5,
                cbar_kws={"label": "Bonferroni-adjusted p-value"})
    plt.xticks(rotation=30, ha="right")
    plt.yticks()
    plt.tight_layout()
    save_fig(fig, "fig_wilcoxon_heatmap")
else:
    log.warning("  Wilcoxon CSV no encontrado — ejecuta run_paper_stats.py primero")


# ─────────────────────────────────────────────────────────────────────────────
# Tabla resumen (CSV + LaTeX-ready)
# ─────────────────────────────────────────────────────────────────────────────
log.info("\n[Table] Tabla de resultados")

table_rows = []
for key in model_order:
    s = data[key]["summary_stable"]
    if not s:
        continue
    table_rows.append({
        "Model":   MODEL_META[key]["label"],
        "Acc":     f"{s['accuracy_mean']:.3f} ± {s['accuracy_std']:.3f}",
        "MCC":     f"{s['mcc_mean']:.3f} ± {s['mcc_std']:.3f}",
        "AUC":     f"{s['roc_auc_mean']:.3f} ± {s['roc_auc_std']:.3f}",
        "MCC_num": s["mcc_mean"],
    })

table_df = (pd.DataFrame(table_rows)
              .sort_values("MCC_num", ascending=False)
              .drop("MCC_num", axis=1))
table_df.to_csv(PLOTS_DIR / "results_table.csv", index=False)

log.info("\nResults table (folds 1-6, 2019-2024):")
log.info(table_df.to_string(index=False))

log.info(f"\nTodas las figuras guardadas en {PLOTS_DIR}")
log.info("✓ Plots complete.")
