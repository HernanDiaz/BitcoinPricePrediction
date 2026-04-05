#!/usr/bin/env python3
"""
run_mc_dropout_v2.py
====================
Análisis de incertidumbre epistémica sobre DECA V2 mediante MC Dropout.

NO modifica mlp_dual_encoder_v2.py. La técnica es puramente de inferencia:
  - Reentrena DECA V2 con los mejores params de Optuna (reproducible)
  - En inferencia activa dropout (model.train()) y ejecuta N=30 passes
  - La varianza de las 30 predicciones = incertidumbre epistémica

Guarda todos los datos numéricos en results/mc_dropout/ antes de plotear.
Figuras en results/plots/.
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
from src.models.mlp_dual_encoder_v2 import MLPDualEncoderModelV2
import torch, yaml
from sklearn.metrics import matthews_corrcoef, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Elsevier figure style ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":        7,
    "axes.labelsize":   7,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "legend.framealpha": 0.85,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})
COL1 = 3.54   # single column 90 mm
COL2 = 7.48   # double column 190 mm

import matplotlib.patches as mpatches

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("mc_dropout_v2")

# ── Config ────────────────────────────────────────────────────────────────────
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
search_folds = all_folds[:-1]   # folds 1-6, excluye 2025

MC_SAMPLES = 30

out_dir    = ROOT / "results" / "mc_dropout"
plots_dir  = ROOT / "results" / "plots"
paper_figs = ROOT / "paper" / "EAAI" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
paper_figs.mkdir(parents=True, exist_ok=True)

# ── Mejores params V2 (ya encontrados por Optuna, no re-buscar) ───────────────
with open(ROOT / "results" / "optuna" / "mlp_dual_v2_optuna.json") as f:
    optuna_res = json.load(f)
best_params = {**optuna_res["best_params"], "epochs": 200, "patience": 30}
log.info(f"DECA V2 best_params cargados: {best_params}")
log.info(f"MC samples: {MC_SAMPLES}")


def split_branches(X):
    tc = [feat_cols.index(c) for c in tech_cols    if c in feat_cols]
    oc = [feat_cols.index(c) for c in onchain_cols if c in feat_cols]
    return X[:, tc], X[:, oc]


def mc_predict(model: MLPDualEncoderModelV2, X_tech, X_onchain,
               n_samples: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    MC Dropout inference SIN modificar el modelo.
    Activa model._net.train() para habilitar dropout durante inferencia.
    Devuelve (mean_p_up, std_p_up) ambos shape (N,).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    xt = torch.tensor(X_tech,   dtype=torch.float32)
    xo = torch.tensor(X_onchain, dtype=torch.float32)
    ds = TensorDataset(xt, xo)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_runs = []
    model._net.train()   # activa dropout — NO modifica el modelo, solo el modo
    with torch.no_grad():
        for _ in range(n_samples):
            run_probs = []
            for batch in loader:
                bt, bo = batch[0].to(model._device), batch[1].to(model._device)
                logits, _, _ = model._net(bt, bo)
                p = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                run_probs.append(p)
            all_runs.append(np.concatenate(run_probs))

    model._net.eval()    # restaura modo eval tras MC
    all_runs = np.stack(all_runs, axis=0)   # (n_samples, N)
    return all_runs.mean(axis=0), all_runs.std(axis=0)


# ── Extraccion por fold ───────────────────────────────────────────────────────
records = []

for fold_idx, fold in enumerate(search_folds):
    fold_year = 2019 + fold_idx
    log.info(f"\n{'='*55}")
    log.info(f"Fold {fold_idx+1} — Test {fold_year}")

    train_df, test_df = fold.split(df)
    prep = FoldPreprocessor(sequence_length=1)
    X_tr, _, y_tr, _ = prep.fit_transform(train_df, feat_cols, target_col)
    X_te, _, y_te, _ = prep.transform(test_df,  feat_cols, target_col)

    X_tr_t, X_tr_o = split_branches(X_tr)
    X_te_t, X_te_o = split_branches(X_te)

    model = MLPDualEncoderModelV2(
        cfg=best_params,
        n_technical=X_tr_t.shape[1],
        n_onchain=X_tr_o.shape[1],
        device=device, random_seed=seed,
    )
    model.fit((X_tr_t, X_tr_o), y_tr,
              class_weight=prep.compute_class_weight(y_tr))

    # Prediccion determinista (eval mode)
    probas_det = model.predict_proba((X_te_t, X_te_o))
    preds_det  = (probas_det[:, 1] >= 0.5).astype(int)
    mcc_det    = matthews_corrcoef(y_te, preds_det)

    # MC Dropout
    mean_p_up, uncertainty = mc_predict(
        model, X_te_t, X_te_o,
        n_samples=MC_SAMPLES,
        batch_size=best_params["batch_size"],
    )
    preds_mc = (mean_p_up >= 0.5).astype(int)
    mcc_mc   = matthews_corrcoef(y_te, preds_mc)

    log.info(f"  Det MCC={mcc_det:.4f}  MC MCC={mcc_mc:.4f}")
    log.info(f"  Uncertainty: mean={uncertainty.mean():.4f}  std={uncertainty.std():.4f}  "
             f"max={uncertainty.max():.4f}")

    for i in range(len(y_te)):
        records.append({
            "fold"       : fold_idx + 1,
            "year"       : fold_year,
            "y_true"     : int(y_te[i]),
            "y_pred_det" : int(preds_det[i]),
            "y_pred_mc"  : int(preds_mc[i]),
            "p_up_det"   : float(probas_det[i, 1]),
            "p_up_mc"    : float(mean_p_up[i]),
            "uncertainty": float(uncertainty[i]),
            "correct_det": int(y_te[i] == preds_det[i]),
            "correct_mc" : int(y_te[i] == preds_mc[i]),
        })

# ── Guardar datos ─────────────────────────────────────────────────────────────
with open(out_dir / "mc_dropout_records.json", "w") as f:
    json.dump(records, f, indent=2)
log.info(f"\nGuardados {len(records)} registros → {out_dir / 'mc_dropout_records.json'}")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS — se generan a partir de los datos guardados
# ═══════════════════════════════════════════════════════════════════════════════

uncert  = np.array([r["uncertainty"]  for r in records])
correct = np.array([r["correct_det"]  for r in records], dtype=bool)
p_up    = np.array([r["p_up_mc"]      for r in records])
folds   = np.array([r["fold"]         for r in records])
years   = np.array([r["year"]         for r in records])

# ── FIG 1: Accuracy vs umbral de incertidumbre ────────────────────────────────
thresholds = np.percentile(uncert, np.linspace(10, 100, 40))
thresholds = np.unique(thresholds)
coverages, accuracies = [], []
for thr in thresholds:
    mask = uncert <= thr
    if mask.sum() >= 20:
        coverages.append(float(mask.mean()))
        accuracies.append(float(correct[mask].mean()))

baseline_acc = correct.mean()

fig, ax1 = plt.subplots(figsize=(COL1, COL1))
color_acc = "#7EB6D4"
color_cov = "#D97070"

ax1.plot(coverages, accuracies, color=color_acc, lw=1.2,
         marker="o", ms=4, label="Accuracy (retained samples)")
ax1.axhline(baseline_acc, ls="--", color=color_acc, alpha=0.45, lw=1.0,
             label=f"Baseline acc = {baseline_acc:.3f} (all samples)")
ax1.set_xlabel("Coverage (fraction of samples retained)")
ax1.set_ylabel("Accuracy")
ax1.tick_params(axis="y", labelcolor="black")
ax1.set_xlim(0, 1.02)
ax1.set_ylim(baseline_acc - 0.02, None)
ax1.grid(alpha=0.3, ls=":")
ax1.spines[["top"]].set_visible(False)

lines1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(lines1, labels1, loc="lower right")
plt.tight_layout()
fig.savefig(plots_dir / "mc_accuracy_vs_coverage.pdf", bbox_inches="tight", dpi=300)
fig.savefig(plots_dir / "mc_accuracy_vs_coverage.png", bbox_inches="tight", dpi=300)
import shutil; shutil.copy(plots_dir / "mc_accuracy_vs_coverage.pdf", paper_figs / "mc_accuracy_vs_coverage.pdf")
plt.close(fig)
log.info("Guardado: mc_accuracy_vs_coverage")

# ── FIG 2: Distribución de incertidumbre por fold ─────────────────────────────
fold_nums  = sorted(set(r["fold"] for r in records))
fold_years_list = [2018 + f for f in fold_nums]
fold_uncert = [uncert[folds == f] for f in fold_nums]

fig, ax = plt.subplots(figsize=(COL1, 2.0))
vp = ax.violinplot(fold_uncert, positions=fold_nums,
                   showmedians=True, showextrema=False)
for patch in vp["bodies"]:
    patch.set_facecolor("#7EB6D4")
    patch.set_alpha(0.7)
vp["cmedians"].set_color("black")
vp["cmedians"].set_linewidth(1.2)
ax.set_xticks(fold_nums)
ax.set_xticklabels([str(y) for y in fold_years_list])
ax.set_xlabel("Test Year (Fold)")
ax.set_ylabel("Epistemic Uncertainty σ\n(MC Dropout, 30 passes)")
ax.grid(axis="y", alpha=0.3, ls=":")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(plots_dir / "mc_uncertainty_per_fold.pdf", bbox_inches="tight", dpi=300)
fig.savefig(plots_dir / "mc_uncertainty_per_fold.png", bbox_inches="tight", dpi=300)
plt.close(fig)
log.info("Guardado: mc_uncertainty_per_fold")

# ── FIG 3: Uncertainty correcto vs incorrecto ─────────────────────────────────
from scipy import stats as scipy_stats
u_correct = uncert[correct]
u_wrong   = uncert[~correct]
t_stat, p_val = scipy_stats.ttest_ind(u_correct, u_wrong)

fig, ax = plt.subplots(figsize=(COL1*0.7, COL1*0.9))
vp2 = ax.violinplot([u_correct, u_wrong], positions=[1, 2],
                    showmedians=True, showextrema=False)
colors2 = ["#6DB88A", "#D97070"]
for patch, c in zip(vp2["bodies"], colors2):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
vp2["cmedians"].set_color("black")
vp2["cmedians"].set_linewidth(1.2)

# Anotacion p-valor
ymax = max(u_correct.max(), u_wrong.max())
ax.plot([1, 2], [ymax * 1.05] * 2, color="black", lw=1.2)
sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
ax.text(1.5, ymax * 1.07, f"p {sig} ({p_val:.4f})", ha="center", fontsize=7)

ax.set_xticks([1, 2])
ax.set_xticklabels(["Correct predictions\n(n={:,})".format(correct.sum()),
                    "Wrong predictions\n(n={:,})".format((~correct).sum())])
ax.set_ylabel("Epistemic Uncertainty σ")
ax.grid(axis="y", alpha=0.3, ls=":")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(plots_dir / "mc_uncertainty_correct_vs_wrong.pdf", bbox_inches="tight", dpi=300)
fig.savefig(plots_dir / "mc_uncertainty_correct_vs_wrong.png", bbox_inches="tight", dpi=300)
shutil.copy(plots_dir / "mc_uncertainty_correct_vs_wrong.pdf", paper_figs / "mc_uncertainty_correct_vs_wrong.pdf")
plt.close(fig)
log.info("Guardado: mc_uncertainty_correct_vs_wrong")

# ── Estadísticas resumen ──────────────────────────────────────────────────────
stats_out = {
    "baseline_accuracy"         : float(baseline_acc),
    "mc_samples"                : MC_SAMPLES,
    "mean_uncertainty"          : float(uncert.mean()),
    "std_uncertainty"           : float(uncert.std()),
    "correct_mean_uncertainty"  : float(u_correct.mean()),
    "wrong_mean_uncertainty"    : float(u_wrong.mean()),
    "ttest_pvalue"              : float(p_val),
    "ttest_stat"                : float(t_stat),
    "acc_top90pct_confidence"   : float(correct[uncert <= np.percentile(uncert, 90)].mean()),
    "acc_top75pct_confidence"   : float(correct[uncert <= np.percentile(uncert, 75)].mean()),
    "acc_top50pct_confidence"   : float(correct[uncert <= np.percentile(uncert, 50)].mean()),
    "coverage_90pct"            : 0.90,
    "coverage_75pct"            : 0.75,
    "coverage_50pct"            : 0.50,
}
with open(out_dir / "mc_dropout_summary.json", "w") as f:
    json.dump(stats_out, f, indent=2)

log.info("\n=== RESUMEN FINAL ===")
log.info(f"  Baseline accuracy (todos los dias)   : {baseline_acc:.4f}")
log.info(f"  Incertidumbre media                  : {uncert.mean():.4f} ± {uncert.std():.4f}")
log.info(f"  Uncertainty correcto vs incorrecto   : {u_correct.mean():.4f} vs {u_wrong.mean():.4f}  p={p_val:.4f} {sig}")
for pct, label in [(90, "90%"), (75, "75%"), (50, "50%")]:
    mask = uncert <= np.percentile(uncert, pct)
    acc  = correct[mask].mean()
    log.info(f"  Acc reteniendo {label} más seguros ({mask.sum():4d} dias): {acc:.4f}  "
             f"(+{acc - baseline_acc:+.4f})")
log.info(f"\nGuardado en {out_dir}")
