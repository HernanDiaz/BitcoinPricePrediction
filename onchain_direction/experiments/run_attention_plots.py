#!/usr/bin/env python3
"""
run_attention_plots.py
======================
Genera las figuras de interpretabilidad de cross-attention a partir de los
datos guardados por run_attention_analysis.py.

Figuras generadas:
  Fig A — Concentracion de cross-attention por fold (bar chart agrupado)
           Muestra estabilidad temporal de la atención bidireccional.
  Fig B — Concentracion segun correctitud de prediccion (violin/box plot)
           Predicciones correctas vs incorrectas — valida que atención
           focalizada correlaciona con acierto.
  Fig C — Concentracion segun dirección predicha: UP vs DOWN (grouped bars)
           Muestra asimetria: el modelo usa señales distintas para subida/bajada.

Las figuras se guardan en results/plots/.
"""

import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Elsevier figure style ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         7,
    "axes.labelsize":    7,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "legend.framealpha": 0.85,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
COL1 = 3.54   # single column 90 mm
COL2 = 7.48   # double column 190 mm
from scipy import stats

ROOT = Path(__file__).parents[2]
attn_dir  = ROOT / "results" / "attention"
plots_dir = ROOT / "results" / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# ── Cargar datos ──────────────────────────────────────────────────────────────
with open(attn_dir / "attention_fold_summaries.json") as f:
    fold_data = json.load(f)

with open(attn_dir / "attention_sample_records.json") as f:
    records = json.load(f)

fold_summaries = fold_data["fold_summaries"]
K = fold_data["K"]

# Arrays globales
conc_to = np.array([r["conc_to"] for r in records])   # tech→onchain
conc_ot = np.array([r["conc_ot"] for r in records])   # onchain→tech
correct = np.array([r["correct"] for r in records], dtype=bool)
pred_up = np.array([r["y_pred"]  for r in records]) == 1
y_true  = np.array([r["y_true"]  for r in records])
folds   = np.array([r["fold"]    for r in records])

COLORS  = {"to": "#7EB6D4", "ot": "#E8A87C",
           "correct": "#6DB88A", "wrong": "#D97070",
           "up": "#6DB88A", "down": "#D97070"}

# ─────────────────────────────────────────────────────────────────────────────
# FIG A — Concentracion por fold
# ─────────────────────────────────────────────────────────────────────────────
fold_nums  = [fs["fold"]         for fs in fold_summaries]
fold_years = [fs["year"]         for fs in fold_summaries]
mean_to    = [fs["mean_conc_to"] for fs in fold_summaries]
std_to     = [fs["std_conc_to"]  for fs in fold_summaries]
mean_ot    = [fs["mean_conc_ot"] for fs in fold_summaries]
std_ot     = [fs["std_conc_ot"]  for fs in fold_summaries]

x = np.arange(len(fold_years))
w = 0.35

fig, ax = plt.subplots(figsize=(COL1, 2.0))
bars_to = ax.bar(x - w/2, mean_to, w, yerr=std_to,
                  color=COLORS["to"], alpha=0.85, capsize=3,
                  label="Tech→On-chain", error_kw=dict(elinewidth=0.8))
bars_ot = ax.bar(x + w/2, mean_ot, w, yerr=std_ot,
                  color=COLORS["ot"], alpha=0.85, capsize=3,
                  label="On-chain→Tech", error_kw=dict(elinewidth=0.8))

ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in fold_years])
ax.set_xlabel("Test Year (Fold)")
ax.set_ylabel("Mean Attention Concentration\n(1 − H / log K)")
ax.legend()
ax.set_ylim(0, 1)
ax.axhline(0.5, ls="--", color="gray", lw=0.8, alpha=0.6, label="Random (uniform)")
ax.grid(axis="y", alpha=0.3, ls=":")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig.savefig(plots_dir / "attn_concentration_per_fold.pdf", bbox_inches="tight")
fig.savefig(plots_dir / "attn_concentration_per_fold.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("Guardado: attn_concentration_per_fold")

# ─────────────────────────────────────────────────────────────────────────────
# FIG B — Correcto vs Incorrecto
# ─────────────────────────────────────────────────────────────────────────────
groups_b = {
    "Correct\n(Tech→OC)"  : conc_to[correct],
    "Wrong\n(Tech→OC)"    : conc_to[~correct],
    "Correct\n(OC→Tech)"  : conc_ot[correct],
    "Wrong\n(OC→Tech)"    : conc_ot[~correct],
}
colors_b = [COLORS["correct"], COLORS["wrong"],
            COLORS["correct"], COLORS["wrong"]]

fig, ax = plt.subplots(figsize=(COL1, 2.5))
positions = [1, 2, 4, 5]

vp = ax.violinplot([v for v in groups_b.values()],
                    positions=positions, showmedians=True,
                    showextrema=False)

for patch, color in zip(vp["bodies"], colors_b):
    patch.set_facecolor(color)
    patch.set_alpha(0.65)
vp["cmedians"].set_color("black")
vp["cmedians"].set_linewidth(2)

# T-test anotaciones
t_to, p_to = stats.ttest_ind(conc_to[correct], conc_to[~correct])
t_ot, p_ot = stats.ttest_ind(conc_ot[correct], conc_ot[~correct])

def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

ymax = max(conc_to.max(), conc_ot.max())
ax.plot([1, 2], [ymax*1.05]*2, color="black", lw=1.2)
ax.text(1.5, ymax*1.07, f"p{sig_label(p_to)} ({p_to:.3f})",
        ha="center", fontsize=7)
ax.plot([4, 5], [ymax*1.05]*2, color="black", lw=1.2)
ax.text(4.5, ymax*1.07, f"p{sig_label(p_ot)} ({p_ot:.3f})",
        ha="center", fontsize=7)

ax.set_xticks(positions)
ax.set_xticklabels(list(groups_b.keys()))
ax.set_ylabel("Attention Concentration (1 − H / log K)")
ax.axvline(3, color="lightgray", lw=1.5, ls="--")
ax.text(3, ax.get_ylim()[0]*0.98, "│", ha="center", color="lightgray")

correct_patch = mpatches.Patch(color=COLORS["correct"], alpha=0.65, label="Correct")
wrong_patch   = mpatches.Patch(color=COLORS["wrong"],   alpha=0.65, label="Wrong")
ax.legend(handles=[correct_patch, wrong_patch], loc="upper right")
ax.grid(axis="y", alpha=0.3, ls=":")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig.savefig(plots_dir / "attn_correct_vs_wrong.pdf", bbox_inches="tight")
fig.savefig(plots_dir / "attn_correct_vs_wrong.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("Guardado: attn_correct_vs_wrong")

# ─────────────────────────────────────────────────────────────────────────────
# FIG C — UP vs DOWN predictions
# ─────────────────────────────────────────────────────────────────────────────
groups_c = {
    "UP\n(Tech→OC)"  : conc_to[pred_up],
    "DOWN\n(Tech→OC)": conc_to[~pred_up],
    "UP\n(OC→Tech)"  : conc_ot[pred_up],
    "DOWN\n(OC→Tech)": conc_ot[~pred_up],
}
colors_c = [COLORS["up"], COLORS["down"],
            COLORS["up"], COLORS["down"]]

t_to_ud, p_to_ud = stats.ttest_ind(conc_to[pred_up], conc_to[~pred_up])
t_ot_ud, p_ot_ud = stats.ttest_ind(conc_ot[pred_up], conc_ot[~pred_up])

fig, ax = plt.subplots(figsize=(COL1, 2.5))
vp2 = ax.violinplot([v for v in groups_c.values()],
                     positions=positions, showmedians=True,
                     showextrema=False)
for patch, color in zip(vp2["bodies"], colors_c):
    patch.set_facecolor(color)
    patch.set_alpha(0.65)
vp2["cmedians"].set_color("black")
vp2["cmedians"].set_linewidth(2)

ymax_c = max(conc_to.max(), conc_ot.max())
ax.plot([1, 2], [ymax_c*1.05]*2, color="black", lw=1.2)
ax.text(1.5, ymax_c*1.07, f"p{sig_label(p_to_ud)} ({p_to_ud:.3f})",
        ha="center", fontsize=7)
ax.plot([4, 5], [ymax_c*1.05]*2, color="black", lw=1.2)
ax.text(4.5, ymax_c*1.07, f"p{sig_label(p_ot_ud)} ({p_ot_ud:.3f})",
        ha="center", fontsize=7)

ax.set_xticks(positions)
ax.set_xticklabels(list(groups_c.keys()))
ax.set_ylabel("Attention Concentration (1 − H / log K)")
ax.axvline(3, color="lightgray", lw=1.5, ls="--")

up_patch   = mpatches.Patch(color=COLORS["up"],   alpha=0.65, label="Predicted UP")
down_patch = mpatches.Patch(color=COLORS["down"], alpha=0.65, label="Predicted DOWN")
ax.legend(handles=[up_patch, down_patch], loc="upper right")
ax.grid(axis="y", alpha=0.3, ls=":")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig.savefig(plots_dir / "attn_up_vs_down.pdf", bbox_inches="tight")
fig.savefig(plots_dir / "attn_up_vs_down.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("Guardado: attn_up_vs_down")

# ─────────────────────────────────────────────────────────────────────────────
# Guardar estadísticas resumen para el paper
# ─────────────────────────────────────────────────────────────────────────────
summary_stats = {
    "global": {
        "mean_conc_to": float(conc_to.mean()),
        "std_conc_to" : float(conc_to.std()),
        "mean_conc_ot": float(conc_ot.mean()),
        "std_conc_ot" : float(conc_ot.std()),
    },
    "correct_vs_wrong": {
        "correct_mean_to"  : float(conc_to[correct].mean()),
        "wrong_mean_to"    : float(conc_to[~correct].mean()),
        "ttest_to_pvalue"  : float(p_to),
        "ttest_to_stat"    : float(t_to),
        "correct_mean_ot"  : float(conc_ot[correct].mean()),
        "wrong_mean_ot"    : float(conc_ot[~correct].mean()),
        "ttest_ot_pvalue"  : float(p_ot),
        "ttest_ot_stat"    : float(t_ot),
    },
    "up_vs_down": {
        "up_mean_to"       : float(conc_to[pred_up].mean()),
        "down_mean_to"     : float(conc_to[~pred_up].mean()),
        "ttest_to_pvalue"  : float(p_to_ud),
        "ttest_to_stat"    : float(t_to_ud),
        "up_mean_ot"       : float(conc_ot[pred_up].mean()),
        "down_mean_ot"     : float(conc_ot[~pred_up].mean()),
        "ttest_ot_pvalue"  : float(p_ot_ud),
        "ttest_ot_stat"    : float(t_ot_ud),
    },
    "n_correct"  : int(correct.sum()),
    "n_wrong"    : int((~correct).sum()),
    "n_pred_up"  : int(pred_up.sum()),
    "n_pred_down": int((~pred_up).sum()),
    "K"          : int(K),
}

with open(attn_dir / "attention_summary_stats.json", "w") as f:
    json.dump(summary_stats, f, indent=2)

print("\n=== ESTADISTICAS FINALES ===")
print(f"  Concentracion media tech→onchain : {summary_stats['global']['mean_conc_to']:.4f}")
print(f"  Concentracion media onchain→tech : {summary_stats['global']['mean_conc_ot']:.4f}")
print(f"\n  Correcto  — conc to: {summary_stats['correct_vs_wrong']['correct_mean_to']:.4f}")
print(f"  Incorrecto — conc to: {summary_stats['correct_vs_wrong']['wrong_mean_to']:.4f}")
print(f"  t-test p = {p_to:.4f}  {sig_label(p_to)}")
print(f"\n  Pred UP   — conc to: {summary_stats['up_vs_down']['up_mean_to']:.4f}")
print(f"  Pred DOWN — conc to: {summary_stats['up_vs_down']['down_mean_to']:.4f}")
print(f"  t-test p = {p_to_ud:.4f}  {sig_label(p_to_ud)}")
print(f"\nGuardado: {attn_dir / 'attention_summary_stats.json'}")
