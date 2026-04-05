"""
All figure generation for the paper.
Saves PDF (vector, for publication) and PNG (300 dpi, for quick review).

Figure inventory:
  Fig 1  — Walk-forward validation diagram
  Fig 2  — Dataset overview: target distribution by year
  Fig 3  — Ablation heatmap: accuracy / F1 / MCC across model × feature group
  Fig 4  — ROC curves for all 13 experiments
  Fig 5  — CNN-LSTM and DualEncoder learning curves
  Fig 6  — SHAP summary plot (best model, G3)
  Fig 7  — SHAP beeswarm (top 15 features)
  Fig 8  — Confusion matrices (best model per feature group)
  Fig 9  — DualEncoder attention weight heatmap (averaged over test set)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PALETTE = sns.color_palette("colorblind", 8)
MODEL_COLORS = {
    "RandomForest": PALETTE[0],
    "XGBoost":      PALETTE[1],
    "CNN-LSTM":     PALETTE[2],
    "DualEncoder":  PALETTE[3],
}
GROUP_MARKERS = {"G0": "o", "G1": "s", "G2": "^", "G3": "D"}


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf")
    fig.savefig(output_dir / f"{name}.png")
    plt.close(fig)


# ─────────────────────────────────────────────
# Fig 1 — Walk-forward validation diagram
# ─────────────────────────────────────────────
def plot_walk_forward_diagram(folds: list, output_dir: Path,
                              df: pd.DataFrame = None) -> None:
    # Elsevier specs: Arial 7 pt, double-column 190 mm = 7.48 in
    FONT      = "Arial"
    FS_LABEL  = 7    # axis labels / tick labels
    FS_BAR    = 6    # text inside bars (min allowed)
    FS_YTICK  = 7
    COL2_IN   = 7.48

    # Excluir Fold 7 (2025, out-of-distribution)
    folds = [f for f in folds if f.fold != 7]

    colors        = {"train": "#4878CF", "test": "#E24A33"}
    bar_height    = 0.30          # barras estrechas
    y_step        = 0.55          # separación entre filas (< 1 = más juntas)
    dataset_start = folds[0].train_start
    x_end_days    = (pd.Timestamp("2025-01-01") - dataset_start).days

    # Posiciones Y: de arriba a abajo con paso y_step
    y_positions = [(len(folds) - i) * y_step for i in range(len(folds))]

    # ── Layout: barras arriba, precio Bitcoin abajo ────────────────────────
    top_height = y_step * len(folds) + 0.3   # alto del panel de barras en unidades de datos
    with_price = df is not None and "Close" in df.columns
    if with_price:
        fig, (ax, ax_price) = plt.subplots(
            2, 1, figsize=(COL2_IN, 3.2),
            gridspec_kw={"height_ratios": [2.2, 1.2], "hspace": 0.06},
            sharex=True,
        )
    else:
        fig, ax = plt.subplots(figsize=(COL2_IN, 2.0))
        ax_price = None

    # ── Barras de folds ────────────────────────────────────────────────────
    for i, fold in enumerate(folds):
        y           = y_positions[i]
        train_start = fold.train_start
        train_end   = fold.train_end
        test_start  = fold.test_start
        test_end    = fold.test_end
        train_width = (train_end  - train_start).days
        test_width  = (test_end   - test_start).days
        origin      = (train_start - dataset_start).days

        ax.barh(y, train_width, left=origin, height=bar_height,
                color=colors["train"], alpha=0.25,
                label="Training set" if i == 0 else "")
        ax.barh(y, test_width, left=origin + train_width, height=bar_height,
                color=colors["test"], alpha=0.25,
                label="Test set" if i == 0 else "")

        # Etiqueta en barra de train: rango completo de años
        train_label = f"{train_start.year}–{train_end.year}"
        ax.text(origin + train_width / 2, y, train_label,
                ha="center", va="center", fontsize=FS_BAR,
                fontfamily=FONT, color="black", fontweight="bold")

        # Etiqueta en barra de test: solo el año de test
        ax.text(origin + train_width + test_width / 2, y,
                str(test_start.year),
                ha="center", va="center", fontsize=FS_BAR,
                fontfamily=FONT, color="black", fontweight="bold")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Fold {f.fold}" for f in folds],
                       fontsize=FS_YTICK, fontfamily=FONT)
    ax.legend(loc="upper right", fontsize=FS_LABEL,
              framealpha=0.85, edgecolor="none")
    ax.set_xlim(0, x_end_days)
    margin = bar_height * 0.6
    ax.set_ylim(min(y_positions) - margin, max(y_positions) + margin)
    ax.tick_params(axis="both", labelsize=FS_LABEL)
    ax.set_ylabel("Fold", fontsize=FS_LABEL, fontfamily=FONT)
    ax.grid(axis="x", alpha=0.0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # ── Precio Bitcoin ─────────────────────────────────────────────────────
    if ax_price is not None:
        price_data = df[df.index < pd.Timestamp("2025-01-01")]["Close"].dropna()
        days = [(d - dataset_start).days for d in price_data.index]
        ax_price.semilogy(days, price_data.values,
                          color="#F7931A", lw=1.0, alpha=0.9)
        ax_price.set_ylabel("BTC/USD (log)", fontsize=FS_LABEL, fontfamily=FONT)
        ax_price.set_ylim(bottom=price_data.min() * 0.8)
        ax_price.tick_params(axis="both", labelsize=FS_LABEL)
        ax_price.grid(axis="y", ls=":", alpha=0.3)
        ax_price.set_xlabel("Year", fontsize=FS_LABEL, fontfamily=FONT)
        for spine in ("top", "right"):
            ax_price.spines[spine].set_visible(False)

    # ── Líneas verticales anuales + ticks ──────────────────────────────────
    tick_positions, tick_labels = [], []
    for yr in range(dataset_start.year, 2026):
        d = (pd.Timestamp(f"{yr}-01-01") - dataset_start).days
        if d > x_end_days:
            break
        tick_positions.append(d)
        tick_labels.append(str(yr))
        for a in ([ax, ax_price] if ax_price is not None else [ax]):
            a.axvline(d, color="grey", lw=0.6, ls=":", zorder=0, alpha=0.55)

    ax_bottom = ax_price if ax_price is not None else ax
    ax_bottom.set_xticks(tick_positions)
    ax_bottom.set_xticklabels(tick_labels, rotation=45, ha="right",
                               fontsize=FS_LABEL, fontfamily=FONT)
    if ax_price is None:
        ax.set_xlabel("Year", fontsize=FS_LABEL, fontfamily=FONT)

    fig.savefig(output_dir / "fig1_walk_forward_diagram.pdf",
                dpi=1000, bbox_inches="tight")
    fig.savefig(output_dir / "fig1_walk_forward_diagram.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────
# Fig 2 — Dataset overview
# ─────────────────────────────────────────────
def plot_dataset_overview(df: pd.DataFrame, target_col: str, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Price series with market cycles
    ax = axes[0]
    ax.semilogy(df.index, df["Close"], color=PALETTE[0], linewidth=0.8, label="BTC/USD")
    ax.set_title("Bitcoin Daily Price (log scale)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(alpha=0.2)

    # Target distribution by year
    ax = axes[1]
    yearly = (
        df.groupby(df.index.year)[target_col]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    if 1 in yearly.columns:
        ax.bar(yearly.index, yearly[1], color=PALETTE[1], label="Up (y=1)", alpha=0.8)
    if 0 in yearly.columns:
        ax.bar(yearly.index, yearly[0], bottom=yearly.get(1, 0),
               color=PALETTE[2], label="Down (y=0)", alpha=0.8)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label="50%")
    ax.set_title("Daily Direction Distribution by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Proportion")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    _save(fig, output_dir, "fig2_dataset_overview")


# ─────────────────────────────────────────────
# Fig 3 — Ablation heatmap
# ─────────────────────────────────────────────
def plot_ablation_heatmap(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    summary_df must have columns: model, group, accuracy_mean, f1_macro_mean, mcc_mean
    """
    metrics = ["accuracy_mean", "f1_macro_mean", "mcc_mean"]
    metric_labels = ["Accuracy", "F1 Macro", "MCC"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, metric, label in zip(axes, metrics, metric_labels):
        pivot = summary_df.pivot(index="model", columns="group", values=metric)
        sns.heatmap(
            pivot,
            ax=ax,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            vmin=0.45,
            vmax=0.75,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(label)
        ax.set_xlabel("Feature Group")
        ax.set_ylabel("Model" if ax == axes[0] else "")

    plt.suptitle("Ablation Study: Mean Metric across Walk-Forward Folds", y=1.02, fontsize=13)
    plt.tight_layout()
    _save(fig, output_dir, "fig3_ablation_heatmap")


# ─────────────────────────────────────────────
# Fig 4 — ROC curves
# ─────────────────────────────────────────────
def plot_roc_curves(
    experiments: dict,  # {exp_id: {"y_true_all": arr, "y_proba_all": arr, "model": str, "group": str}}
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    for exp_id, data in experiments.items():
        y_true = data["y_true_all"]
        y_proba = data["y_proba_all"]
        model = data["model"]
        group = data["group"]

        if len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        color = MODEL_COLORS.get(model, "gray")
        ls = {"G0": "-", "G1": "--", "G2": "-.", "G3": ":"}.get(group, "-")
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=1.5,
                label=f"{model}+{group} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Experiments (Concatenated Test Folds)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    _save(fig, output_dir, "fig4_roc_curves")


# ─────────────────────────────────────────────
# Fig 5 — Learning curves (PyTorch models)
# ─────────────────────────────────────────────
def plot_learning_curves(
    histories: dict,  # {model_name: [{"epoch", "train_loss", "val_loss"}, ...]}
    output_dir: Path,
) -> None:
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    axes = axes[0]

    for ax, (model_name, history) in zip(axes, histories.items()):
        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        ax.plot(epochs, train_loss, label="Train loss", color=PALETTE[0])
        if "val_loss" in history[0]:
            val_loss = [h["val_loss"] for h in history]
            ax.plot(epochs, val_loss, label="Val loss", color=PALETTE[1], linestyle="--")
        if "val_acc" in history[0]:
            ax2 = ax.twinx()
            val_acc = [h["val_acc"] for h in history]
            ax2.plot(epochs, val_acc, label="Val acc", color=PALETTE[2], linestyle=":", alpha=0.7)
            ax2.set_ylabel("Validation Accuracy")
            ax2.set_ylim(0.4, 0.9)
        ax.set_title(f"{model_name} — Learning Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.2)

    plt.tight_layout()
    _save(fig, output_dir, "fig5_learning_curves")


# ─────────────────────────────────────────────
# Fig 6 — SHAP summary (bar chart)
# ─────────────────────────────────────────────
def plot_shap_summary(
    feature_importance: list[tuple[str, float]],  # [(name, mean_abs_shap), ...]
    model_name: str,
    group_name: str,
    output_dir: Path,
    top_n: int = 20,
) -> None:
    fi = feature_importance[:top_n]
    names = [f[0] for f in fi]
    values = [f[1] for f in fi]

    fig, ax = plt.subplots(figsize=(8, 0.4 * top_n + 1.5))
    bars = ax.barh(range(len(names)), values[::-1], color=PALETTE[0], alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature Importance — {model_name} on {group_name} (Top {top_n})")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    _save(fig, output_dir, f"fig6_shap_summary_{model_name}_{group_name}")


# ─────────────────────────────────────────────
# Fig 7 — SHAP beeswarm (top 15)
# ─────────────────────────────────────────────
def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    top_n: int = 15,
) -> None:
    import shap as shap_lib
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_n = min(top_n, len(feature_names))
    top_idx = np.argsort(mean_abs)[::-1][:top_n]

    sv_top = shap_values[:, top_idx]
    X_top = X_explain[:, top_idx]
    names_top = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(8, 0.4 * top_n + 1.5))
    # Manual beeswarm-style plot
    for j, name in enumerate(names_top):
        sv_col = sv_top[:, j]
        x_col = X_top[:, j]
        # Normalize feature values to [0,1] for colouring
        x_norm = (x_col - x_col.min()) / ((x_col.max() - x_col.min()) + 1e-10)
        colors = plt.cm.RdBu_r(x_norm)
        # Add jitter on y-axis
        jitter = np.random.default_rng(42).uniform(-0.2, 0.2, size=len(sv_col))
        ax.scatter(sv_col, top_n - 1 - j + jitter, c=colors, s=4, alpha=0.6)

    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(list(reversed(names_top)))
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title(f"SHAP Beeswarm — Top {top_n} Features")
    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    _save(fig, output_dir, "fig7_shap_beeswarm")


# ─────────────────────────────────────────────
# Fig 8 — Confusion matrices
# ─────────────────────────────────────────────
def plot_confusion_matrices(
    cms: dict,  # {exp_id: np.ndarray (2,2)}
    output_dir: Path,
) -> None:
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    axes = axes[0]

    for ax, (exp_id, cm) in zip(axes, cms.items()):
        sns.heatmap(
            cm,
            ax=ax,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Down (0)", "Up (1)"],
            yticklabels=["Down (0)", "Up (1)"],
            cbar=False,
        )
        ax.set_title(exp_id, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual" if ax == axes[0] else "")

    plt.suptitle("Confusion Matrices — Best Model per Feature Group", y=1.02)
    plt.tight_layout()
    _save(fig, output_dir, "fig8_confusion_matrices")


# ─────────────────────────────────────────────
# Fig 9 — DualEncoder attention weights
# ─────────────────────────────────────────────
def plot_attention_weights(
    tech_weights: np.ndarray,
    onchain_weights: np.ndarray,
    seq_len: int,
    output_dir: Path,
) -> None:
    """
    Visualise mean cross-attention weights averaged over the test set.
    tech_weights   : (n_samples, seq_len, seq_len) — tech→onchain attention
    onchain_weights: (n_samples, seq_len, seq_len) — onchain→tech attention
    """
    mean_tech = tech_weights.mean(axis=0)       # (seq_len, seq_len)
    mean_onchain = onchain_weights.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, weights, title in zip(
        axes,
        [mean_tech, mean_onchain],
        ["Technical → On-Chain Attention", "On-Chain → Technical Attention"],
    ):
        sns.heatmap(weights, ax=ax, cmap="viridis", cbar=True,
                    xticklabels=5, yticklabels=5)
        ax.set_title(title)
        ax.set_xlabel("Key (days ago)")
        ax.set_ylabel("Query (days ago)")

    plt.suptitle("DualEncoder Cross-Attention Weights (Mean over Test Set)", y=1.02)
    plt.tight_layout()
    _save(fig, output_dir, "fig9_attention_weights")
