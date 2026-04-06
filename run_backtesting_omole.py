"""
Backtesting — Omole (2025) methodology replication.

Strategy: Long/Short (same as Omole & Enke 2025, EAAI).
  - Predict UP  (proba > 0.5) → go LONG  (hold BTC)
  - Predict DOWN (proba <= 0.5) → go SHORT (short BTC)

Parameters matching Omole (2025):
  - Commission: 0.5% per trade (one-way), applied on position change
  - Tax: 30% on each REALISED profit when closing a position
  - Starting capital: $1,000
  - Compounding: yes (reinvest after each trade)

Annual ROR formula (Omole eq. 8):
  Annual ROR = ((End / Start)^(365 / backtest_days)) - 1

We additionally report Sharpe, MaxDD, WinRate for completeness.
Results printed to console only — paper not updated until review.
"""

import json
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "onchain_direction"))

from src.validation.walk_forward import WalkForwardCV
import yaml

# ── Elsevier figure style ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.framealpha": 0.85,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
COL2 = 7.48   # double column 190 mm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────
with open(ROOT / "onchain_direction" / "config.yaml") as f:
    config = yaml.safe_load(f)

COMMISSION   = 0.005          # 0.5% per trade (one-way)
TAX_RATE     = 0.30           # 30% on each realised profit
INITIAL_CAP  = 1_000.0        # USD — matches Omole
STABLE_FOLDS = 6
THRESHOLD    = 0.50

MODEL_FILES = {
    "XGBoost":  ROOT / "results/optuna/xgboost_g3_optuna_final.json",
    "LightGBM": ROOT / "results/optuna/lightgbm_g3_optuna_final.json",
    "SVM":      ROOT / "results/optuna/svm_g3_optuna_final.json",
    "CNN-LSTM": ROOT / "results/optuna/cnn_lstm_g3_optuna_final.json",
    "DECA":     ROOT / "results/optuna/mlp_dual_v2_optuna.json",
}

# ── load dataset ──────────────────────────────────────────────────────────────
log.info("Loading dataset …")
DATASET_FILE = "data/bitcoin_onchain_2013_2025.csv"
dataset_path = ROOT / DATASET_FILE
df = pd.read_csv(dataset_path, parse_dates=["Date"])
df = df.set_index("Date").ffill().bfill()
log.info(f"  Loaded {len(df)} rows")

cv = WalkForwardCV(config)

fold_close, fold_dates = [], []
for fold in list(cv)[:STABLE_FOLDS]:
    _, df_test = fold.split(df)
    fold_close.append(df_test["Close"])
    fold_dates.append(df_test.index)

close_all = pd.concat(fold_close)
dates_all  = close_all.index
log.info(f"Test period: {dates_all[0].date()} → {dates_all[-1].date()}  "
         f"({len(dates_all)} trading days)")


# ────────────────────────────────────────────────────────────────────────────
def simulate_omole(proba_array: np.ndarray,
                   close_series: pd.Series,
                   threshold: float = THRESHOLD,
                   commission: float = COMMISSION,
                   tax_rate: float = TAX_RATE,
                   initial_cap: float = INITIAL_CAP) -> dict:
    """
    Long/Short simulation matching Omole (2025) methodology:
      - Signal +1 (long) when proba > threshold, -1 (short) otherwise
      - Commission paid on each position change (entry and exit)
      - 30% tax on each realised profit when closing a position
      - Compounding throughout
    """
    n = len(close_series)
    prices = close_series.values
    signal = np.where(proba_array > threshold, 1, -1)   # +1=long, -1=short

    capital      = initial_cap
    equity       = np.zeros(n)
    equity[0]    = capital
    daily_ret    = np.zeros(n)

    position     = 0          # current position: 0=none, +1=long, -1=short
    cap_at_open  = capital    # capital when position was opened (for tax calc)

    for i in range(n - 1):
        p_today    = prices[i]
        p_tomorrow = prices[i + 1]
        raw_ret    = (p_tomorrow - p_today) / p_today   # BTC day return

        sig = signal[i]

        # ── position change ────────────────────────────────────────────────
        if sig != position:
            # 1. Close existing position (if any) and apply tax on profit
            if position != 0:
                capital *= (1 - commission)          # exit commission
                profit = capital - cap_at_open
                if profit > 0:
                    capital -= tax_rate * profit     # tax on realised gain

            # 2. Open new position
            capital *= (1 - commission)              # entry commission
            position    = sig
            cap_at_open = capital

        # ── apply day return for current position ──────────────────────────
        if position == 1:        # long: profit when price rises
            capital *= (1 + raw_ret)
            daily_ret[i + 1] = raw_ret
        elif position == -1:     # short: profit when price falls
            capital *= (1 - raw_ret)
            daily_ret[i + 1] = -raw_ret

        equity[i + 1] = capital

    # Close final position
    if position != 0:
        capital *= (1 - commission)
        profit = capital - cap_at_open
        if profit > 0:
            capital -= tax_rate * profit
        equity[-1] = capital

    # ── metrics ──────────────────────────────────────────────────────────────
    total_roi  = (equity[-1] / initial_cap - 1) * 100
    n_years    = len(close_series) / 365.0
    # Omole annual ROR formula (eq. 8)
    annual_ror = ((equity[-1] / initial_cap) ** (1 / n_years) - 1) * 100

    dr = pd.Series(daily_ret[1:])
    sharpe = (dr.mean() / dr.std() * np.sqrt(365)) if dr.std() > 1e-10 else 0.0

    roll_max = np.maximum.accumulate(equity)
    drawdown = (equity - roll_max) / roll_max * 100
    max_dd   = float(drawdown.min())

    # Win rate: days where position direction matched price move
    wins = 0; total_days = 0
    for i in range(n - 1):
        raw_ret = (prices[i + 1] - prices[i]) / prices[i]
        sig = signal[i]
        if sig == 1 and raw_ret > 0:
            wins += 1
        elif sig == -1 and raw_ret < 0:
            wins += 1
        total_days += 1
    win_rate = wins / total_days * 100 if total_days > 0 else 0.0

    n_changes = int(np.sum(np.abs(np.diff(signal)) > 0))

    return {
        "equity":      equity.tolist(),
        "daily_ret":   daily_ret.tolist(),
        "total_roi":   round(total_roi, 2),
        "annual_ror":  round(annual_ror, 2),     # Omole metric
        "sharpe":      round(float(sharpe), 3),
        "max_dd":      round(max_dd, 2),
        "win_rate":    round(win_rate, 2),
        "n_trades":    n_changes,
        "final_cap":   round(float(equity[-1]), 2),
    }


def buy_and_hold_omole(close_series: pd.Series,
                       initial_cap: float = INITIAL_CAP) -> dict:
    """Buy & Hold — no commission, no tax (reference benchmark)."""
    prices = close_series.values
    equity = initial_cap * prices / prices[0]
    daily_ret = np.diff(prices) / prices[:-1]
    dr = pd.Series(daily_ret)
    total_roi  = (equity[-1] / initial_cap - 1) * 100
    n_years    = len(close_series) / 365.0
    annual_ror = ((equity[-1] / initial_cap) ** (1 / n_years) - 1) * 100
    sharpe     = (dr.mean() / dr.std() * np.sqrt(365)) if dr.std() > 1e-10 else 0.0
    roll_max   = np.maximum.accumulate(equity)
    drawdown   = (equity - roll_max) / roll_max * 100
    return {
        "equity":     equity.tolist(),
        "total_roi":  round(float(total_roi), 2),
        "annual_ror": round(float(annual_ror), 2),
        "sharpe":     round(float(sharpe), 3),
        "max_dd":     round(float(drawdown.min()), 2),
        "win_rate":   round(float((daily_ret > 0).mean() * 100), 2),
        "n_trades":   1,
        "final_cap":  round(float(equity[-1]), 2),
    }


# ── run simulations ───────────────────────────────────────────────────────────
results = {}

log.info("\nSimulating Buy & Hold …")
results["Buy&Hold"] = buy_and_hold_omole(close_all)
r = results["Buy&Hold"]
log.info(f"  Buy&Hold   AnnROR={r['annual_ror']:.0f}%  Sharpe={r['sharpe']:.2f}"
         f"  MaxDD={r['max_dd']:.1f}%  FinalCap=${r['final_cap']:,.0f}")

for model_name, json_path in MODEL_FILES.items():
    log.info(f"Simulating {model_name} …")
    if not json_path.exists():
        log.warning(f"  ⚠  Not found: {json_path}")
        continue
    with open(json_path) as f:
        data = json.load(f)
    proba_all    = np.array(data["y_proba_all"])
    n_stable     = len(close_all)
    if len(proba_all) >= n_stable:
        proba_stable = proba_all[:n_stable]
        close_use    = close_all
    else:
        proba_stable = proba_all[:len(proba_all)]
        close_use    = close_all.iloc[:len(proba_stable)]

    sim = simulate_omole(proba_stable, close_use)
    results[model_name] = sim
    log.info(f"  {model_name:10s}  AnnROR={sim['annual_ror']:7.0f}%  "
             f"Sharpe={sim['sharpe']:5.2f}  MaxDD={sim['max_dd']:6.1f}%  "
             f"WinRate={sim['win_rate']:.1f}%  FinalCap=${sim['final_cap']:,.0f}")

# ── summary table ─────────────────────────────────────────────────────────────
log.info("\n" + "="*75)
log.info(f"{'Model':<12} {'AnnROR%':>9} {'Sharpe':>8} {'MaxDD%':>8} "
         f"{'WinRate%':>10} {'FinalCap$':>12}")
log.info("-"*75)
for name, r in results.items():
    log.info(f"{name:<12} {r['annual_ror']:>9.0f} {r['sharpe']:>8.2f} "
             f"{r['max_dd']:>8.1f} {r['win_rate']:>10.1f} {r['final_cap']:>12,.0f}")
log.info("="*75)
log.info("\nAssumptions (matching Omole 2025):")
log.info("  Strategy  : Long/Short (long when UP predicted, short when DOWN)")
log.info(f"  Commission: {COMMISSION*100:.1f}% per trade (one-way)")
log.info(f"  Tax       : {TAX_RATE*100:.0f}% on each realised profit")
log.info(f"  Capital   : ${INITIAL_CAP:,.0f} starting")
log.info("  Slippage  : zero (same assumption as Omole)")

# ── Figure: cumulative portfolio value (log scale) ────────────────────────
COLOURS = {
    "Buy&Hold": "#7f7f7f",
    "CNN-LSTM": "#d62728",
    "XGBoost":  "#ff7f0e",
    "LightGBM": "#2ca02c",
    "SVM":      "#9467bd",
    "DECA":     "#1f77b4",
}
MARKERS = {
    "Buy&Hold": "D",
    "CNN-LSTM": "x",
    "XGBoost":  "s",
    "LightGBM": "^",
    "SVM":      "v",
    "DECA":     "o",
}
PLOT_ORDER = ["Buy&Hold", "CNN-LSTM", "XGBoost", "LightGBM", "SVM", "DECA"]

fig, ax = plt.subplots(figsize=(COL2, 2.8))
for name in PLOT_ORDER:
    if name not in results:
        continue
    eq  = results[name]["equity"]
    ann = results[name]["annual_ror"]
    wr  = results[name]["win_rate"]
    ls  = "--" if name == "Buy&Hold" else "-"
    lbl = f"{name}  ({ann:,.0f}%/yr, WR={wr:.0f}%)"
    ax.plot(dates_all[:len(eq)], eq,
            color=COLOURS.get(name, "#999"),
            lw=1.0, ls=ls,
            marker=MARKERS.get(name, "o"), markersize=3, markevery=90,
            label=lbl)

for year in range(2019, 2025):
    ax.axvline(pd.Timestamp(f"{year}-01-01"),
               color="lightgrey", lw=0.7, ls=":", zorder=1)

ax.set_yscale("log")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value (USD)")
ax.yaxis.set_major_formatter(mtick.LogFormatterMathtext())
ax.legend(loc="upper left", ncol=2)
ax.grid(axis="y", ls=":", alpha=0.4)
plt.tight_layout()

fig_dir = ROOT / "paper" / "EAAI" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "backtesting_cumulative.pdf", bbox_inches="tight")
plt.close()
log.info(f"Figure saved → {fig_dir / 'backtesting_cumulative.pdf'}")
