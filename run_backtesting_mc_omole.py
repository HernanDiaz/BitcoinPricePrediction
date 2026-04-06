"""
MC Dropout backtesting — Omole (2025) methodology.

Strategy variants (all using Omole parameters):
  - MC-100%  : trade every day (long/short, no uncertainty filter)
  - MC-75%   : trade on low-σ days (long/short), hold cash on high-σ days
  - MC-50%   : trade on low-σ days, hold cash on high-σ days
  - MC-25%   : trade on low-σ days, hold cash on high-σ days
  - Det       : deterministic DECA, trade every day

Omole parameters:
  - Commission: 0.5% per trade (one-way), on every position change
  - Tax: 30% on each realised profit when closing a position
  - Starting capital: $1,000
  - Annual ROR: ((End/Start)^(365/days)) - 1
  - No commission/tax applied on days where we hold cash

On unmasked (allowed) days:
  - proba > 0.5 → LONG
  - proba <= 0.5 → SHORT

On masked (high uncertainty) days:
  - Hold CASH (close any open position, no new position)

Results printed to console only.
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

# ── Elsevier figure style ─────────────────────────────────────────────────
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

from src.validation.walk_forward import WalkForwardCV
import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

COMMISSION   = 0.005      # 0.5% per trade (one-way)
TAX_RATE     = 0.30       # 30% on realised profit
INITIAL_CAP  = 1_000.0   # USD
STABLE_FOLDS = 6
THRESHOLD    = 0.50

# ── load config & dataset ────────────────────────────────────────────────────
with open(ROOT / "onchain_direction" / "config.yaml") as f:
    config = yaml.safe_load(f)

log.info("Loading dataset …")
df = pd.read_csv(ROOT / "data/bitcoin_onchain_2013_2025.csv",
                 parse_dates=["Date"])
df = df.set_index("Date").ffill().bfill()

cv = WalkForwardCV(config)
fold_closes = []
for fold in list(cv)[:STABLE_FOLDS]:
    _, df_test = fold.split(df)
    fold_closes.append(df_test["Close"])
close_all = pd.concat(fold_closes)
dates_all  = close_all.index
log.info(f"  {len(close_all)} trading days, {dates_all[0].date()} → {dates_all[-1].date()}")

# ── load MC Dropout records ──────────────────────────────────────────────────
log.info("Loading MC Dropout records …")
with open(ROOT / "results/mc_dropout/mc_dropout_records.json") as f:
    records = json.load(f)

assert len(records) == len(close_all), \
    f"Mismatch: {len(records)} MC records vs {len(close_all)} price days"

p_up_mc   = np.array([r["p_up_mc"]    for r in records])
p_up_det  = np.array([r["p_up_det"]   for r in records])
sigma     = np.array([r["uncertainty"] for r in records])

log.info(f"  σ range: [{sigma.min():.4f}, {sigma.max():.4f}]  mean={sigma.mean():.4f}")


# ── simulation ───────────────────────────────────────────────────────────────
def simulate_omole_mc(proba: np.ndarray,
                      mask: np.ndarray,
                      close_series: pd.Series,
                      threshold: float = THRESHOLD,
                      commission: float = COMMISSION,
                      tax_rate: float = TAX_RATE,
                      initial_cap: float = INITIAL_CAP) -> dict:
    """
    Long/Short simulation with Omole (2025) parameters + MC uncertainty mask.

    mask[i] = True  → low uncertainty: trade (long or short based on proba)
    mask[i] = False → high uncertainty: close any open position, hold cash

    Position transitions:
      - Cash → Long/Short: pay entry commission
      - Long/Short → opposite: pay exit commission + 30% tax on profit, then entry commission
      - Long/Short → Cash: pay exit commission + 30% tax on profit
      - Cash → Cash: no cost
    """
    n       = len(close_series)
    prices  = close_series.values

    # Desired signal: +1=long, -1=short, 0=cash (masked days)
    desired = np.where(mask, np.where(proba > threshold, 1, -1), 0)

    capital      = initial_cap
    equity       = np.zeros(n)
    equity[0]    = capital
    daily_ret    = np.zeros(n)

    position     = 0       # current: 0=cash, +1=long, -1=short
    cap_at_open  = capital

    for i in range(n - 1):
        p_today    = prices[i]
        p_tomorrow = prices[i + 1]
        raw_ret    = (p_tomorrow - p_today) / p_today

        sig = desired[i]

        # ── position change ──────────────────────────────────────────────────
        if sig != position:
            # Close existing position (if any)
            if position != 0:
                capital *= (1 - commission)          # exit commission
                profit = capital - cap_at_open
                if profit > 0:
                    capital -= tax_rate * profit     # tax on realised gain

            # Open new position (if signal is not cash)
            if sig != 0:
                capital *= (1 - commission)          # entry commission
                cap_at_open = capital

            position = sig

        # ── apply day return ────────────────────────────────────────────────
        if position == 1:        # long
            capital *= (1 + raw_ret)
            daily_ret[i + 1] = raw_ret
        elif position == -1:     # short
            capital *= (1 - raw_ret)
            daily_ret[i + 1] = -raw_ret
        # position == 0: cash, no return

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
    annual_ror = ((equity[-1] / initial_cap) ** (1 / n_years) - 1) * 100

    dr = pd.Series(daily_ret[1:])
    sharpe = (dr.mean() / dr.std() * np.sqrt(365)) if dr.std() > 1e-10 else 0.0

    roll_max = np.maximum.accumulate(equity)
    drawdown = (equity - roll_max) / roll_max * 100
    max_dd   = float(drawdown.min())

    # Win rate on active (non-cash) days
    wins = 0; total_active = 0
    for i in range(n - 1):
        sig = desired[i]
        if sig == 0:
            continue
        raw_ret = (prices[i + 1] - prices[i]) / prices[i]
        if (sig == 1 and raw_ret > 0) or (sig == -1 and raw_ret < 0):
            wins += 1
        total_active += 1
    win_rate = wins / total_active * 100 if total_active > 0 else 0.0

    coverage  = float(mask.mean() * 100)
    active_pct = float((desired != 0).mean() * 100)

    n_changes = int(np.sum(np.abs(np.diff(np.where(desired == 0, position, desired))) > 0
                          if False else 0))
    # Count position changes properly
    pos_seq = []
    pos = 0
    for i in range(n):
        s = desired[i]
        if s != pos:
            pos_seq.append(s)
            pos = s
    n_trades = max(0, len(pos_seq) - 1)

    return {
        "equity":      equity.tolist(),
        "daily_ret":   daily_ret.tolist(),
        "total_roi":   round(total_roi, 2),
        "annual_ror":  round(annual_ror, 2),
        "sharpe":      round(float(sharpe), 3),
        "max_dd":      round(max_dd, 2),
        "win_rate":    round(win_rate, 2),
        "coverage":    round(coverage, 1),
        "active_pct":  round(active_pct, 1),
        "n_long_days": int((desired == 1).sum()),
        "n_short_days": int((desired == -1).sum()),
        "n_cash_days": int((desired == 0).sum()),
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
        "equity":      equity.tolist(),
        "total_roi":   round(float(total_roi), 2),
        "annual_ror":  round(float(annual_ror), 2),
        "sharpe":      round(float(sharpe), 3),
        "max_dd":      round(float(drawdown.min()), 2),
        "win_rate":    round(float((daily_ret > 0).mean() * 100), 2),
        "coverage":    100.0,
        "active_pct":  100.0,
        "n_long_days": len(close_series),
        "n_short_days": 0,
        "n_cash_days": 0,
        "final_cap":   round(float(equity[-1]), 2),
    }


# ── load SVM proba for reference comparison ───────────────────────────────
SVM_JSON = ROOT / "results/optuna/svm_g3_optuna_final.json"

# ── run simulations ───────────────────────────────────────────────────────────
results = {}

log.info("\nSimulating Buy & Hold …")
results["Buy&Hold"] = buy_and_hold_omole(close_all)

if SVM_JSON.exists():
    log.info("Simulating SVM (reference) …")
    with open(SVM_JSON) as f:
        svm_data = json.load(f)
    svm_proba = np.array(svm_data["y_proba_all"])
    n_stable  = len(close_all)
    if len(svm_proba) >= n_stable:
        svm_proba = svm_proba[:n_stable]
        close_svm = close_all
    else:
        close_svm = close_all.iloc[:len(svm_proba)]
    mask_svm = np.ones(len(svm_proba), dtype=bool)
    results["SVM"] = simulate_omole_mc(svm_proba, mask_svm, close_svm)
    r = results["SVM"]
    log.info(f"  SVM        AnnROR={r['annual_ror']:.0f}%  Sharpe={r['sharpe']:.2f}"
             f"  MaxDD={r['max_dd']:.1f}%  WinRate={r['win_rate']:.1f}%  FinalCap=${r['final_cap']:,.0f}")
r = results["Buy&Hold"]
log.info(f"  Buy&Hold   AnnROR={r['annual_ror']:.0f}%  Sharpe={r['sharpe']:.2f}"
         f"  MaxDD={r['max_dd']:.1f}%  FinalCap=${r['final_cap']:,.0f}")

# Deterministic DECA — trade every day
log.info("Simulating Det-DECA (all days, long/short) …")
mask_all = np.ones(len(p_up_det), dtype=bool)
results["Det-100%"] = simulate_omole_mc(p_up_det, mask_all, close_all)
r = results["Det-100%"]
log.info(f"  Det-100%   AnnROR={r['annual_ror']:.0f}%  Sharpe={r['sharpe']:.2f}"
         f"  MaxDD={r['max_dd']:.1f}%  WinRate={r['win_rate']:.1f}%  FinalCap=${r['final_cap']:,.0f}")

# MC Dropout coverage levels
COVERAGES = [100, 75, 50, 25]
for cov in COVERAGES:
    label = f"MC-{cov}%"
    if cov == 100:
        mask = np.ones(len(sigma), dtype=bool)
    else:
        thr = np.percentile(sigma, cov)
        mask = sigma <= thr
    n_active = mask.sum()
    log.info(f"Simulating {label}  ({n_active} trading days, {n_active/len(sigma)*100:.0f}% coverage) …")
    results[label] = simulate_omole_mc(p_up_mc, mask, close_all)
    r = results[label]
    log.info(f"  {label:<10}  AnnROR={r['annual_ror']:7.0f}%  "
             f"Sharpe={r['sharpe']:5.2f}  MaxDD={r['max_dd']:6.1f}%  "
             f"WinRate={r['win_rate']:.1f}%  Long={r['n_long_days']}d  "
             f"Short={r['n_short_days']}d  Cash={r['n_cash_days']}d  "
             f"FinalCap=${r['final_cap']:,.0f}")

# ── summary table ─────────────────────────────────────────────────────────────
log.info("\n" + "="*90)
log.info(f"{'Strategy':<12} {'AnnROR%':>9} {'Sharpe':>8} {'MaxDD%':>8} "
         f"{'WinRate%':>10} {'Coverage%':>11} {'FinalCap$':>12}")
log.info("-"*90)
order = ["Buy&Hold", "SVM", "Det-100%"] + [f"MC-{c}%" for c in COVERAGES]
for name in order:
    if name not in results:
        continue
    r = results[name]
    log.info(f"{name:<12} {r['annual_ror']:>9.0f} {r['sharpe']:>8.2f} "
             f"{r['max_dd']:>8.1f} {r['win_rate']:>10.1f} {r['coverage']:>11.1f} "
             f"{r['final_cap']:>12,.0f}")
log.info("="*90)
log.info("\nAssumptions (Omole 2025 methodology):")
log.info("  Strategy  : Long/Short when trading, Cash when σ exceeds threshold")
log.info(f"  Commission: {COMMISSION*100:.1f}% per trade (one-way)")
log.info(f"  Tax       : {TAX_RATE*100:.0f}% on each realised profit")
log.info(f"  Capital   : ${INITIAL_CAP:,.0f} starting")
log.info("  Slippage  : zero")

# ── Figure 1: Cumulative portfolio (log scale) ────────────────────────────
fig_dir = ROOT / "paper" / "EAAI" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

COLOURS = {
    "Buy&Hold":  "#7f7f7f",
    "SVM":       "#9467bd",
    "Det-100%":  "#aec7e8",
    "MC-100%":   "#1f77b4",
    "MC-75%":    "#2ca02c",
    "MC-50%":    "#ff7f0e",
    "MC-25%":    "#d62728",
}
MARKERS = {
    "Buy&Hold":  "D",
    "SVM":       "v",
    "Det-100%":  "^",
    "MC-100%":   "o",
    "MC-75%":    "s",
    "MC-50%":    "P",
    "MC-25%":    "*",
}

fig, ax = plt.subplots(figsize=(COL2, 2.8))
plot_order = ["Buy&Hold", "SVM", "Det-100%", "MC-100%", "MC-75%", "MC-50%", "MC-25%"]
for name in plot_order:
    if name not in results:
        continue
    eq  = results[name]["equity"]
    ann = results[name]["annual_ror"]
    wr  = results[name]["win_rate"]
    ls  = "--" if name == "Buy&Hold" else "-"
    ax.plot(dates_all[:len(eq)], eq,
            color=COLOURS.get(name, "#999"),
            lw=1.0, ls=ls,
            marker=MARKERS.get(name, "o"), markersize=3, markevery=90,
            label=f"{name}  ({ann:,.0f}%/yr, WR={wr:.0f}%)")

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
plt.savefig(fig_dir / "mc_backtesting_cumulative.pdf", bbox_inches="tight")
plt.close()
log.info(f"Figure saved → {fig_dir / 'mc_backtesting_cumulative.pdf'}")

# ── Figure 2: Coverage vs Sharpe / WinRate / AnnROR ─────────────────────
COVERAGES_LIST = [100, 75, 50, 25]
mc_coverages = [results[f"MC-{c}%"]["coverage"]   for c in COVERAGES_LIST]
mc_sharpes   = [results[f"MC-{c}%"]["sharpe"]     for c in COVERAGES_LIST]
mc_winrates  = [results[f"MC-{c}%"]["win_rate"]   for c in COVERAGES_LIST]
mc_annroi    = [results[f"MC-{c}%"]["annual_ror"] for c in COVERAGES_LIST]

det_sharpe = results["Det-100%"]["sharpe"]
det_wr     = results["Det-100%"]["win_rate"]
svm_sharpe = results["SVM"]["sharpe"]
svm_wr     = results["SVM"]["win_rate"]

fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.6))

# Sharpe
axes[0].plot(mc_coverages, mc_sharpes, "o-", color="#1f77b4", lw=1.0, ms=4, label="MC Dropout")
axes[0].axhline(det_sharpe, color="#aec7e8", ls="--", lw=0.8, label=f"Det. DECA ({det_sharpe:.2f})")
axes[0].axhline(svm_sharpe, color="#9467bd", ls=":",  lw=0.8, label=f"SVM ({svm_sharpe:.2f})")
axes[0].set_xlabel("Coverage (%)")
axes[0].set_ylabel("Sharpe Ratio")
axes[0].legend()
axes[0].grid(ls=":", alpha=0.4)
axes[0].invert_xaxis()

# Win Rate
axes[1].plot(mc_coverages, mc_winrates, "s-", color="#2ca02c", lw=1.0, ms=4, label="MC Dropout")
axes[1].axhline(det_wr, color="#aec7e8", ls="--", lw=0.8, label=f"Det. DECA ({det_wr:.1f}%)")
axes[1].axhline(svm_wr, color="#9467bd", ls=":",  lw=0.8, label=f"SVM ({svm_wr:.1f}%)")
axes[1].set_xlabel("Coverage (%)")
axes[1].set_ylabel("Win Rate (%)")
axes[1].legend()
axes[1].grid(ls=":", alpha=0.4)
axes[1].invert_xaxis()

# Annualised ROR
det_roi = results["Det-100%"]["annual_ror"]
svm_roi = results["SVM"]["annual_ror"]
axes[2].plot(mc_coverages, mc_annroi, "^-", color="#ff7f0e", lw=1.0, ms=4, label="MC Dropout")
axes[2].axhline(det_roi, color="#aec7e8", ls="--", lw=0.8, label=f"Det. DECA ({det_roi:,.0f}%/yr)")
axes[2].axhline(svm_roi, color="#9467bd", ls=":",  lw=0.8, label=f"SVM ({svm_roi:,.0f}%/yr)")
axes[2].set_xlabel("Coverage (%)")
axes[2].set_ylabel("Annualised ROR (%)")
axes[2].legend()
axes[2].grid(ls=":", alpha=0.4)
axes[2].invert_xaxis()

plt.tight_layout()
plt.savefig(fig_dir / "mc_backtesting_coverage.pdf", bbox_inches="tight")
plt.close()
log.info(f"Figure saved → {fig_dir / 'mc_backtesting_coverage.pdf'}")
log.info("\nDone.")
