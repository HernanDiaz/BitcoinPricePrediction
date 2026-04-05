"""
MC Dropout backtesting — Dynamic Threshold Strategy.

Instead of a fixed threshold of 0.5, use:
    threshold_up   = 0.5 + k * σ_t
    threshold_down = 0.5 - k * σ_t

Rules:
    p_t > 0.5 + k*σ_t  → LONG   (strong UP signal relative to uncertainty)
    p_t < 0.5 - k*σ_t  → SHORT  (strong DOWN signal relative to uncertainty)
    |p_t - 0.5| ≤ k*σ_t → CASH  (signal too weak relative to uncertainty)

Rationale: σ quantifies the model's epistemic uncertainty on day t.
Requiring the signal to exceed its own uncertainty margin filters out
low-confidence borderline days without discarding ALL high-σ days
indiscriminately.  A strong signal (p=0.80) survives even with σ=0.20
(k=1: threshold=0.70 < 0.80 → LONG). A weak signal (p=0.55) is
filtered out when σ=0.10 (k=1: threshold=0.60 > 0.55 → CASH).

We sweep k over a grid to find the optimal value and report full metrics.
Omole (2025) conditions throughout:
  Commission : 0.5% per trade (one-way)
  Tax        : 30% on each realised profit
  Capital    : $1,000 starting, full compounding
  Annual ROR : ((C_T/C_0)^(365/N)) - 1
"""

import json
import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "onchain_direction"))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.framealpha": 0.85,
    "axes.spines.top": False, "axes.spines.right": False,
})
COL2 = 7.48

from src.validation.walk_forward import WalkForwardCV
import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

COMMISSION  = 0.005
TAX_RATE    = 0.30
INITIAL_CAP = 1_000.0
STABLE_FOLDS = 6

# ── load data ──────────────────────────────────────────────────────────────
with open(ROOT / "onchain_direction" / "config.yaml") as f:
    config = yaml.safe_load(f)

log.info("Loading dataset …")
df = pd.read_csv(ROOT / "dataset_COMPLETO_con_OHLCV_20251221_014211.csv",
                 parse_dates=["Date"])
df = df.set_index("Date").ffill().bfill()

cv = WalkForwardCV(config)
fold_closes = []
for fold in list(cv)[:STABLE_FOLDS]:
    _, df_test = fold.split(df)
    fold_closes.append(df_test["Close"])
close_all = pd.concat(fold_closes)
dates_all = close_all.index
N = len(close_all)
log.info(f"  {N} trading days, {dates_all[0].date()} → {dates_all[-1].date()}")

log.info("Loading MC Dropout records …")
with open(ROOT / "results/mc_dropout/mc_dropout_records.json") as f:
    records = json.load(f)
assert len(records) == N
p_up_mc  = np.array([r["p_up_mc"]    for r in records])
p_up_det = np.array([r["p_up_det"]   for r in records])
sigma    = np.array([r["uncertainty"] for r in records])
log.info(f"  σ  mean={sigma.mean():.4f}  median={np.median(sigma):.4f}"
         f"  min={sigma.min():.4f}  max={sigma.max():.4f}")
log.info(f"  |p_mc - 0.5| mean={np.abs(p_up_mc - 0.5).mean():.4f}")


# ── simulation engine ──────────────────────────────────────────────────────
def simulate(desired: np.ndarray, close_series: pd.Series) -> dict:
    n      = len(close_series)
    prices = close_series.values
    capital     = INITIAL_CAP
    equity      = np.zeros(n);  equity[0] = capital
    daily_ret   = np.zeros(n)
    position    = 0
    cap_at_open = capital

    for i in range(n - 1):
        raw_ret = (prices[i + 1] - prices[i]) / prices[i]
        sig     = int(desired[i])
        if sig != position:
            if position != 0:
                capital *= (1 - COMMISSION)
                profit = capital - cap_at_open
                if profit > 0:
                    capital -= TAX_RATE * profit
            if sig != 0:
                capital *= (1 - COMMISSION)
                cap_at_open = capital
            position = sig
        if   position ==  1: capital *= (1 + raw_ret); daily_ret[i+1] = raw_ret
        elif position == -1: capital *= (1 - raw_ret); daily_ret[i+1] = -raw_ret
        equity[i+1] = capital

    if position != 0:
        capital *= (1 - COMMISSION)
        profit = capital - cap_at_open
        if profit > 0:
            capital -= TAX_RATE * profit
        equity[-1] = capital

    n_years    = n / 365.0
    annual_ror = ((equity[-1] / INITIAL_CAP) ** (1 / n_years) - 1) * 100
    dr = pd.Series(daily_ret[1:])
    sharpe = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 1e-10 else 0.0
    roll_max = np.maximum.accumulate(equity)
    max_dd   = float(((equity - roll_max) / roll_max * 100).min())

    wins = tot = 0
    for i in range(n - 1):
        s = int(desired[i])
        if s == 0: continue
        r = (prices[i+1] - prices[i]) / prices[i]
        if (s == 1 and r > 0) or (s == -1 and r < 0): wins += 1
        tot += 1
    win_rate = wins / tot * 100 if tot > 0 else 0.0

    return dict(
        equity    = equity.tolist(),
        annual_ror= round(annual_ror, 2),
        sharpe    = round(float(sharpe), 3),
        max_dd    = round(max_dd, 2),
        win_rate  = round(win_rate, 2),
        final_cap = round(float(equity[-1]), 2),
        n_long    = int((desired == 1).sum()),
        n_short   = int((desired == -1).sum()),
        n_cash    = int((desired == 0).sum()),
        coverage  = round(float((desired != 0).mean() * 100), 1),
    )


# ── baselines ──────────────────────────────────────────────────────────────
log.info("\nBaselines …")

# Det-100% (from mc records, p_up_det)
des_det = np.where(p_up_det > 0.5, 1, -1).astype(int)
base_det = simulate(des_det, close_all)
log.info(f"  Det-100%   AnnROR={base_det['annual_ror']:6.0f}%  "
         f"Sharpe={base_det['sharpe']:.2f}  MaxDD={base_det['max_dd']:.1f}%  "
         f"WinRate={base_det['win_rate']:.1f}%  "
         f"Long={base_det['n_long']}d  Short={base_det['n_short']}d")

# MC-100% (p_up_mc, no filter)
des_mc = np.where(p_up_mc > 0.5, 1, -1).astype(int)
base_mc = simulate(des_mc, close_all)
log.info(f"  MC-100%    AnnROR={base_mc['annual_ror']:6.0f}%  "
         f"Sharpe={base_mc['sharpe']:.2f}  MaxDD={base_mc['max_dd']:.1f}%  "
         f"WinRate={base_mc['win_rate']:.1f}%  "
         f"Long={base_mc['n_long']}d  Short={base_mc['n_short']}d")

# ── sweep k ────────────────────────────────────────────────────────────────
# Fine grid: 0 → 10 in small steps + extra fine near promising region
k_values = sorted(set(
    list(np.arange(0.0, 1.0,  0.1)) +
    list(np.arange(1.0, 3.0,  0.25)) +
    list(np.arange(3.0, 6.0,  0.5)) +
    list(np.arange(6.0, 12.0, 1.0))
))

log.info(f"\nSweeping k over {len(k_values)} values …")
log.info(f"  {'k':>5}  {'AnnROR%':>8}  {'Sharpe':>6}  {'MaxDD%':>7}  "
         f"{'WinRate%':>9}  {'Coverage%':>10}  {'Long':>5}  {'Short':>5}  {'Cash':>5}")
log.info("  " + "-"*80)

sweep_results = {}
for k in k_values:
    margin   = k * sigma
    desired  = np.where(p_up_mc > 0.5 + margin,  1,
               np.where(p_up_mc < 0.5 - margin, -1, 0)).astype(int)
    r = simulate(desired, close_all)
    sweep_results[k] = r
    log.info(f"  {k:>5.2f}  {r['annual_ror']:>8.0f}  {r['sharpe']:>6.2f}  "
             f"{r['max_dd']:>7.1f}  {r['win_rate']:>9.1f}  {r['coverage']:>10.1f}  "
             f"{r['n_long']:>5}  {r['n_short']:>5}  {r['n_cash']:>5}")

# ── find best k ────────────────────────────────────────────────────────────
best_k_ror    = max(sweep_results, key=lambda k: sweep_results[k]["annual_ror"])
best_k_sharpe = max(sweep_results, key=lambda k: sweep_results[k]["sharpe"])

log.info(f"\nBest by AnnROR  : k={best_k_ror:.2f}  "
         f"→ {sweep_results[best_k_ror]['annual_ror']:.0f}%/yr  "
         f"Sharpe={sweep_results[best_k_ror]['sharpe']:.2f}  "
         f"MaxDD={sweep_results[best_k_ror]['max_dd']:.1f}%  "
         f"Coverage={sweep_results[best_k_ror]['coverage']:.0f}%")
log.info(f"Best by Sharpe  : k={best_k_sharpe:.2f}  "
         f"→ {sweep_results[best_k_sharpe]['annual_ror']:.0f}%/yr  "
         f"Sharpe={sweep_results[best_k_sharpe]['sharpe']:.2f}  "
         f"MaxDD={sweep_results[best_k_sharpe]['max_dd']:.1f}%  "
         f"Coverage={sweep_results[best_k_sharpe]['coverage']:.0f}%")
log.info(f"\nReference DECA (omole script) : 3922%/yr")
log.info(f"Reference Det-100% (mc script): {base_det['annual_ror']:.0f}%/yr")
log.info(f"Reference MC-100%             : {base_mc['annual_ror']:.0f}%/yr")

# ── save results ────────────────────────────────────────────────────────────
out_dir = ROOT / "results" / "backtesting"
out_dir.mkdir(parents=True, exist_ok=True)
ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = out_dir / f"dynamic_threshold_results_{ts}.json"

save = {
    "description": "Dynamic threshold: LONG if p > 0.5+k*σ, SHORT if p < 0.5-k*σ, else CASH",
    "parameters": {
        "commission": COMMISSION, "tax_rate": TAX_RATE,
        "initial_cap": INITIAL_CAP,
        "test_period": f"{dates_all[0].date()} → {dates_all[-1].date()}",
        "n_days": N,
    },
    "baselines": {
        "Det-100%": {kk: vv for kk, vv in base_det.items() if kk != "equity"},
        "MC-100%":  {kk: vv for kk, vv in base_mc.items()  if kk != "equity"},
    },
    "sweep": {
        str(k): {kk: vv for kk, vv in r.items() if kk != "equity"}
        for k, r in sweep_results.items()
    },
    "best_k_ror":    best_k_ror,
    "best_k_sharpe": best_k_sharpe,
}
with open(out_path, "w") as f:
    json.dump(save, f, indent=2)
log.info(f"\nResults saved → {out_path}")

# ── figures ────────────────────────────────────────────────────────────────
fig_dir = ROOT / "paper" / "EAAI" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

ks      = sorted(sweep_results.keys())
rors    = [sweep_results[k]["annual_ror"] for k in ks]
sharpes = [sweep_results[k]["sharpe"]     for k in ks]
mdd     = [sweep_results[k]["max_dd"]     for k in ks]
wrs     = [sweep_results[k]["win_rate"]   for k in ks]
covs    = [sweep_results[k]["coverage"]   for k in ks]

fig, axes = plt.subplots(2, 2, figsize=(COL2, 4.5))

# Ann ROR vs k
axes[0,0].plot(ks, rors, "-o", color="#1f77b4", lw=1.0, ms=3)
axes[0,0].axhline(base_det["annual_ror"], color="#aec7e8", ls="--", lw=0.9,
                  label=f"Det-100% ({base_det['annual_ror']:.0f}%)")
axes[0,0].axhline(base_mc["annual_ror"],  color="#9467bd",  ls=":",  lw=0.9,
                  label=f"MC-100% ({base_mc['annual_ror']:.0f}%)")
axes[0,0].axhline(3922, color="red", ls="-.", lw=0.9, label="DECA (3922%)")
axes[0,0].axvline(best_k_ror, color="orange", ls="--", lw=0.8,
                  label=f"best k={best_k_ror:.2f}")
axes[0,0].set_xlabel("k")
axes[0,0].set_ylabel("Ann. ROR (%)")
axes[0,0].legend(fontsize=7)
axes[0,0].grid(ls=":", alpha=0.4)

# Sharpe vs k
axes[0,1].plot(ks, sharpes, "-s", color="#2ca02c", lw=1.0, ms=3)
axes[0,1].axhline(base_det["sharpe"], color="#aec7e8", ls="--", lw=0.9,
                  label=f"Det ({base_det['sharpe']:.2f})")
axes[0,1].axhline(base_mc["sharpe"],  color="#9467bd",  ls=":",  lw=0.9,
                  label=f"MC ({base_mc['sharpe']:.2f})")
axes[0,1].axvline(best_k_sharpe, color="orange", ls="--", lw=0.8,
                  label=f"best k={best_k_sharpe:.2f}")
axes[0,1].set_xlabel("k")
axes[0,1].set_ylabel("Sharpe Ratio")
axes[0,1].legend(fontsize=7)
axes[0,1].grid(ls=":", alpha=0.4)

# WinRate and Coverage vs k
ax2 = axes[1,0]
ax2.plot(ks, wrs,  "-^", color="#d62728", lw=1.0, ms=3, label="Win Rate (%)")
ax2.plot(ks, covs, "-D", color="#8c564b", lw=1.0, ms=3, label="Coverage (%)")
ax2.set_xlabel("k")
ax2.set_ylabel("(%)")
ax2.legend(fontsize=7)
ax2.grid(ls=":", alpha=0.4)

# Max Drawdown vs k
axes[1,1].plot(ks, mdd, "-P", color="#ff7f0e", lw=1.0, ms=3)
axes[1,1].axhline(base_det["max_dd"], color="#aec7e8", ls="--", lw=0.9,
                  label=f"Det ({base_det['max_dd']:.1f}%)")
axes[1,1].axhline(base_mc["max_dd"],  color="#9467bd",  ls=":",  lw=0.9,
                  label=f"MC ({base_mc['max_dd']:.1f}%)")
axes[1,1].set_xlabel("k")
axes[1,1].set_ylabel("Max Drawdown (%)")
axes[1,1].legend(fontsize=7)
axes[1,1].grid(ls=":", alpha=0.4)

plt.suptitle("Dynamic Threshold k-sweep  (Long/Short + 0.5% comm + 30% tax)",
             fontsize=9)
plt.tight_layout()
fp = fig_dir / "dynamic_threshold_sweep.pdf"
plt.savefig(fp, bbox_inches="tight")
plt.close()
log.info(f"Figure saved → {fp}")

log.info("\nDone.")
