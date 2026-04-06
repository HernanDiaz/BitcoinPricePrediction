#!/usr/bin/env python3
"""
enrich_dataset.py
=================
Downloads additional on-chain and sentiment features from free public APIs:

  1. alternative.me  -> Fear & Greed Index (2018-present)
  2. blockchain.info -> HashRate, Difficulty, TxCount, MinerRevenue (2009-present)

Derived features computed internally:
  FearGreed_7d_MA, FearGreed_30d_MA, FearGreed_zone
  HashRate_30d_MA, HashRate_60d_MA, HashRibbon
  NVT_approx (Close * circulating_supply / tx_volume_usd approx)
  PuellMultiple (daily miner rev / 365d MA miner rev)
  TxCount_30d_MA

Usage:
  python onchain_direction/scripts/enrich_dataset.py
"""

import time
import logging
from pathlib import Path

import requests
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT         = Path(__file__).parents[2]
ORIGINAL_CSV = ROOT / "data/bitcoin_onchain_2013_2025.csv"
OUTPUT_CSV   = ROOT / "dataset_COMPLETO_enriched.csv"

# ---------------------------------------------------------------------------
# 1. Fear & Greed Index  (alternative.me) — FREE, no auth
# ---------------------------------------------------------------------------

def fetch_fear_greed() -> pd.DataFrame:
    log.info("Fetching Fear & Greed Index from alternative.me ...")
    url = "https://api.alternative.me/fng/?limit=0&format=json&date_format=us"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()["data"]
    df = pd.DataFrame(data)[["timestamp", "value"]]
    df.columns = ["Date", "FearGreed"]
    df["Date"]      = pd.to_datetime(df["Date"]).dt.normalize()
    df["FearGreed"] = df["FearGreed"].astype(float)
    df = df.sort_values("Date").reset_index(drop=True)
    log.info(f"  {len(df)} rows  ({df.Date.min().date()} -> {df.Date.max().date()})")
    return df


# ---------------------------------------------------------------------------
# 2. Blockchain.info Charts API — FREE, no auth
# ---------------------------------------------------------------------------

BLOCKCHAIN_BASE = "https://api.blockchain.info/charts"

BLOCKCHAIN_CHARTS = {
    "hash-rate"       : "HashRate",       # TH/s
    "difficulty"      : "Difficulty",
    "n-transactions"  : "TxCount",        # daily confirmed txs
    "miners-revenue"  : "MinerRevUSD",    # USD per day
    "estimated-transaction-volume-usd": "TxVolumeUSD",  # USD on-chain volume
}

def fetch_blockchain_chart(chart_name: str, col_name: str) -> pd.DataFrame:
    """Fetch a single chart from blockchain.info."""
    url = f"{BLOCKCHAIN_BASE}/{chart_name}"
    params = {"timespan": "all", "format": "json", "sampled": "false"}
    log.info(f"  Fetching {chart_name} ...")
    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        values = resp.json().get("values", [])
        df = pd.DataFrame(values)          # columns: x (unix ts), y (value)
        df["Date"]   = pd.to_datetime(df["x"], unit="s").dt.normalize()
        df[col_name] = pd.to_numeric(df["y"], errors="coerce")
        df = df[["Date", col_name]].sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
        log.info(f"    -> {len(df)} rows  ({df.Date.min().date()} -> {df.Date.max().date()})")
        return df
    except Exception as e:
        log.warning(f"    Failed to fetch {chart_name}: {e}")
        return pd.DataFrame(columns=["Date", col_name])


def fetch_blockchain() -> pd.DataFrame:
    frames = []
    for chart, col in BLOCKCHAIN_CHARTS.items():
        df = fetch_blockchain_chart(chart, col)
        if not df.empty:
            frames.append(df.set_index("Date"))
        time.sleep(0.5)   # polite rate limit

    if not frames:
        log.error("All blockchain.info requests failed!")
        return pd.DataFrame(columns=["Date"])

    merged = pd.concat(frames, axis=1)
    merged.index.name = "Date"
    merged = merged.reset_index()
    log.info(f"  Blockchain.info merged: {merged.shape}")
    return merged


# ---------------------------------------------------------------------------
# 3. Derived / engineered features
# ---------------------------------------------------------------------------

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Computing derived features ...")

    # --- Fear & Greed ---
    if "FearGreed" in df.columns:
        df["FearGreed_7d_MA"]  = df["FearGreed"].rolling(7,  min_periods=1).mean()
        df["FearGreed_30d_MA"] = df["FearGreed"].rolling(30, min_periods=1).mean()
        # Sentiment regime: -2 extreme fear / -1 fear / 0 neutral / 1 greed / 2 extreme greed
        df["FearGreed_zone"] = pd.cut(
            df["FearGreed"],
            bins=[-0.001, 25, 45, 55, 75, 100],
            labels=[-2, -1, 0, 1, 2]
        ).astype(float)

    # --- Hash Rate ---
    if "HashRate" in df.columns:
        df["HashRate_30d_MA"] = df["HashRate"].rolling(30, min_periods=1).mean()
        df["HashRate_60d_MA"] = df["HashRate"].rolling(60, min_periods=1).mean()
        # Hash Ribbon = 30d - 60d MA  (negative -> miner capitulation)
        df["HashRibbon"] = df["HashRate_30d_MA"] - df["HashRate_60d_MA"]

    # --- Miner Revenue / Puell Multiple ---
    if "MinerRevUSD" in df.columns:
        ma365 = df["MinerRevUSD"].rolling(365, min_periods=90).mean()
        df["PuellMultiple"] = df["MinerRevUSD"] / (ma365 + 1e-10)

    # --- Transaction volume NVT approximation ---
    # NVT = MarketCap / OnChainTxVolumeUSD  (lower = undervalued)
    if "TxVolumeUSD" in df.columns and "Close" in df.columns:
        # Use Close as proxy for market cap relative value
        # NVT_approx = Close / 30d-MA of TxVolumeUSD  (same signal, interpretable)
        df["TxVolume_30d_MA"] = df["TxVolumeUSD"].rolling(30, min_periods=1).mean()
        df["NVT_approx"] = df["Close"] / (df["TxVolume_30d_MA"] + 1e-10)
        # Normalize by its own 365d MA to make it stationary
        df["NVT_ratio"] = df["NVT_approx"] / (
            df["NVT_approx"].rolling(365, min_periods=90).mean() + 1e-10
        )

    # --- Transaction count momentum ---
    if "TxCount" in df.columns:
        df["TxCount_30d_MA"]  = df["TxCount"].rolling(30, min_periods=1).mean()
        df["TxCount_momentum"] = df["TxCount"] / (df["TxCount_30d_MA"] + 1e-10)

    log.info(f"  Done. Total columns: {len(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    # Load original
    log.info(f"Loading: {ORIGINAL_CSV.name}")
    df = pd.read_csv(ORIGINAL_CSV, parse_dates=["Date"])
    df["Date"] = df["Date"].dt.normalize()
    original_cols = set(df.columns)
    log.info(f"  Shape: {df.shape}  |  {df.Date.min().date()} -> {df.Date.max().date()}")

    # --- Fear & Greed ---
    try:
        df_fg = fetch_fear_greed()
        df = df.merge(df_fg, on="Date", how="left")
    except Exception as e:
        log.error(f"Fear & Greed failed: {e}")

    # --- Blockchain.info ---
    df_bc = fetch_blockchain()
    if "Date" in df_bc.columns and len(df_bc) > 1:
        df_bc["Date"] = pd.to_datetime(df_bc["Date"]).dt.normalize()
        df = df.merge(df_bc, on="Date", how="left")

    # --- Derived ---
    df = add_derived_features(df)

    # --- Report ---
    new_cols = [c for c in df.columns if c not in original_cols]
    log.info(f"\n{'='*60}")
    log.info(f"New features added: {len(new_cols)}")

    df_2019 = df[df["Date"] >= "2019-01-01"]
    log.info(f"\nCoverage from 2019+ ({len(df_2019)} rows):")
    for c in new_cols:
        pct = df_2019[c].notna().mean() * 100
        log.info(f"  {c:30s}  {pct:5.1f}%")

    # --- Save ---
    df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"\nSaved -> {OUTPUT_CSV.name}")
    log.info(f"Final shape: {df.shape[0]} rows x {df.shape[1]} cols")
    log.info(f"Total features (excl Date, y): {df.shape[1] - 2}")


if __name__ == "__main__":
    main()
