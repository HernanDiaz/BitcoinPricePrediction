#!/usr/bin/env python3
"""
run_feature_selection.py — Feature selection pipeline reproducible
=======================================================================
Replica el pipeline ultra-robusto de 8 técnicas combinadas aplicado
durante la fase exploratoria (Copia_de_resultadosPaperBitcoinJesus.ipynb,
celda 5) sobre el dataset completo de 579 métricas.

Objetivo: verificar que las 33 features del grupo G3 emergen
naturalmente como las más predictivas, proporcionando justificación
metodológica para el paper.

Pipeline:
  1. Eliminación de correlación alta (>0.95)
  2. SelectKBest (F-statistic)
  3. Mutual Information
  4. Boruta (features relevantes)
  5. RFE (Recursive Feature Elimination)
  6. Random Forest Importance
  7. XGBoost Importance
  8. SHAP Values

Salidas:
  results/feature_selection/feature_ranking_full.csv   — ranking de todas las features
  results/feature_selection/top_features_N.json        — top N para N in [26, 28, 33]
  results/feature_selection/overlap_with_g3.json       — solapamiento con G3
"""

import os
import sys
import json
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_FILE  = os.path.join(ROOT, "bitcoin_dataset_ULTIMATE_579_metricas.csv")
OUT_DIR    = os.path.join(ROOT, "results", "feature_selection")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── G3 reference set (our paper's 33 features) ────────────────────────────────
G3_FEATURES = [
    # OHLCV
    "Open", "High", "Low", "Close", "Volume",
    # Technical
    "RSI_14", "Stoch_K_14_3", "Stoch_D_14_3", "BB_PercentB_20", "OBV",
    "Dist_to_SMA200", "ROI30d", "Drawdown_from_ATH", "Sharpe_30d", "logret_1d",
    # On-Chain
    "MVRV", "RealizedPrice", "Short_Term_Holder_SOPR", "Supply_in_Loss",
    "Spent_Output_Profit_Ratio__SOPR_Day__1", "Net_Realized_Profit_and_Loss__NRPL",
    "UTXOs_in_Loss", "UTXOs_in_Loss_pct", "Supply_in_Loss_pct",
    "Adjusted_SOPR__aSOPR", "Price_to_Realized", "MVRV_z_365",
    "Supply_in_Profit_pct", "UTXOs_in_Profit_pct", "CapMVRVCur",
    "Supply_in_Profit", "Net_Unrealized_Loss__NUL", "Realized_Cap_UTXO_Age_Bands_pct",
]

# Columns to always exclude from feature selection
EXCLUDE_COLS = {"Date", "y", "ret_1d"}


# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    log.info(f"Loading dataset: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    log.info(f"Shape: {df.shape}  |  range: {df['Date'].min().date()} → {df['Date'].max().date()}")

    # Drop non-feature columns
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    X = df[feature_cols].copy()
    y = df["y"].copy()

    # Replace inf
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN with column median
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())

    # Drop rows where y is NaN
    mask = y.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    log.info(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
def select_top_features(X: pd.DataFrame, y: pd.Series, n_features: int) -> tuple[list[str], pd.DataFrame]:
    """
    Ultra-robust feature selection combining 8 techniques.
    Returns (top_n_feature_names, full_ranking_dataframe).
    """
    log.info(f"\n{'='*70}")
    log.info(f"Selecting TOP {n_features} features with 8 combined techniques")
    log.info(f"{'='*70}")

    # ── TECHNIQUE 1: Remove high correlation (>0.95) ──────────────────────────
    log.info("1/8  Removing highly correlated features (r > 0.95)...")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    X_filt = X.drop(columns=to_drop)
    log.info(f"     Dropped {len(to_drop)} correlated features → {X_filt.shape[1]} remaining")

    if X_filt.shape[1] <= n_features:
        log.warning(f"Only {X_filt.shape[1]} features left after correlation filter — returning all")
        return X_filt.columns.tolist(), pd.DataFrame()

    # ── TECHNIQUE 2: SelectKBest (F-statistic) ────────────────────────────────
    log.info("2/8  SelectKBest (F-statistic)...")
    selector_f = SelectKBest(f_classif, k=min(200, X_filt.shape[1]))
    selector_f.fit(X_filt, y)
    f_scores = pd.Series(selector_f.scores_, index=X_filt.columns)
    f_rank = f_scores.rank(ascending=False)
    log.info(f"     Done. Top-3: {f_scores.nlargest(3).index.tolist()}")

    # ── TECHNIQUE 3: Mutual Information ───────────────────────────────────────
    log.info("3/8  Mutual Information...")
    mi_vals = mutual_info_classif(X_filt, y, random_state=42, n_jobs=-1)
    mi_scores = pd.Series(mi_vals, index=X_filt.columns)
    mi_rank = mi_scores.rank(ascending=False)
    log.info(f"     Done. Top-3: {mi_scores.nlargest(3).index.tolist()}")

    # ── TECHNIQUE 4: Boruta ───────────────────────────────────────────────────
    log.info("4/8  Boruta...")
    try:
        from boruta import BorutaPy
        rf_boruta = RandomForestClassifier(
            n_jobs=-1, max_depth=7, random_state=42, class_weight='balanced'
        )
        boruta_sel = BorutaPy(
            estimator=rf_boruta, n_estimators='auto',
            max_iter=50, random_state=42, verbose=0
        )
        boruta_sel.fit(X_filt.values, y.values)

        boruta_rank = pd.Series(index=X_filt.columns, dtype=float)
        for i, col in enumerate(X_filt.columns):
            if boruta_sel.support_[i]:
                boruta_rank[col] = boruta_sel.ranking_[i]
            elif boruta_sel.support_weak_[i]:
                boruta_rank[col] = boruta_sel.ranking_[i] + 100
            else:
                boruta_rank[col] = boruta_sel.ranking_[i] + 1000

        confirmed = X_filt.columns[boruta_sel.support_].tolist()
        log.info(f"     Done. {len(confirmed)} confirmed features")
    except Exception as e:
        log.warning(f"     Boruta failed ({e}) — using neutral rank")
        boruta_rank = pd.Series(range(1, X_filt.shape[1] + 1), index=X_filt.columns, dtype=float)

    # ── TECHNIQUE 5: RFE ──────────────────────────────────────────────────────
    log.info("5/8  RFE (Recursive Feature Elimination)...")
    try:
        rfe_estimator = RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
        )
        n_rfe = min(n_features * 2, X_filt.shape[1])
        rfe = RFE(estimator=rfe_estimator, n_features_to_select=n_rfe, step=10)
        rfe.fit(X_filt, y)
        rfe_rank = pd.Series(rfe.ranking_, index=X_filt.columns, dtype=float)
        log.info(f"     Done. {n_rfe} features selected by RFE")
    except Exception as e:
        log.warning(f"     RFE failed ({e}) — using neutral rank")
        rfe_rank = pd.Series(range(1, X_filt.shape[1] + 1), index=X_filt.columns, dtype=float)

    # ── TECHNIQUE 6: Random Forest Importance ─────────────────────────────────
    log.info("6/8  Random Forest Importance...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=7, random_state=42,
        n_jobs=-1, class_weight='balanced'
    )
    rf.fit(X_filt, y)
    rf_imp = pd.Series(rf.feature_importances_, index=X_filt.columns)
    rf_rank = rf_imp.rank(ascending=False)
    log.info(f"     Done. Top-3: {rf_imp.nlargest(3).index.tolist()}")

    # ── TECHNIQUE 7: XGBoost Importance ───────────────────────────────────────
    log.info("7/8  XGBoost Importance...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, random_state=42,
        n_jobs=-1, verbosity=0
    )
    xgb_model.fit(X_filt, y)
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=X_filt.columns)
    xgb_rank = xgb_imp.rank(ascending=False)
    log.info(f"     Done. Top-3: {xgb_imp.nlargest(3).index.tolist()}")

    # ── TECHNIQUE 8: SHAP Values ──────────────────────────────────────────────
    log.info("8/8  SHAP Values...")
    try:
        sample_size = min(1000, len(X_filt))
        X_sample = X_filt.sample(n=sample_size, random_state=42)
        y_sample = y.loc[X_sample.index]

        shap_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, random_state=42,
            n_jobs=-1, verbosity=0
        )
        shap_model.fit(X_sample, y_sample)

        explainer = shap.TreeExplainer(shap_model)
        shap_values = explainer.shap_values(X_sample)
        shap_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=X_filt.columns)
        shap_rank = shap_imp.rank(ascending=False)
        log.info(f"     Done. Top-3: {shap_imp.nlargest(3).index.tolist()}")
    except Exception as e:
        log.warning(f"     SHAP failed ({e}) — using neutral rank")
        shap_rank = pd.Series(range(1, X_filt.shape[1] + 1), index=X_filt.columns, dtype=float)

    # ── COMBINE RANKINGS (same weights as original notebook) ──────────────────
    log.info("Combining 8 rankings with weights...")
    weights = {
        'f':      0.10,   # SelectKBest
        'mi':     0.10,   # Mutual Info
        'boruta': 0.20,   # Boruta  ← highest weight
        'rfe':    0.15,   # RFE
        'rf':     0.15,   # Random Forest
        'xgb':    0.15,   # XGBoost
        'shap':   0.15,   # SHAP    ← high weight
    }

    combined = (
        f_rank    * weights['f']      +
        mi_rank   * weights['mi']     +
        boruta_rank * weights['boruta'] +
        rfe_rank  * weights['rfe']    +
        rf_rank   * weights['rf']     +
        xgb_rank  * weights['xgb']    +
        shap_rank * weights['shap']
    ).sort_values()

    top_features = combined.head(n_features).index.tolist()

    # Full ranking dataframe
    ranking_df = pd.DataFrame({
        'feature':      combined.index,
        'combined_rank': combined.values,
        'f_rank':       f_rank.reindex(combined.index).values,
        'mi_rank':      mi_rank.reindex(combined.index).values,
        'boruta_rank':  boruta_rank.reindex(combined.index).values,
        'rfe_rank':     rfe_rank.reindex(combined.index).values,
        'rf_rank':      rf_rank.reindex(combined.index).values,
        'xgb_rank':     xgb_rank.reindex(combined.index).values,
        'shap_rank':    shap_rank.reindex(combined.index).values,
        'in_g3':        [f in G3_FEATURES for f in combined.index],
    })

    return top_features, ranking_df


# ─────────────────────────────────────────────────────────────────────────────
def compute_overlap(top_features: list[str], n: int) -> dict:
    """Compute overlap statistics between top_features and G3."""
    overlap = [f for f in top_features if f in G3_FEATURES]
    only_top = [f for f in top_features if f not in G3_FEATURES]
    only_g3  = [f for f in G3_FEATURES if f not in top_features]
    return {
        "n_requested": n,
        "top_features": top_features,
        "overlap_count": len(overlap),
        "overlap_pct": round(len(overlap) / n * 100, 1),
        "overlap_features": overlap,
        "in_top_not_g3": only_top,
        "in_g3_not_top": only_g3,
    }


# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("FEATURE SELECTION PIPELINE — 579 metrics → reproducible ranking")
    log.info("=" * 70)

    X, y = load_data()

    # Run selection once and reuse rankings for all N values
    # We use max(N) = 33 for the main run, then just take head(N) for smaller
    N_LIST = [26, 28, 33]
    N_MAX = max(N_LIST)

    top_features, ranking_df = select_top_features(X, y, n_features=N_MAX)

    # ── Save full ranking CSV ──────────────────────────────────────────────────
    ranking_path = os.path.join(OUT_DIR, "feature_ranking_full.csv")
    ranking_df.to_csv(ranking_path, index=False)
    log.info(f"\nFull ranking saved → {ranking_path}")

    # ── Save per-N results ─────────────────────────────────────────────────────
    overlap_results = {}
    for n in N_LIST:
        top_n = ranking_df.head(n)['feature'].tolist()
        overlap = compute_overlap(top_n, n)
        overlap_results[str(n)] = overlap

        path = os.path.join(OUT_DIR, f"top_features_{n}.json")
        with open(path, "w") as f:
            json.dump(overlap, f, indent=2)
        log.info(f"Top-{n} saved → {path}")

    # ── Save combined overlap summary ─────────────────────────────────────────
    summary_path = os.path.join(OUT_DIR, "overlap_with_g3.json")
    with open(summary_path, "w") as f:
        json.dump(overlap_results, f, indent=2)
    log.info(f"Overlap summary saved → {summary_path}")

    # ── Print results ──────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("RESULTS SUMMARY")
    log.info("=" * 70)
    log.info(f"\nG3 has {len(G3_FEATURES)} features")
    log.info(f"\nFull ranking — top 40 features:")
    for i, row in ranking_df.head(40).iterrows():
        marker = "✓ G3" if row['in_g3'] else "    "
        log.info(f"  {i+1:3d}. {marker}  {row['feature']:<55}  score={row['combined_rank']:.2f}")

    log.info("\n" + "-" * 70)
    for n in N_LIST:
        r = overlap_results[str(n)]
        log.info(f"\nTop-{n}:  overlap with G3 = {r['overlap_count']}/{n} ({r['overlap_pct']}%)")
        log.info(f"  In top-{n} but NOT in G3: {r['in_top_not_g3']}")
        log.info(f"  In G3 but NOT in top-{n}: {r['in_g3_not_top']}")

    log.info("\nDone.")


if __name__ == "__main__":
    main()
