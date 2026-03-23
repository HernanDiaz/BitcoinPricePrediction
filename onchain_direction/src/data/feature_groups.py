"""
Feature group registry for ablation study.

Defines the feature groups used in the ablation experiment:
  G0 — OHLCV only (pure market baseline)
  G1 — OHLCV + Technical Indicators
  G2 — OHLCV + On-Chain Metrics (Glassnode/original)
  G3 — Full original set (G1 union G2)
  G4 — OHLCV + On-Chain + Sentiment + Mining/Network (enriched)
  G5 — Full enriched set (G1 union G4)

The DualEncoder model uses the technical and on-chain sub-lists
to route features to independent encoders.
"""

from dataclasses import dataclass

import pandas as pd


OHLCV_FEATURES = ["Open", "High", "Low", "Close", "Volume"]

TECHNICAL_FEATURES = [
    "RSI_14",
    "Stoch_K_14_3",
    "Stoch_D_14_3",
    "BB_PercentB_20",
    "OBV",
    "Dist_to_SMA200",
    "ROI30d",
    "Drawdown_from_ATH",
    "Sharpe_30d",
    "logret_1d",
]

ONCHAIN_FEATURES = [
    "MVRV",
    "RealizedPrice",
    "Short_Term_Holder_SOPR",
    "Supply_in_Loss",
    "Spent_Output_Profit_Ratio__SOPR_Day__1",
    "Net_Realized_Profit_and_Loss__NRPL",
    "UTXOs_in_Loss",
    "UTXOs_in_Loss_pct",
    "Supply_in_Loss_pct",
    "Adjusted_SOPR__aSOPR",
    "Price_to_Realized",
    "MVRV_z_365",
    "Supply_in_Profit_pct",
    "UTXOs_in_Profit_pct",
    "CapMVRVCur",
    "Supply_in_Profit",
    "Net_Unrealized_Loss__NUL",
    "Realized_Cap_UTXO_Age_Bands_pct",
]

# New features from blockchain.info + alternative.me (enriched dataset)
SENTIMENT_FEATURES = [
    "FearGreed",
    "FearGreed_7d_MA",
    "FearGreed_30d_MA",
    "FearGreed_zone",
]

MINING_NETWORK_FEATURES = [
    "HashRate",
    "Difficulty",
    "HashRate_30d_MA",
    "HashRate_60d_MA",
    "HashRibbon",
    "MinerRevUSD",
    "PuellMultiple",
    "TxCount",
    "TxCount_30d_MA",
    "TxCount_momentum",
    "TxVolumeUSD",
    "TxVolume_30d_MA",
    "NVT_approx",
    "NVT_ratio",
]

# All new enrichment features together
ENRICHED_FEATURES = SENTIMENT_FEATURES + MINING_NETWORK_FEATURES


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    label: str          # short label for plots/tables
    description: str
    features: tuple[str, ...]


FEATURE_GROUPS: dict[str, FeatureGroup] = {
    "G0": FeatureGroup(
        name="G0",
        label="OHLCV",
        description="Raw market data only (baseline)",
        features=tuple(OHLCV_FEATURES),
    ),
    "G1": FeatureGroup(
        name="G1",
        label="OHLCV+Tech",
        description="OHLCV + classical technical indicators",
        features=tuple(OHLCV_FEATURES + TECHNICAL_FEATURES),
    ),
    "G2": FeatureGroup(
        name="G2",
        label="OHLCV+OnChain",
        description="OHLCV + blockchain on-chain metrics (Glassnode)",
        features=tuple(OHLCV_FEATURES + ONCHAIN_FEATURES),
    ),
    "G3": FeatureGroup(
        name="G3",
        label="Full",
        description="Complete original set (OHLCV + Technical + On-Chain)",
        features=tuple(OHLCV_FEATURES + TECHNICAL_FEATURES + ONCHAIN_FEATURES),
    ),
    "G4": FeatureGroup(
        name="G4",
        label="OHLCV+OnChain+Enriched",
        description="OHLCV + On-Chain + Sentiment + Mining/Network metrics",
        features=tuple(OHLCV_FEATURES + ONCHAIN_FEATURES + ENRICHED_FEATURES),
    ),
    "G5": FeatureGroup(
        name="G5",
        label="Full+Enriched",
        description="Full enriched set (G3 + Sentiment + Mining/Network)",
        features=tuple(OHLCV_FEATURES + TECHNICAL_FEATURES + ONCHAIN_FEATURES + ENRICHED_FEATURES),
    ),
}


def get_group(name: str) -> FeatureGroup:
    """Return a FeatureGroup by name (G0–G3). Raises KeyError if not found."""
    if name not in FEATURE_GROUPS:
        raise KeyError(f"Unknown feature group '{name}'. Valid: {list(FEATURE_GROUPS)}")
    return FEATURE_GROUPS[name]


def validate_group_against_df(group: FeatureGroup, df: pd.DataFrame) -> list[str]:
    """
    Verify all features in the group exist in the dataframe.
    Returns the list of valid features (silently drops any missing ones with a warning).
    """
    import logging
    logger = logging.getLogger(__name__)

    valid = []
    for feat in group.features:
        if feat in df.columns:
            valid.append(feat)
        else:
            logger.warning(f"[{group.name}] Feature '{feat}' not found in dataset — skipped.")
    return valid


def get_dual_encoder_splits(df_columns: list[str],
                             enriched: bool = False) -> tuple[list[str], list[str]]:
    """
    For the DualEncoder model, return (technical_cols, onchain_cols)
    that are actually present in df_columns.
    OHLCV features are routed to the technical branch.
    If enriched=True, sentiment/mining features are added to the onchain branch.
    """
    tech_cols = OHLCV_FEATURES + TECHNICAL_FEATURES
    onchain_cols = ONCHAIN_FEATURES
    if enriched:
        onchain_cols = onchain_cols + ENRICHED_FEATURES
    tech_present    = [f for f in tech_cols    if f in df_columns]
    onchain_present = [f for f in onchain_cols if f in df_columns]
    return tech_present, onchain_present
