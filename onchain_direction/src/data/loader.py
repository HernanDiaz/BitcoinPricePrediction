"""
Dataset loader with validation and quality reporting.
Loads the Bitcoin OHLCV + technical + on-chain dataset and ensures
data integrity before any experiment.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset(config: dict) -> pd.DataFrame:
    """
    Load and validate the Bitcoin dataset.

    Returns a cleaned DataFrame indexed by Date with all features and target.
    Raises ValueError on integrity failures.
    """
    dataset_path = Path(__file__).parents[3] / config["paths"]["dataset"]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    date_col = config["data"]["date_column"]
    target_col = config["data"]["target_column"]

    df = pd.read_csv(dataset_path, parse_dates=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.set_index(date_col)

    _validate_dataset(df, target_col)
    _log_dataset_summary(df, target_col)

    return df


def _validate_dataset(df: pd.DataFrame, target_col: str) -> None:
    """Hard checks that abort execution if the dataset is corrupted."""
    if df.index.duplicated().any():
        raise ValueError("Dataset contains duplicate dates.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Dataset is not sorted by date.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    if not set(df[target_col].dropna().unique()).issubset({0, 1}):
        raise ValueError(f"Target '{target_col}' must be binary (0/1).")

    n_nan = df.isna().sum().sum()
    n_total = df.shape[0] * df.shape[1]
    nan_pct = n_nan / n_total * 100
    if nan_pct > 20:
        raise ValueError(
            f"Dataset has {nan_pct:.1f}% missing values — exceeds 20% threshold."
        )


def _log_dataset_summary(df: pd.DataFrame, target_col: str) -> None:
    """Log dataset statistics for reproducibility tracking."""
    class_counts = df[target_col].value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()

    logger.info("=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Date range   : {df.index.min().date()} -> {df.index.max().date()}")
    logger.info(f"  Total rows   : {len(df):,}")
    logger.info(f"  Features     : {df.shape[1] - 1}")
    logger.info(f"  Target (y=1) : {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/len(df)*100:.1f}%)")
    logger.info(f"  Target (y=0) : {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/len(df)*100:.1f}%)")
    logger.info(f"  Imbalance    : {imbalance_ratio:.2f}x")
    logger.info(f"  Missing vals : {df.isna().sum().sum():,} ({df.isna().sum().sum()/df.size*100:.2f}%)")
    logger.info("=" * 60)


def get_feature_columns(df: pd.DataFrame, config: dict, group_name: str) -> list[str]:
    """
    Return the list of feature columns for a given feature group name
    (G0, G1, G2, or G3).
    """
    target_col = config["data"]["target_column"]
    group_cfg = config["feature_groups"][group_name]

    if group_cfg["features"] == "all":
        return [c for c in df.columns if c != target_col]

    # Verify all requested features exist in the dataframe
    requested = group_cfg["features"]
    missing = [f for f in requested if f not in df.columns]
    if missing:
        raise ValueError(
            f"Feature group {group_name} requests columns not in dataset: {missing}"
        )

    return list(requested)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill then backward-fill missing values.
    This is the correct strategy for financial time series:
    carry the last known value forward, then fill any leading NaNs backward.
    Applied BEFORE any train/test split to avoid leakage on the fill itself
    (forward fill uses only past values).
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    remaining = df[numeric_cols].isna().sum().sum()
    if remaining > 0:
        logger.warning(f"After imputation, {remaining} NaN values remain — dropping rows.")
        df = df.dropna(subset=numeric_cols)

    return df
