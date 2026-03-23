"""
Preprocessing pipeline with strict no-leakage guarantees.

Rules enforced:
  1. Scaler is fit ONLY on training data, then applied to test.
  2. Sequences are created AFTER scaling.
  3. Class weights are computed ONLY from training labels.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


class FoldPreprocessor:
    """
    Stateful preprocessor for a single walk-forward fold.

    Usage:
        prep = FoldPreprocessor(sequence_length=30)
        X_train, y_train = prep.fit_transform(df_train, feature_cols, target_col)
        X_test, y_test   = prep.transform(df_test, feature_cols, target_col)

    Returns flat arrays for sklearn models and sequenced arrays for PyTorch models.
    """

    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.scaler = RobustScaler()   # Robust to outliers — important for financial data
        self._fitted = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the scaler on training data and return both flat and sequential arrays.

        Returns:
            X_flat  : (n_samples, n_features)          for RF / XGBoost
            X_seq   : (n_samples, seq_len, n_features) for CNN-LSTM / DualEncoder
            y_flat  : (n_samples,)                     labels for flat models
            y_seq   : (n_samples,)                     labels aligned with sequences
        """
        # Drop rows where target is NaN before fitting
        df = df.dropna(subset=[target_col])
        X_raw = df[feature_cols].values.astype(np.float32)
        y_raw = df[target_col].values.astype(np.int64)

        X_scaled = self.scaler.fit_transform(X_raw)
        self._fitted = True

        X_seq, y_seq = self._make_sequences(X_scaled, y_raw)
        # Flat arrays are aligned with sequence output (drop first seq_len-1 rows)
        X_flat = X_scaled[self.sequence_length - 1:]
        y_flat = y_raw[self.sequence_length - 1:]

        return X_flat, X_seq, y_flat, y_seq

    def transform(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply the fitted scaler to test data. Must call fit_transform first.
        """
        if not self._fitted:
            raise RuntimeError("FoldPreprocessor must be fit before calling transform.")

        # Drop rows where target is NaN
        df = df.dropna(subset=[target_col])
        X_raw = df[feature_cols].values.astype(np.float32)
        y_raw = df[target_col].values.astype(np.int64)

        X_scaled = self.scaler.transform(X_raw)

        X_seq, y_seq = self._make_sequences(X_scaled, y_raw)
        X_flat = X_scaled[self.sequence_length - 1:]
        y_flat = y_raw[self.sequence_length - 1:]

        return X_flat, X_seq, y_flat, y_seq

    def _make_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create rolling windows of shape (n_samples, seq_len, n_features).
        The label for each window is the label of the LAST day in the window.
        """
        n = len(X)
        seq_len = self.sequence_length
        n_seq = n - seq_len + 1

        X_seq = np.lib.stride_tricks.sliding_window_view(X, (seq_len, X.shape[1]))
        X_seq = X_seq.squeeze(axis=1).astype(np.float32)

        y_seq = y[seq_len - 1:].astype(np.int64)

        assert len(X_seq) == n_seq
        assert len(y_seq) == n_seq

        return X_seq, y_seq

    def compute_class_weight(self, y_train: np.ndarray) -> float:
        """
        Compute scale_pos_weight for XGBoost (ratio negative/positive).
        Also usable as weight tensor for PyTorch BCEWithLogitsLoss.
        """
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        if n_pos == 0:
            return 1.0
        return float(n_neg / n_pos)


def split_dual_encoder_features(
    X_seq: np.ndarray,
    feature_cols: list[str],
    tech_cols: list[str],
    onchain_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a sequence array (n, seq_len, n_features) into two arrays:
    one for the technical branch and one for the on-chain branch.

    Used exclusively by the DualEncoder model.
    """
    tech_idx = [feature_cols.index(c) for c in tech_cols if c in feature_cols]
    onchain_idx = [feature_cols.index(c) for c in onchain_cols if c in feature_cols]

    X_tech = X_seq[:, :, tech_idx]
    X_onchain = X_seq[:, :, onchain_idx]

    return X_tech, X_onchain
