"""
Abstract base class for all models in the ablation study.
Enforces a common interface: fit, predict, predict_proba, save, load.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseModel(ABC):
    """
    Common interface for all classifiers (sklearn-based and PyTorch-based).
    """

    def __init__(self, name: str, random_seed: int = 42):
        self.name = name
        self.random_seed = random_seed

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        class_weight: float = 1.0,
    ) -> dict:
        """
        Train the model. Returns a dict with training history/metadata.
        X_train shape depends on model type:
          - Sklearn models : (n_samples, n_features)
          - PyTorch models : (n_samples, seq_len, n_features)
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (0/1) of shape (n_samples,)."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities of shape (n_samples, 2)."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize the model to disk."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore the model from disk."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, seed={self.random_seed})"
