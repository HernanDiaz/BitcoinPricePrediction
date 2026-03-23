"""
Random Forest classifier — M1 baseline model.
Uses sklearn with balanced class weights and full reproducibility.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseModel


class RandomForestModel(BaseModel):
    """
    Baseline ensemble model. Accepts flat feature arrays (no sequences).
    """

    def __init__(self, config: dict, random_seed: int = 42):
        super().__init__(name="RandomForest", random_seed=random_seed)
        cfg = config["models"]["random_forest"]
        self.clf = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_split=cfg["min_samples_split"],
            min_samples_leaf=cfg["min_samples_leaf"],
            max_features=cfg["max_features"],
            class_weight=cfg["class_weight"],
            n_jobs=cfg["n_jobs"],
            random_state=random_seed,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        class_weight: float = 1.0,
    ) -> dict:
        # RF uses class_weight="balanced" internally; class_weight param not used
        self.clf.fit(X_train, y_train)
        return {"model": "RandomForest", "n_estimators": self.clf.n_estimators}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.clf, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            self.clf = pickle.load(f)
