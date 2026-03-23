"""
XGBoost classifier — M2 gradient boosting model.
scale_pos_weight computed per fold from actual class distribution.
Supports early stopping with a validation set.
"""

import pickle
from pathlib import Path

import numpy as np
import xgboost as xgb

from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    Gradient boosting model. Accepts flat feature arrays (no sequences).
    """

    def __init__(self, config: dict, random_seed: int = 42):
        super().__init__(name="XGBoost", random_seed=random_seed)
        self._cfg = config["models"]["xgboost"]
        self.clf: xgb.XGBClassifier | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        class_weight: float = 1.0,
    ) -> dict:
        cfg = self._cfg
        self.clf = xgb.XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            min_child_weight=cfg["min_child_weight"],
            gamma=cfg["gamma"],
            reg_alpha=cfg["reg_alpha"],
            reg_lambda=cfg["reg_lambda"],
            scale_pos_weight=class_weight,
            eval_metric=cfg["eval_metric"],
            early_stopping_rounds=cfg["early_stopping_rounds"] if X_val is not None else None,
            random_state=self.random_seed,
            device="cpu",  # XGBoost CPU; GPU reserved for PyTorch models
        )

        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.clf.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        best_iter = self.clf.best_iteration if hasattr(self.clf, "best_iteration") else cfg["n_estimators"]
        return {"model": "XGBoost", "best_iteration": best_iter}

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
