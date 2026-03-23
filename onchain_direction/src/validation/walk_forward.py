"""
Walk-forward expanding window cross-validation for time series.

Each fold trains on all data from the dataset start up to train_end,
and tests on the next calendar year. This is the only valid CV strategy
for financial time series — it prevents any form of temporal data leakage.

Fold structure (7 folds, 2019–2025):
  Fold 1: Train 2013–2018 | Test 2019
  Fold 2: Train 2013–2019 | Test 2020
  ...
  Fold 7: Train 2013–2024 | Test 2025
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def label(self) -> str:
        return f"Fold{self.fold}_{self.test_start.year}"

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (df_train, df_test) for this fold."""
        df_train = df.loc[
            (df.index >= self.train_start) & (df.index <= self.train_end)
        ]
        df_test = df.loc[
            (df.index >= self.test_start) & (df.index <= self.test_end)
        ]

        if len(df_train) == 0:
            raise ValueError(
                f"[{self.label}] Training set is empty. "
                f"Check date range {self.train_start} → {self.train_end}."
            )
        if len(df_test) == 0:
            raise ValueError(
                f"[{self.label}] Test set is empty. "
                f"Check date range {self.test_start} → {self.test_end}."
            )

        return df_train, df_test

    def __str__(self) -> str:
        return (
            f"Fold {self.fold}: "
            f"train [{self.train_start.date()} -> {self.train_end.date()}] "
            f"({(self.train_end - self.train_start).days} days)  |  "
            f"test [{self.test_start.date()} -> {self.test_end.date()}] "
            f"({(self.test_end - self.test_start).days} days)"
        )


class WalkForwardCV:
    """
    Generates walk-forward folds from config.yaml specification.

    Usage:
        cv = WalkForwardCV(config)
        for fold in cv.folds:
            df_train, df_test = fold.split(df)
    """

    def __init__(self, config: dict):
        wf_cfg = config["walk_forward"]
        global_train_start = pd.Timestamp(wf_cfg["train_start"])

        self.folds: list[WalkForwardFold] = []
        for fold_cfg in wf_cfg["folds"]:
            self.folds.append(
                WalkForwardFold(
                    fold=fold_cfg["fold"],
                    train_start=global_train_start,
                    train_end=pd.Timestamp(fold_cfg["train_end"]),
                    test_start=pd.Timestamp(fold_cfg["test_start"]),
                    test_end=pd.Timestamp(fold_cfg["test_end"]),
                )
            )

    def __len__(self) -> int:
        return len(self.folds)

    def __iter__(self):
        return iter(self.folds)

    def summary(self) -> str:
        lines = ["Walk-Forward CV Summary", "=" * 70]
        for fold in self.folds:
            lines.append(str(fold))
        lines.append(f"\nTotal folds: {len(self.folds)}")
        return "\n".join(lines)
