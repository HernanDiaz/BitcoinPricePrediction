"""
CNN-LSTM single-encoder classifier — M3 deep learning baseline.
Processes a single feature set through CNN feature extraction
followed by stacked LSTM temporal modeling.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseModel
from .torch_utils import (
    EarlyStopping,
    set_all_seeds,
    get_device,
    make_weighted_sampler,
)


class CNNLSTMEncoder(nn.Module):
    """Shared encoder: Conv1D feature extraction + stacked LSTM."""

    def __init__(
        self,
        n_features: int,
        conv_filters: int,
        conv_kernel_size: int,
        lstm_hidden: int,
        lstm_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, conv_filters, kernel_size=conv_kernel_size, padding=1),
            nn.BatchNorm1d(conv_filters),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_filters, conv_filters, kernel_size=conv_kernel_size, padding=1),
            nn.BatchNorm1d(conv_filters),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            conv_filters,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, n_features)
        out = x.transpose(1, 2)        # (B, n_features, seq_len)
        out = self.conv(out)            # (B, conv_filters, seq_len)
        out = out.transpose(1, 2)      # (B, seq_len, conv_filters)
        out, _ = self.lstm(out)        # (B, seq_len, lstm_hidden)
        return self.dropout(out)


class CNNLSTMNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        conv_filters: int,
        conv_kernel_size: int,
        lstm_hidden: int,
        lstm_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = CNNLSTMEncoder(
            n_features, conv_filters, conv_kernel_size,
            lstm_hidden, lstm_layers, dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)           # (B, seq_len, lstm_hidden)
        pooled = encoded.mean(dim=1)        # (B, lstm_hidden)
        return self.classifier(pooled)      # (B, 1)


class CNNLSTMModel(BaseModel):
    """
    Single-encoder CNN-LSTM classifier.
    Accepts sequential arrays of shape (n_samples, seq_len, n_features).
    """

    def __init__(self, config: dict, n_features: int, random_seed: int = 42):
        super().__init__(name="CNN-LSTM", random_seed=random_seed)
        self._cfg = config["models"]["cnn_lstm"]
        self._n_features = n_features
        self._device = get_device(config)
        self._net: CNNLSTMNet | None = None
        self.training_history: list[dict] = []

    def _build_net(self) -> CNNLSTMNet:
        cfg = self._cfg
        return CNNLSTMNet(
            n_features=self._n_features,
            conv_filters=cfg["conv_filters"],
            conv_kernel_size=cfg["conv_kernel_size"],
            lstm_hidden=cfg["lstm_hidden"],
            lstm_layers=cfg["lstm_layers"],
            dropout=cfg["dropout"],
        ).to(self._device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        class_weight: float = 1.0,
    ) -> dict:
        set_all_seeds(self.random_seed)
        cfg = self._cfg

        self._net = self._build_net()
        optimizer = torch.optim.Adam(
            self._net.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=cfg["lr_scheduler_patience"],
            factor=cfg["lr_scheduler_factor"],
            min_lr=cfg["lr_min"],
        )
        pos_weight = torch.tensor([class_weight], device=self._device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        early_stop = EarlyStopping(patience=cfg["early_stopping_patience"])

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False) if X_val is not None else None

        self.training_history = []
        for epoch in range(cfg["epochs"]):
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, criterion)
            history_entry = {"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc}

            if val_loader is not None:
                val_loss, val_acc = self._eval_epoch(val_loader, criterion)
                history_entry.update({"val_loss": val_loss, "val_acc": val_acc})
                scheduler.step(val_loss)
                if early_stop(val_loss, self._net):
                    history_entry["early_stop"] = True
                    self.training_history.append(history_entry)
                    break
            else:
                scheduler.step(train_loss)

            self.training_history.append(history_entry)

        # Restore best weights
        early_stop.restore_best(self._net)
        return {"model": "CNN-LSTM", "epochs_trained": len(self.training_history)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._net.eval()
        loader = self._make_loader(X, None, shuffle=False)
        all_probs = []
        with torch.no_grad():
            for (x_batch,) in loader:
                x_batch = x_batch.to(self._device)
                logits = self._net(x_batch).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
        probs = np.concatenate(all_probs)
        return np.column_stack([1 - probs, probs])

    def _train_epoch(self, loader, optimizer, criterion):
        self._net.train()
        total_loss, correct, total = 0.0, 0, 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(self._device), y_batch.to(self._device)
            optimizer.zero_grad()
            logits = self._net(x_batch).squeeze(-1)
            loss = criterion(logits, y_batch.float())
            loss.backward()
            nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
        return total_loss / total, correct / total

    def _eval_epoch(self, loader, criterion):
        self._net.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self._device), y_batch.to(self._device)
                logits = self._net(x_batch).squeeze(-1)
                loss = criterion(logits, y_batch.float())
                total_loss += loss.item() * len(y_batch)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)
        return total_loss / total, correct / total

    def _make_loader(self, X, y, shuffle: bool) -> DataLoader:
        cfg = self._cfg
        X_t = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            y_t = torch.tensor(y, dtype=torch.long)
            dataset = TensorDataset(X_t, y_t)
        else:
            dataset = TensorDataset(X_t)
        return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=shuffle, pin_memory=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self._net.state_dict(), "n_features": self._n_features}, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self._device)
        self._net = self._build_net()
        self._net.load_state_dict(checkpoint["state_dict"])
        self._net.eval()
