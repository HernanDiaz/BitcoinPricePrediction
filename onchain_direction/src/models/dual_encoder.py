"""
Dual-Encoder with Cross-Attention — M4, the novel architecture.

Scientific contribution:
  Processes technical indicators and on-chain metrics through independent
  CNN-LSTM encoders, then fuses them via bidirectional cross-attention.
  Each branch attends to the other, learning WHEN and HOW MUCH each
  information domain contributes to the prediction.

  The cross-attention weights are the interpretability mechanism:
  - tech_weights[b, i, j]: how much technical timestep i attends to on-chain timestep j
  - onchain_weights[b, i, j]: how much on-chain timestep i attends to technical timestep j

Architecture:
  Technical (seq, n_tech) ──► CNN-LSTM ──► cross-attn (Q=tech, K/V=onchain) ──┐
                                                                                ├─► concat ──► MLP ──► logit
  On-chain  (seq, n_on)  ──► CNN-LSTM ──► cross-attn (Q=onchain, K/V=tech)  ──┘
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseModel
from .cnn_lstm import CNNLSTMEncoder
from .torch_utils import EarlyStopping, get_device, set_all_seeds


class DualEncoderNet(nn.Module):
    """
    Core PyTorch module for the dual-encoder architecture.
    """

    def __init__(
        self,
        n_technical: int,
        n_onchain: int,
        conv_filters: int,
        conv_kernel_size: int,
        lstm_hidden: int,
        lstm_layers: int,
        n_attention_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.lstm_hidden = lstm_hidden

        # Independent branch encoders
        self.tech_encoder = CNNLSTMEncoder(
            n_features=n_technical,
            conv_filters=conv_filters,
            conv_kernel_size=conv_kernel_size,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
        )
        self.onchain_encoder = CNNLSTMEncoder(
            n_features=n_onchain,
            conv_filters=conv_filters,
            conv_kernel_size=conv_kernel_size,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
        )

        # Cross-attention: technical queries ← on-chain keys/values
        self.cross_attn_tech = nn.MultiheadAttention(
            embed_dim=lstm_hidden,
            num_heads=n_attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention: on-chain queries ← technical keys/values
        self.cross_attn_onchain = nn.MultiheadAttention(
            embed_dim=lstm_hidden,
            num_heads=n_attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalisation (post-attention residual connections)
        self.norm_tech = nn.LayerNorm(lstm_hidden)
        self.norm_onchain = nn.LayerNorm(lstm_hidden)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x_tech: torch.Tensor,
        x_onchain: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_tech    : (B, seq_len, n_technical)
            x_onchain : (B, seq_len, n_onchain)

        Returns:
            logits           : (B, 1)
            tech_attn_weights: (B, seq_len, seq_len)  — for interpretability
            onchain_attn_weights: (B, seq_len, seq_len)
        """
        # Independent encoding
        tech_seq = self.tech_encoder(x_tech)        # (B, seq_len, H)
        onchain_seq = self.onchain_encoder(x_onchain)  # (B, seq_len, H)

        # Bidirectional cross-attention with residual connections
        tech_attended, tech_weights = self.cross_attn_tech(
            query=tech_seq, key=onchain_seq, value=onchain_seq
        )
        tech_out = self.norm_tech(tech_seq + tech_attended)   # (B, seq_len, H)

        onchain_attended, onchain_weights = self.cross_attn_onchain(
            query=onchain_seq, key=tech_seq, value=tech_seq
        )
        onchain_out = self.norm_onchain(onchain_seq + onchain_attended)  # (B, seq_len, H)

        # Global average pooling over the time dimension
        tech_pooled = tech_out.mean(dim=1)        # (B, H)
        onchain_pooled = onchain_out.mean(dim=1)  # (B, H)

        # Concatenate and classify
        fused = torch.cat([tech_pooled, onchain_pooled], dim=-1)  # (B, 2H)
        logits = self.classifier(fused)                            # (B, 1)

        return logits, tech_weights, onchain_weights


class DualEncoderModel(BaseModel):
    """
    Dual-encoder with cross-attention. Only valid for the full feature set (G3),
    where both technical and on-chain features are available.

    Requires the feature arrays to be pre-split by the caller using
    data.preprocessor.split_dual_encoder_features().
    """

    def __init__(
        self,
        config: dict,
        n_technical: int,
        n_onchain: int,
        random_seed: int = 42,
    ):
        super().__init__(name="DualEncoder", random_seed=random_seed)
        self._cfg = config["models"]["dual_encoder"]
        self._n_technical = n_technical
        self._n_onchain = n_onchain
        self._device = get_device(config)
        self._net: DualEncoderNet | None = None
        self.training_history: list[dict] = []

        # Stored for post-hoc interpretability (last inference pass)
        self._last_tech_weights: np.ndarray | None = None
        self._last_onchain_weights: np.ndarray | None = None

    def _build_net(self) -> DualEncoderNet:
        cfg = self._cfg
        return DualEncoderNet(
            n_technical=self._n_technical,
            n_onchain=self._n_onchain,
            conv_filters=cfg["conv_filters"],
            conv_kernel_size=cfg["conv_kernel_size"],
            lstm_hidden=cfg["lstm_hidden"],
            lstm_layers=cfg["lstm_layers"],
            n_attention_heads=cfg["n_attention_heads"],
            dropout=cfg["dropout"],
        ).to(self._device)

    def fit(
        self,
        X_train: tuple[np.ndarray, np.ndarray],
        y_train: np.ndarray,
        X_val: tuple[np.ndarray, np.ndarray] | None = None,
        y_val: np.ndarray | None = None,
        class_weight: float = 1.0,
    ) -> dict:
        """
        X_train and X_val must be tuples: (X_tech, X_onchain)
        each of shape (n_samples, seq_len, n_features_branch).
        """
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
            entry = {"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc}

            if val_loader is not None:
                val_loss, val_acc = self._eval_epoch(val_loader, criterion)
                entry.update({"val_loss": val_loss, "val_acc": val_acc})
                scheduler.step(val_loss)
                if early_stop(val_loss, self._net):
                    entry["early_stop"] = True
                    self.training_history.append(entry)
                    break
            else:
                scheduler.step(train_loss)

            self.training_history.append(entry)

        early_stop.restore_best(self._net)
        return {"model": "DualEncoder", "epochs_trained": len(self.training_history)}

    def predict(self, X: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(np.int64)

    def predict_proba(self, X: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        self._net.eval()
        loader = self._make_loader(X, None, shuffle=False)
        all_probs, all_tech_w, all_onchain_w = [], [], []

        with torch.no_grad():
            for batch in loader:
                xt, xo = batch[0].to(self._device), batch[1].to(self._device)
                logits, tw, ow = self._net(xt, xo)
                probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                all_probs.append(probs)
                all_tech_w.append(tw.cpu().numpy())
                all_onchain_w.append(ow.cpu().numpy())

        self._last_tech_weights = np.concatenate(all_tech_w, axis=0)
        self._last_onchain_weights = np.concatenate(all_onchain_w, axis=0)

        probs = np.concatenate(all_probs)
        return np.column_stack([1 - probs, probs])

    def get_attention_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (tech_weights, onchain_weights) from the last predict_proba call.
        Shape: (n_samples, n_heads, seq_len, seq_len) averaged across heads.
        """
        return self._last_tech_weights, self._last_onchain_weights

    def _train_epoch(self, loader, optimizer, criterion):
        self._net.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in loader:
            xt, xo, yb = batch[0].to(self._device), batch[1].to(self._device), batch[2].to(self._device)
            optimizer.zero_grad()
            logits, _, _ = self._net(xt, xo)
            loss = criterion(logits.squeeze(-1), yb.float())
            loss.backward()
            nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)
            preds = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long()
            correct += (preds == yb).sum().item()
            total += len(yb)
        return total_loss / total, correct / total

    def _eval_epoch(self, loader, criterion):
        self._net.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in loader:
                xt, xo, yb = batch[0].to(self._device), batch[1].to(self._device), batch[2].to(self._device)
                logits, _, _ = self._net(xt, xo)
                loss = criterion(logits.squeeze(-1), yb.float())
                total_loss += loss.item() * len(yb)
                preds = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long()
                correct += (preds == yb).sum().item()
                total += len(yb)
        return total_loss / total, correct / total

    def _make_loader(self, X, y, shuffle: bool) -> DataLoader:
        cfg = self._cfg
        if X is None:
            return None
        X_tech_t = torch.tensor(X[0], dtype=torch.float32)
        X_onchain_t = torch.tensor(X[1], dtype=torch.float32)
        if y is not None:
            y_t = torch.tensor(y, dtype=torch.long)
            dataset = TensorDataset(X_tech_t, X_onchain_t, y_t)
        else:
            dataset = TensorDataset(X_tech_t, X_onchain_t)
        return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=shuffle, pin_memory=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "n_technical": self._n_technical,
                "n_onchain": self._n_onchain,
            },
            path,
        )

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self._device)
        self._n_technical = checkpoint["n_technical"]
        self._n_onchain = checkpoint["n_onchain"]
        self._net = self._build_net()
        self._net.load_state_dict(checkpoint["state_dict"])
        self._net.eval()
