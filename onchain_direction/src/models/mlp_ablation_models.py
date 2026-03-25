"""
mlp_ablation_models.py
======================
Ablation variants for the MLP Dual Encoder V2 architecture.

Three variants to isolate the contribution of each component:
  1. MLPSimpleModel       - Single MLP on concatenated features (no dual encoder, no attention)
  2. MLPDualNoAttnModel   - Dual encoder with concatenation fusion (no cross-attention)
  3. MLPDualEncoderModelV2 - Full model with cross-attention (already in mlp_dual_encoder_v2.py)

All variants share the same training loop and accept the same (X_tech, X_onchain) tuple
interface so the ablation script can treat them identically.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from .base_model import BaseModel
from .torch_utils import EarlyStopping, set_all_seeds
from .mlp_dual_encoder_v2 import MLPEncoderV2


# ---------------------------------------------------------------------------
# Shared training mixin — identical loop used by all ablation variants
# ---------------------------------------------------------------------------

class _MLPTrainerMixin:
    """
    Shared training / inference logic.
    Subclasses must implement:
      - _build_net() -> nn.Module  (returns the raw torch network)
      - _forward(net, xt, xo)     (runs a forward pass, returns logits tensor)
      - _make_loader(X, y, shuffle, batch_size)
    """

    def fit(self, X_train, y_train, X_val=None, y_val=None, class_weight=1.0):
        set_all_seeds(self.random_seed)
        c = self._cfg

        self._net = self._build_net()
        optimizer = torch.optim.AdamW(
            self._net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"]
        )

        total_epochs = c["epochs"]
        warmup = c.get("warmup_epochs", 10)

        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(total_epochs - warmup, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        pos_weight = torch.tensor([class_weight], device=self._device)
        label_smooth = c.get("label_smoothing", 0.05)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        early_stop = EarlyStopping(patience=c["patience"])

        train_loader = self._make_loader(X_train, y_train, shuffle=True, batch_size=c["batch_size"])
        val_loader = (
            self._make_loader(X_val, y_val, shuffle=False, batch_size=c["batch_size"])
            if X_val is not None else None
        )

        self.training_history = []
        for epoch in range(total_epochs):
            tr_loss, tr_acc = self._train_epoch(train_loader, optimizer, criterion, label_smooth)
            scheduler.step()
            entry = {"epoch": epoch + 1, "train_loss": tr_loss, "train_acc": tr_acc}

            if val_loader is not None:
                vl_loss, vl_acc = self._eval_epoch(val_loader, criterion)
                entry.update({"val_loss": vl_loss, "val_acc": vl_acc})
                if early_stop(vl_loss, self._net):
                    self.training_history.append(entry)
                    break
            self.training_history.append(entry)

        early_stop.restore_best(self._net)
        return {"epochs_trained": len(self.training_history)}

    def _train_epoch(self, loader, optimizer, criterion, label_smooth):
        self._net.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in loader:
            xt, xo, yb = (
                batch[0].to(self._device),
                batch[1].to(self._device),
                batch[2].to(self._device),
            )
            optimizer.zero_grad()
            logits = self._forward(self._net, xt, xo)
            y_smooth = yb.float() * (1 - label_smooth) + 0.5 * label_smooth
            loss = criterion(logits.squeeze(-1), y_smooth)
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
                xt, xo, yb = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                    batch[2].to(self._device),
                )
                logits = self._forward(self._net, xt, xo)
                loss = criterion(logits.squeeze(-1), yb.float())
                total_loss += loss.item() * len(yb)
                preds = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long()
                correct += (preds == yb).sum().item()
                total += len(yb)
        return total_loss / total, correct / total

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)

    def predict_proba(self, X):
        self._net.eval()
        loader = self._make_loader(X, None, shuffle=False, batch_size=self._cfg["batch_size"])
        probs = []
        with torch.no_grad():
            for batch in loader:
                xt, xo = batch[0].to(self._device), batch[1].to(self._device)
                logits = self._forward(self._net, xt, xo)
                probs.append(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
        p = np.concatenate(probs)
        return np.column_stack([1 - p, p])

    def _make_loader(self, X, y, shuffle, batch_size):
        xt = torch.tensor(X[0], dtype=torch.float32)
        xo = torch.tensor(X[1], dtype=torch.float32)
        if y is not None:
            yt = torch.tensor(y, dtype=torch.long)
            ds = TensorDataset(xt, xo, yt)
        else:
            ds = TensorDataset(xt, xo)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self._net.state_dict()}, path)

    def load(self, path: Path):
        ckpt = torch.load(path, map_location=self._device)
        self._net = self._build_net()
        self._net.load_state_dict(ckpt["state_dict"])
        self._net.eval()


# ---------------------------------------------------------------------------
# Variant 1: MLP Simple — single encoder on concatenated features
# ---------------------------------------------------------------------------

class _MLPSimpleNet(nn.Module):
    """Single MLP on all features concatenated. No dual encoder, no attention."""

    def __init__(self, n_features: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.encoder = MLPEncoderV2(n_features, hidden_dim, n_layers, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_tech, x_onchain):
        # Concatenate both feature sets → single vector
        x = torch.cat([x_tech, x_onchain], dim=-1)
        h = self.encoder(x)
        return self.classifier(h)


class MLPSimpleModel(_MLPTrainerMixin, BaseModel):
    """
    Ablation baseline: single MLP encoder on all features concatenated.
    No dual-domain separation, no cross-attention.
    """

    def __init__(self, cfg: dict, n_technical: int, n_onchain: int,
                 device, random_seed: int = 42):
        BaseModel.__init__(self, name="MLPSimple", random_seed=random_seed)
        self._cfg = cfg
        self._n_technical = n_technical
        self._n_onchain = n_onchain
        self._device = device
        self._net = None
        self.training_history = []

    def _build_net(self):
        c = self._cfg
        return _MLPSimpleNet(
            n_features=self._n_technical + self._n_onchain,
            hidden_dim=c["hidden_dim"],
            n_layers=c["n_encoder_layers"],
            dropout=c["dropout"],
        ).to(self._device)

    def _forward(self, net, xt, xo):
        return net(xt, xo)


# ---------------------------------------------------------------------------
# Variant 2: MLP Dual Encoder (no cross-attention) — two branches, concat fusion
# ---------------------------------------------------------------------------

class _MLPDualNoAttnNet(nn.Module):
    """
    Two independent MLP encoders (tech + onchain), outputs concatenated.
    No cross-attention between domains.
    """

    def __init__(self, n_technical: int, n_onchain: int,
                 hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.tech_encoder = MLPEncoderV2(n_technical, hidden_dim, n_layers, dropout)
        self.onchain_encoder = MLPEncoderV2(n_onchain, hidden_dim, n_layers, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_tech, x_onchain):
        h_tech = self.tech_encoder(x_tech)
        h_onchain = self.onchain_encoder(x_onchain)
        fused = torch.cat([h_tech, h_onchain], dim=-1)
        return self.classifier(fused)


class MLPDualNoAttnModel(_MLPTrainerMixin, BaseModel):
    """
    Ablation variant: dual MLP encoders with concatenation fusion.
    No cross-attention — isolates the value of the attention mechanism.
    """

    def __init__(self, cfg: dict, n_technical: int, n_onchain: int,
                 device, random_seed: int = 42):
        BaseModel.__init__(self, name="MLPDualNoAttn", random_seed=random_seed)
        self._cfg = cfg
        self._n_technical = n_technical
        self._n_onchain = n_onchain
        self._device = device
        self._net = None
        self.training_history = []

    def _build_net(self):
        c = self._cfg
        return _MLPDualNoAttnNet(
            n_technical=self._n_technical,
            n_onchain=self._n_onchain,
            hidden_dim=c["hidden_dim"],
            n_layers=c["n_encoder_layers"],
            dropout=c["dropout"],
        ).to(self._device)

    def _forward(self, net, xt, xo):
        return net(xt, xo)
