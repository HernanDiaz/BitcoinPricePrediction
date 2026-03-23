"""
MLP Dual Encoder with Feature Cross-Attention — revised novel architecture.

Motivation:
  The sequential CNN-LSTM approach failed because the predictive signal in
  on-chain metrics (MVRV, SOPR, Supply in Profit, etc.) resides in their
  CURRENT VALUE, not in 30-day temporal trajectories. This is confirmed by
  XGBoost outperforming CNN-LSTM by >20 accuracy points.

  This model processes current-day flat features (no sequences) through
  independent MLP encoders for technical and on-chain branches, then applies
  bidirectional cross-attention at the representation level. The attention
  weights indicate how much each domain's encoding modulates the other.

Architecture:
  x_tech  (n_tech)  --> MLP encoder --> h_tech  (H) -┐
                                                       ├--> cross-attn --> concat --> MLP --> logit
  x_onchain (n_on)  --> MLP encoder --> h_onchain(H) -┘

  Cross-attention expands each representation to K learned "query tokens"
  before attending, giving richer interaction than single-vector attention.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseModel
from .torch_utils import EarlyStopping, get_device, set_all_seeds


class MLPEncoder(nn.Module):
    """Simple MLP encoder: projects flat features to a hidden representation."""

    def __init__(self, n_features: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, H)


class MLPDualEncoderNet(nn.Module):
    """
    Core module: two MLP branches + bidirectional cross-attention.
    The cross-attention treats the two branch representations as single
    tokens. A learnable 'context' matrix expands each token into K vectors
    before attention, allowing richer feature interaction.
    """

    def __init__(
        self,
        n_technical: int,
        n_onchain: int,
        hidden_dim: int = 128,
        n_context_tokens: int = 4,
        n_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_context_tokens = n_context_tokens

        # Independent encoders
        self.tech_encoder = MLPEncoder(n_technical, hidden_dim, dropout)
        self.onchain_encoder = MLPEncoder(n_onchain, hidden_dim, dropout)

        # Learnable context projections: expand single representation to K tokens
        self.tech_context = nn.Linear(hidden_dim, n_context_tokens * hidden_dim)
        self.onchain_context = nn.Linear(hidden_dim, n_context_tokens * hidden_dim)

        # Cross-attention: tech tokens attend to onchain tokens
        self.cross_attn_tech = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        # Cross-attention: onchain tokens attend to tech tokens
        self.cross_attn_onchain = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )

        self.norm_tech = nn.LayerNorm(hidden_dim)
        self.norm_onchain = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1),
        )

    def forward(
        self, x_tech: torch.Tensor, x_onchain: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x_tech.size(0)
        K = self.n_context_tokens
        H = self.hidden_dim

        # Encode
        h_tech = self.tech_encoder(x_tech)         # (B, H)
        h_onchain = self.onchain_encoder(x_onchain) # (B, H)

        # Expand to K context tokens each
        tech_ctx = self.tech_context(h_tech).view(B, K, H)       # (B, K, H)
        onchain_ctx = self.onchain_context(h_onchain).view(B, K, H)  # (B, K, H)

        # Bidirectional cross-attention with residual
        tech_attended, tech_weights = self.cross_attn_tech(
            query=tech_ctx, key=onchain_ctx, value=onchain_ctx
        )
        tech_out = self.norm_tech(tech_ctx + tech_attended)   # (B, K, H)

        onchain_attended, onchain_weights = self.cross_attn_onchain(
            query=onchain_ctx, key=tech_ctx, value=tech_ctx
        )
        onchain_out = self.norm_onchain(onchain_ctx + onchain_attended)  # (B, K, H)

        # Mean pool over context tokens
        tech_pooled = tech_out.mean(dim=1)        # (B, H)
        onchain_pooled = onchain_out.mean(dim=1)  # (B, H)

        fused = torch.cat([tech_pooled, onchain_pooled], dim=-1)  # (B, 2H)
        logits = self.classifier(self.dropout(fused))              # (B, 1)

        return logits, tech_weights, onchain_weights


class MLPDualEncoderModel(BaseModel):
    """
    Flat (non-sequential) dual encoder. Accepts current-day feature vectors.
    X_train and X_val must be tuples: (X_tech, X_onchain)
    each of shape (n_samples, n_features_branch).
    """

    def __init__(
        self,
        config: dict,
        n_technical: int,
        n_onchain: int,
        random_seed: int = 42,
    ):
        super().__init__(name="MLPDualEncoder", random_seed=random_seed)
        cfg = config["models"]["mlp_dual_encoder"]
        self._cfg = cfg
        self._n_technical = n_technical
        self._n_onchain = n_onchain
        self._device = get_device(config)
        self._net: MLPDualEncoderNet | None = None
        self.training_history: list[dict] = []
        self._last_tech_weights: np.ndarray | None = None
        self._last_onchain_weights: np.ndarray | None = None

    def _build_net(self) -> MLPDualEncoderNet:
        cfg = self._cfg
        return MLPDualEncoderNet(
            n_technical=self._n_technical,
            n_onchain=self._n_onchain,
            hidden_dim=cfg["hidden_dim"],
            n_context_tokens=cfg["n_context_tokens"],
            n_heads=cfg["n_heads"],
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
        set_all_seeds(self.random_seed)
        cfg = self._cfg

        self._net = self._build_net()
        optimizer = torch.optim.AdamW(
            self._net.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=cfg["lr_min"]
        )
        pos_weight = torch.tensor([class_weight], device=self._device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        early_stop = EarlyStopping(patience=cfg["early_stopping_patience"])

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False) if X_val is not None else None

        self.training_history = []
        for epoch in range(cfg["epochs"]):
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, criterion)
            scheduler.step()
            entry = {"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc}

            if val_loader is not None:
                val_loss, val_acc = self._eval_epoch(val_loader, criterion)
                entry.update({"val_loss": val_loss, "val_acc": val_acc})
                if early_stop(val_loss, self._net):
                    entry["early_stop"] = True
                    self.training_history.append(entry)
                    break

            self.training_history.append(entry)

        early_stop.restore_best(self._net)
        return {"model": "MLPDualEncoder", "epochs_trained": len(self.training_history)}

    def predict(self, X: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(np.int64)

    def predict_proba(self, X: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        self._net.eval()
        loader = self._make_loader(X, None, shuffle=False)
        all_probs, all_tw, all_ow = [], [], []
        with torch.no_grad():
            for batch in loader:
                xt, xo = batch[0].to(self._device), batch[1].to(self._device)
                logits, tw, ow = self._net(xt, xo)
                probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                all_probs.append(probs)
                all_tw.append(tw.cpu().numpy())
                all_ow.append(ow.cpu().numpy())
        self._last_tech_weights = np.concatenate(all_tw, axis=0)
        self._last_onchain_weights = np.concatenate(all_ow, axis=0)
        probs = np.concatenate(all_probs)
        return np.column_stack([1 - probs, probs])

    def get_attention_weights(self) -> tuple[np.ndarray, np.ndarray]:
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
            nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=2.0)
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
        if X is None:
            return None
        xt = torch.tensor(X[0], dtype=torch.float32)
        xo = torch.tensor(X[1], dtype=torch.float32)
        if y is not None:
            yt = torch.tensor(y, dtype=torch.long)
            dataset = TensorDataset(xt, xo, yt)
        else:
            dataset = TensorDataset(xt, xo)
        return DataLoader(dataset, batch_size=self._cfg["batch_size"], shuffle=shuffle, pin_memory=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self._net.state_dict(),
            "n_technical": self._n_technical,
            "n_onchain": self._n_onchain,
        }, path)

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self._device)
        self._n_technical = ckpt["n_technical"]
        self._n_onchain = ckpt["n_onchain"]
        self._net = self._build_net()
        self._net.load_state_dict(ckpt["state_dict"])
        self._net.eval()
