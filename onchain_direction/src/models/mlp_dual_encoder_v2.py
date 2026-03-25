"""
MLP Dual Encoder V2 with Stacked Cross-Attention — improved architecture.

Key improvements over V1:
  1. Deeper MLPEncoder with residual skip connections (gradient flow)
  2. Self-attention within each domain before cross-attention
  3. Stacked cross-attention layers (transformer-style depth)
  4. Gated fusion instead of plain concatenation
  5. Learning rate warmup + cosine annealing
  6. Label smoothing for regularisation
  7. Larger batch size options
  8. All hyperparameters exposed for Optuna search

Architecture:
  x_tech   --> MLPEncoderV2 --> K tokens --> Self-Attn --> [Cross-Attn x L] --> gate-fuse --> MLP --> logit
  x_onchain --> MLPEncoderV2 --> K tokens --> Self-Attn --> [Cross-Attn x L] --^
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .base_model import BaseModel
from .torch_utils import EarlyStopping, get_device, set_all_seeds


class MLPEncoderV2(nn.Module):
    """
    Deep MLP encoder with residual connections.
    Input -> hidden_dim via a stack of (Linear + LN + GELU + Dropout) blocks.
    Residual added every 2 layers.
    """

    def __init__(self, n_features: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        layers = [nn.Linear(n_features, hidden_dim), nn.LayerNorm(hidden_dim),
                  nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                       nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.LayerNorm(hidden_dim)]   # final norm before residual add
        self.net = nn.Sequential(*layers)
        # Residual projection: map input to hidden_dim for skip connection
        self.proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
        ) if n_features != hidden_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x) + self.proj(x)   # residual skip
        return out


class CrossAttentionBlock(nn.Module):
    """
    One layer of bidirectional cross-attention with self-attention pre-norm.
    Processes K-token representations for both domains.
    """

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        # Self-attention within each domain
        self.self_attn_tech    = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.self_attn_onchain = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        # Cross-attention
        self.cross_attn_tech    = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn_onchain = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        # Norms
        self.norm_self_t  = nn.LayerNorm(hidden_dim)
        self.norm_self_o  = nn.LayerNorm(hidden_dim)
        self.norm_cross_t = nn.LayerNorm(hidden_dim)
        self.norm_cross_o = nn.LayerNorm(hidden_dim)
        # FFN after cross-attention
        self.ffn_t = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.ffn_o = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm_ffn_t = nn.LayerNorm(hidden_dim)
        self.norm_ffn_o = nn.LayerNorm(hidden_dim)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, tech_tokens, onchain_tokens):
        # Self-attention (pre-norm)
        t2, _ = self.self_attn_tech(tech_tokens, tech_tokens, tech_tokens)
        tech_tokens = self.norm_self_t(tech_tokens + self.dropout(t2))
        o2, _ = self.self_attn_onchain(onchain_tokens, onchain_tokens, onchain_tokens)
        onchain_tokens = self.norm_self_o(onchain_tokens + self.dropout(o2))

        # Cross-attention
        t3, t_weights = self.cross_attn_tech(tech_tokens, onchain_tokens, onchain_tokens)
        tech_out = self.norm_cross_t(tech_tokens + self.dropout(t3))
        o3, o_weights = self.cross_attn_onchain(onchain_tokens, tech_tokens, tech_tokens)
        onchain_out = self.norm_cross_o(onchain_tokens + self.dropout(o3))

        # FFN
        tech_out    = self.norm_ffn_t(tech_out + self.dropout(self.ffn_t(tech_out)))
        onchain_out = self.norm_ffn_o(onchain_out + self.dropout(self.ffn_o(onchain_out)))

        return tech_out, onchain_out, t_weights, o_weights


class GatedFusion(nn.Module):
    """Gated fusion of two representations: learns how much each contributes."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
        )

    def forward(self, t: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([t, o], dim=-1)
        gate   = self.gate(concat)
        return gate * self.proj(concat)


class MLPDualEncoderNetV2(nn.Module):

    def __init__(
        self,
        n_technical:       int,
        n_onchain:         int,
        hidden_dim:        int   = 256,
        n_context_tokens:  int   = 8,
        n_heads:           int   = 4,
        n_encoder_layers:  int   = 3,
        n_cross_layers:    int   = 2,
        dropout:           float = 0.25,
    ):
        super().__init__()
        self.hidden_dim       = hidden_dim
        self.n_context_tokens = n_context_tokens

        self.tech_encoder    = MLPEncoderV2(n_technical, hidden_dim, n_encoder_layers, dropout)
        self.onchain_encoder = MLPEncoderV2(n_onchain,   hidden_dim, n_encoder_layers, dropout)

        # Expand single rep to K tokens
        self.tech_context    = nn.Linear(hidden_dim, n_context_tokens * hidden_dim)
        self.onchain_context = nn.Linear(hidden_dim, n_context_tokens * hidden_dim)

        # Stack of cross-attention blocks
        self.cross_layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_cross_layers)
        ])

        # Gated fusion
        self.fusion = GatedFusion(hidden_dim)

        # Classifier
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
        B, K, H = x_tech.size(0), self.n_context_tokens, self.hidden_dim

        h_tech    = self.tech_encoder(x_tech)
        h_onchain = self.onchain_encoder(x_onchain)

        tech_tokens    = self.tech_context(h_tech).view(B, K, H)
        onchain_tokens = self.onchain_context(h_onchain).view(B, K, H)

        t_weights_last, o_weights_last = None, None
        for layer in self.cross_layers:
            tech_tokens, onchain_tokens, t_weights_last, o_weights_last = layer(tech_tokens, onchain_tokens)

        tech_pooled    = tech_tokens.mean(dim=1)
        onchain_pooled = onchain_tokens.mean(dim=1)

        fused  = self.fusion(tech_pooled, onchain_pooled)
        logits = self.classifier(fused)

        return logits, t_weights_last, o_weights_last


class MLPDualEncoderModelV2(BaseModel):

    def __init__(self, cfg: dict, n_technical: int, n_onchain: int,
                 device, random_seed: int = 42):
        super().__init__(name="MLPDualEncoderV2", random_seed=random_seed)
        self._cfg        = cfg
        self._n_technical = n_technical
        self._n_onchain   = n_onchain
        self._device      = device
        self._net         = None
        self.training_history = []
        self._last_tech_weights    = None
        self._last_onchain_weights = None

    def _build_net(self) -> MLPDualEncoderNetV2:
        c = self._cfg
        return MLPDualEncoderNetV2(
            n_technical=self._n_technical, n_onchain=self._n_onchain,
            hidden_dim=c["hidden_dim"], n_context_tokens=c["n_context_tokens"],
            n_heads=c["n_heads"], n_encoder_layers=c["n_encoder_layers"],
            n_cross_layers=c["n_cross_layers"], dropout=c["dropout"],
        ).to(self._device)

    def fit(self, X_train, y_train, X_val=None, y_val=None, class_weight=1.0):
        set_all_seeds(self.random_seed)
        c = self._cfg

        self._net = self._build_net()
        optimizer = torch.optim.AdamW(self._net.parameters(),
                                       lr=c["lr"], weight_decay=c["weight_decay"])

        total_epochs = c["epochs"]
        warmup       = c.get("warmup_epochs", 10)

        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(total_epochs - warmup, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        pos_weight = torch.tensor([class_weight], device=self._device)
        label_smooth = c.get("label_smoothing", 0.05)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        early_stop = EarlyStopping(patience=c["patience"])

        train_loader = self._make_loader(X_train, y_train, shuffle=True,
                                          batch_size=c["batch_size"])
        val_loader   = (self._make_loader(X_val, y_val, shuffle=False,
                                           batch_size=c["batch_size"])
                        if X_val is not None else None)

        self.training_history = []
        for epoch in range(total_epochs):
            tr_loss, tr_acc = self._train_epoch(train_loader, optimizer, criterion,
                                                 label_smooth)
            scheduler.step()
            entry = {"epoch": epoch+1, "train_loss": tr_loss, "train_acc": tr_acc}

            if val_loader is not None:
                vl_loss, vl_acc = self._eval_epoch(val_loader, criterion)
                entry.update({"val_loss": vl_loss, "val_acc": vl_acc})
                if early_stop(vl_loss, self._net):
                    self.training_history.append(entry)
                    break
            self.training_history.append(entry)

        early_stop.restore_best(self._net)
        return {"epochs_trained": len(self.training_history)}

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)

    def predict_proba(self, X):
        self._net.eval()
        loader = self._make_loader(X, None, shuffle=False,
                                    batch_size=self._cfg["batch_size"])
        probs, all_tw, all_ow = [], [], []
        with torch.no_grad():
            for batch in loader:
                xt, xo = batch[0].to(self._device), batch[1].to(self._device)
                logits, tw, ow = self._net(xt, xo)
                probs.append(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
                if tw is not None:
                    all_tw.append(tw.cpu().numpy())
                    all_ow.append(ow.cpu().numpy())
        if all_tw:
            self._last_tech_weights    = np.concatenate(all_tw, axis=0)
            self._last_onchain_weights = np.concatenate(all_ow, axis=0)
        p = np.concatenate(probs)
        return np.column_stack([1 - p, p])

    def get_attention_weights(self):
        return self._last_tech_weights, self._last_onchain_weights

    def _train_epoch(self, loader, optimizer, criterion, label_smooth=0.0):
        self._net.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in loader:
            xt, xo, yb = (batch[0].to(self._device), batch[1].to(self._device),
                          batch[2].to(self._device))
            optimizer.zero_grad()
            logits, _, _ = self._net(xt, xo)
            # Label smoothing
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
                xt, xo, yb = (batch[0].to(self._device), batch[1].to(self._device),
                               batch[2].to(self._device))
                logits, _, _ = self._net(xt, xo)
                loss = criterion(logits.squeeze(-1), yb.float())
                total_loss += loss.item() * len(yb)
                preds = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long()
                correct += (preds == yb).sum().item()
                total += len(yb)
        return total_loss / total, correct / total

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
        torch.save({"state_dict": self._net.state_dict(),
                    "n_technical": self._n_technical,
                    "n_onchain": self._n_onchain}, path)

    def load(self, path: Path):
        ckpt = torch.load(path, map_location=self._device)
        self._n_technical = ckpt["n_technical"]
        self._n_onchain   = ckpt["n_onchain"]
        self._net = self._build_net()
        self._net.load_state_dict(ckpt["state_dict"])
        self._net.eval()
