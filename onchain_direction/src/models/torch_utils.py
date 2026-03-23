"""
Shared PyTorch utilities: seed management, device selection, early stopping.
"""

import copy
import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Set seeds for full reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config: dict) -> torch.device:
    requested = config.get("project", {}).get("device", "cuda")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_weighted_sampler(y: np.ndarray):
    """
    WeightedRandomSampler for imbalanced datasets.
    Each class is sampled with equal frequency.
    """
    from torch.utils.data import WeightedRandomSampler
    class_counts = np.bincount(y)
    weights = 1.0 / class_counts[y]
    weights_tensor = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights_tensor, num_samples=len(y), replacement=True)


class EarlyStopping:
    """
    Monitors validation loss. Saves the best model weights.
    Stops training when val_loss has not improved for `patience` epochs.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self._best_loss = float("inf")
        self._counter = 0
        self._best_weights: dict | None = None
        self.stopped = False

    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self._best_loss - self.min_delta:
            self._best_loss = val_loss
            self._best_weights = copy.deepcopy(model.state_dict())
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self.stopped = True
                return True
        return False

    def restore_best(self, model: torch.nn.Module) -> None:
        """Load the best saved weights back into the model."""
        if self._best_weights is not None:
            model.load_state_dict(self._best_weights)
