"""Directional classification metrics for evaluation (no plotting dependencies)."""

from __future__ import annotations

import numpy as np
import torch


def oracle_actions_from_returns(market_returns: torch.Tensor) -> torch.Tensor:
    """
    Greedy discrete oracle aligned with ``evaluate_baselines`` oracle positions:
    long (+1) if return > 0, short (-1) if return < 0, else flat.
    Maps to DT action indices: 0=short, 1=hold, 2=long (position = action - 1).
    """
    pos = torch.where(
        market_returns > 0,
        torch.ones_like(market_returns),
        torch.where(market_returns < 0, -torch.ones_like(market_returns), torch.zeros_like(market_returns)),
    )
    return (pos + 1).long()


def _macro_f1_multiclass(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> float:
    """Unweighted macro F1 over class indices ``0 .. num_classes-1`` (NumPy only)."""
    f1s: list[float] = []
    for c in range(num_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(sum(f1s) / num_classes)


def compute_directional_f1(
    pred_actions: torch.Tensor,
    market_returns: torch.Tensor,
    average: str = "macro",
) -> float:
    """
    Macro F1 between predicted actions and oracle_actions_from_returns (classes 0,1,2).
    Comparable in spirit to directional / movement classification metrics in LOB literature,
    though labels come from the rollout's mid-proxy returns, not FI-2010 smoothing.
    """
    if average != "macro":
        raise ValueError(f"Only average='macro' is supported, got {average!r}")
    y_true = oracle_actions_from_returns(market_returns).detach().cpu().numpy().ravel()
    y_pred = pred_actions.detach().cpu().numpy().ravel()
    return _macro_f1_multiclass(y_true, y_pred, num_classes=3)
