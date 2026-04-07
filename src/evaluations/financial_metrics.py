"""Advanced financial evaluation metrics for RL trading agents.

Rigorous implementations of:
  - Sortino Ratio             (asymmetric, downside-only volatility penalty)
  - Value at Risk (VaR)       (quantile of the empirical loss distribution)
  - Conditional VaR / CVaR   (Expected Shortfall — coherent risk measure)
  - Maximum Drawdown (MDD)   (peak-to-trough equity decline)
  - Calmar Ratio              (return per unit of maximum drawdown)
  - Profit Factor             (gross profit / gross loss)
  - Hit Ratio                 (win rate — fraction of positive-reward steps)

All functions operate on *step-wise reward arrays* (additive PnL framework).
For batch evaluation, use ``compute_batch_advanced_metrics``.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. Sortino Ratio
# ---------------------------------------------------------------------------

def compute_sortino_ratio(rewards: np.ndarray, tau: float = 0.0) -> float:
    """Sortino ratio with minimum acceptable return ``tau``.

    sigma_d = sqrt( E[ min(0, R_t - tau)^2 ] )   (lower partial moment, order 2)
    Sortino  = ( E[R] - tau ) / sigma_d

    Only downside deviations below ``tau`` are penalised; upward volatility
    (favourable to the agent) does not inflate the denominator.

    Parameters
    ----------
    rewards : 1-D array of step-wise returns.
    tau     : Minimum acceptable return.  Use 0.0 for LOB/HFT contexts.
    """
    downside = np.minimum(rewards - tau, 0.0)
    sigma_d = float(np.sqrt(np.mean(downside ** 2)))
    return float((np.mean(rewards) - tau) / (sigma_d + 1e-8))


# ---------------------------------------------------------------------------
# 2. Value at Risk and Conditional VaR (Expected Shortfall)
# ---------------------------------------------------------------------------

def compute_var(rewards: np.ndarray, alpha: float = 0.95) -> float:
    """Empirical VaR at confidence level ``alpha``.

    VaR_alpha = inf { l in R : F_L(l) >= alpha }  where L = -R.

    Parameters
    ----------
    rewards : 1-D array of step-wise returns.
    alpha   : Confidence level (e.g. 0.95 or 0.99).
    """
    return float(np.quantile(-rewards, alpha))


def compute_cvar(rewards: np.ndarray, alpha: float = 0.95) -> float:
    """Empirical CVaR (Expected Shortfall) at confidence level ``alpha``.

    CVaR_alpha = E[ L | L >= VaR_alpha ]

    CVaR is a *coherent* risk measure (Artzner et al., 1999): it satisfies
    translation invariance, subadditivity, positive homogeneity and monotonicity.
    Unlike VaR it correctly captures the severity of tail losses.

    Parameters
    ----------
    rewards : 1-D array of step-wise returns.
    alpha   : Confidence level (e.g. 0.95 or 0.99).
    """
    losses = -rewards
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    return float(tail.mean()) if len(tail) > 0 else float(var)


def compute_var_cvar(
    rewards: np.ndarray, alpha: float = 0.95
) -> tuple[float, float]:
    """Return ``(VaR_alpha, CVaR_alpha)`` together to avoid recomputing quantiles."""
    losses = -rewards
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if len(tail) > 0 else var
    return var, cvar


# ---------------------------------------------------------------------------
# 3. Maximum Drawdown and Calmar Ratio
# ---------------------------------------------------------------------------

def compute_max_drawdown(rewards: np.ndarray) -> float:
    """Absolute maximum drawdown of the cumulative PnL curve.

    D_t   = M_t - P_t   where M_t = max_{tau <= t} P_tau
    MDD_T = max_{t in [0,T]} D_t

    Path-dependent: any permutation of the same rewards may yield a different MDD.

    Parameters
    ----------
    rewards : 1-D array of step-wise returns (additive PnL).
    """
    cum_pnl = np.cumsum(rewards)
    running_max = np.maximum.accumulate(cum_pnl)
    return float((running_max - cum_pnl).max())


def compute_calmar_ratio(rewards: np.ndarray) -> float:
    """Calmar ratio = total_return / MDD.

    Uses raw (non-annualised) total return because all episodes share the same
    fixed horizon; cross-agent comparisons remain valid.

    Parameters
    ----------
    rewards : 1-D array of step-wise returns.
    """
    mdd = compute_max_drawdown(rewards)
    return float(rewards.sum() / (mdd + 1e-8))


# ---------------------------------------------------------------------------
# 4. Profit Factor and Hit Ratio
# ---------------------------------------------------------------------------

def compute_hit_ratio(rewards: np.ndarray) -> float:
    """Win rate — fraction of time-steps with strictly positive reward.

    A Hit Ratio < 0.5 can still be profitable if the asymmetry of payoffs
    compensates (verified by Profit Factor > 1).

    Parameters
    ----------
    rewards : 1-D array of step-wise returns.
    """
    return float((rewards > 0).mean())


def compute_profit_factor(rewards: np.ndarray) -> float:
    """Profit Factor = gross_profit / |gross_loss|.

    > 1  ⟺  strategy has positive expected value.
    < 1  ⟺  strategy is net negative (losing more on losses than winning on gains).

    Parameters
    ----------
    rewards : 1-D array of step-wise returns.
    """
    gross_profit = float(rewards[rewards > 0].sum())
    gross_loss = float(np.abs(rewards[rewards < 0].sum()))
    return gross_profit / (gross_loss + 1e-8)


# ---------------------------------------------------------------------------
# 5. Aggregate helpers
# ---------------------------------------------------------------------------

def compute_advanced_metrics(rewards: np.ndarray) -> dict[str, float]:
    """Compute all advanced metrics for a single trajectory.

    Parameters
    ----------
    rewards : 1-D numpy float array of step-wise returns.

    Returns
    -------
    Flat dict of scalar floats with keys:
      Sortino, VaR_95, CVaR_95, VaR_99, CVaR_99,
      MaxDD, Calmar, ProfitFactor, HitRatio.
    """
    var95, cvar95 = compute_var_cvar(rewards, alpha=0.95)
    var99, cvar99 = compute_var_cvar(rewards, alpha=0.99)
    return {
        "Sortino":      compute_sortino_ratio(rewards),
        "VaR_95":       var95,
        "CVaR_95":      cvar95,
        "VaR_99":       var99,
        "CVaR_99":      cvar99,
        "MaxDD":        compute_max_drawdown(rewards),
        "Calmar":       compute_calmar_ratio(rewards),
        "ProfitFactor": compute_profit_factor(rewards),
        "HitRatio":     compute_hit_ratio(rewards),
    }


def compute_batch_advanced_metrics(rewards_input) -> dict[str, float]:
    """Batch-averaged advanced metrics from a (B, T) rewards array or tensor.

    Parameters
    ----------
    rewards_input : shape (B, T) — PyTorch tensor or NumPy array.

    Returns
    -------
    Dict of batch-mean scalar floats.
    """
    try:
        import torch
        if isinstance(rewards_input, torch.Tensor):
            rewards_np = rewards_input.detach().cpu().numpy().astype(np.float64)
        else:
            rewards_np = np.asarray(rewards_input, dtype=np.float64)
    except ImportError:
        rewards_np = np.asarray(rewards_input, dtype=np.float64)

    B = rewards_np.shape[0]
    all_m = [compute_advanced_metrics(rewards_np[i]) for i in range(B)]
    return {
        key: float(np.mean([m[key] for m in all_m]))
        for key in all_m[0]
    }
