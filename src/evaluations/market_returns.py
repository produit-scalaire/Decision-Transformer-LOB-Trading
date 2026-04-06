"""Market return series for evaluation rollouts (no plotting dependencies)."""

from __future__ import annotations

import torch


def get_market_returns(
    states_batch: torch.Tensor,
    state_representation: str = "raw",
) -> torch.Tensor:
    """
    Per-timestep scalar series used with ``position * returns`` in rollout evaluations.

    For ``raw`` states, best bid/ask columns are treated as (possibly normalized) price
    levels and returns are mid-price first differences across time.

    For ``log_returns`` / ``bps`` states (see ``LOBTradingEnv.state_representation``),
    those columns already encode per-tick ask/bid transforms; we use their mean as a
    mid-proxy increment at each step (last step zeroed for parity with the raw path).
    """
    ask1 = states_batch[:, :, 0]
    bid1 = states_batch[:, :, 2]

    if state_representation == "raw":
        mid_price = (ask1 + bid1) / 2.0
        returns = torch.zeros_like(mid_price)
        returns[:, :-1] = mid_price[:, 1:] - mid_price[:, :-1]
        return returns

    if state_representation in ("log_returns", "bps"):
        r = (ask1 + bid1) / 2.0
        returns = r.clone()
        returns[:, -1] = 0.0
        return returns

    raise ValueError(
        f"Unknown state_representation {state_representation!r}; "
        "expected 'raw', 'log_returns', or 'bps'."
    )
