"""Market return series for evaluation rollouts (no plotting dependencies)."""

from __future__ import annotations

import torch


def get_market_returns(
    states_batch: torch.Tensor,
    state_representation: str = "raw",
) -> torch.Tensor:
    """
    Per-timestep scalar series used with ``position * returns`` in rollout evaluations.

    For ``raw`` states, returns are aligned to the environment reward indexing used
    when trajectories were generated:
      r_t = mid[s+1+t] - mid[s+t],
    while flattened ``state_t`` stores the last observed snapshot at index s-1+t.
    This implies ``returns[:, t] = mid_price[:, t+2] - mid_price[:, t+1]`` inside
    the flattened state tensor.

    For ``log_returns`` / ``bps`` states (see ``LOBTradingEnv.state_representation``),
    the transformed channels are used as a proxy increment sequence.
    """
    ask1 = states_batch[:, :, 0]
    bid1 = states_batch[:, :, 2]

    if state_representation == "raw":
        mid_price = (ask1 + bid1) / 2.0
        returns = torch.zeros_like(mid_price)
        if mid_price.size(1) >= 3:
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
