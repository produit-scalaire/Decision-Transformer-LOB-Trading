from __future__ import annotations

import os
import time
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.evaluations.direction_metrics import compute_directional_f1
from src.evaluations.financial_metrics import compute_batch_advanced_metrics
from src.models.model_factory import build_model
from src.evaluations.market_returns import get_market_returns
from src.env.lob_trading_env import LOBTradingEnv
from src.data.trajectories_generator import detect_stock_boundaries

plt.ioff()


def _as_float_list(x: Any) -> list[float]:
    if x is None:
        return []
    return [float(v) for v in list(x)]


def _load_train_initial_rtg(train_data_path: str) -> np.ndarray:
    """Distribution of R̂_0 per trajectory — matches the RTG token at the first step of training.

    For full-episode RTG horizon this equals ``total_return``; for bounded horizons it is the
    sum of only the first H rewards (see ``compute_rtg`` in the trajectory generator).
    """
    trajs = torch.load(train_data_path, map_location="cpu", weights_only=False)
    if isinstance(trajs, dict):
        trajs = list(trajs.values())
    out: list[float] = []
    for t in trajs:
        rtg = t["rtg"]
        if isinstance(rtg, torch.Tensor):
            out.append(float(rtg[0, 0].item()))
        else:
            out.append(float(rtg[0, 0]))
    return np.asarray(out, dtype=np.float64)


def resolve_target_rtgs(eval_cfg: Any, train_data_path: str | None) -> list[float]:
    """Choose evaluation RTG targets: fixed list or percentiles of train ``total_return``."""
    source = getattr(eval_cfg, "rtg_source", "manual")
    if source == "train_percentiles":
        if not train_data_path or not Path(train_data_path).is_file():
            raise FileNotFoundError(
                f"rtg_source=train_percentiles requires an existing train_data file; "
                f"got {train_data_path!r}"
            )
        pct = _as_float_list(getattr(eval_cfg, "train_rtg_percentiles", [10, 25, 50, 75, 90]))
        returns = _load_train_initial_rtg(train_data_path)
        return [float(np.percentile(returns, p)) for p in pct]
    return _as_float_list(getattr(eval_cfg, "target_rtgs", []))


def precompute_chronological_lob_tails_and_returns(
    lob_data: np.ndarray,
    window_size: int,
    state_representation: str,
    price_offset: float,
    reward_type: str = "mid_price",
    reward_shaping: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One full chronological pass over ``lob_data`` (e.g. FI-2010 Day 10).

    Returns LOB tail features (last row of each transformed window) and aligned
    mid-price step returns, matching ``LOBTradingEnv`` indexing for a full episode
    (``episode_length=None``).
    """
    rs = reward_shaping or {}
    env = LOBTradingEnv(
        lob_data.astype(np.float32),
        window_size=window_size,
        transaction_cost=0.0,
        episode_length=None,
        reward_type=reward_type,
        drawdown_coef=float(rs.get("drawdown_coef", 0.0)),
        variance_coef=float(rs.get("variance_coef", 0.0)),
        time_in_market_coef=float(rs.get("time_in_market_coef", 0.0)),
        variance_window=int(rs.get("variance_window", 20)),
        state_representation=state_representation,
        price_offset=float(price_offset),
    )
    n = env.n_timesteps
    # Steps where the agent acts: _current_step = W .. n-2 (inclusive)
    t_eff = n - 1 - window_size
    if t_eff <= 0:
        raise ValueError(
            f"LOB too short for window_size={window_size}: n_timesteps={n}"
        )
    tails = np.empty((t_eff, env.n_features), dtype=np.float32)
    mid = env.mid_prices
    returns_1d = np.empty(t_eff, dtype=np.float32)
    for k in range(t_eff):
        i = window_size + k
        start = i - window_size
        lob_window = env.lob_data[start:i]
        tw = env._transform_lob_window(start, lob_window)
        tails[k] = tw[-1]
        returns_1d[k] = float(mid[i + 1] - mid[i])
    return tails, returns_1d


def _load_fi2010_test_lob_matrix() -> np.ndarray:
    """Load raw LOB matrix for FI-2010 fold 9 test file (same layout as trajectory generator)."""
    import kagglehub

    dataset_path = Path(kagglehub.dataset_download("ulfricirons/fi-2010"))
    test_file = next(dataset_path.rglob("*NoAuction_Zscore*Testing/Test*CF_9.txt"))
    test_raw = np.loadtxt(test_file)
    return test_raw[:40, :].T.astype(np.float32)


# =============================================================================
# 1. Financial Evaluation Metrics
# =============================================================================

def compute_financial_metrics(rewards_tensor: torch.Tensor) -> dict:
    """Compute all financial metrics for a batch of reward trajectories.

    Parameters
    ----------
    rewards_tensor : shape (B, T) — step-wise rewards.

    Returns
    -------
    Dict with keys: PnL, Sharpe, and all advanced metrics from
    ``compute_batch_advanced_metrics`` (Sortino, VaR_95/99, CVaR_95/99,
    MaxDD, Calmar, ProfitFactor, HitRatio).
    """
    cum_pnl = torch.cumsum(rewards_tensor, dim=1)
    final_pnl = cum_pnl[:, -1].mean().item()

    mean_return = rewards_tensor.mean(dim=1)
    std_return = rewards_tensor.std(dim=1) + 1e-8
    sharpe = (mean_return / std_return).mean().item()

    advanced = compute_batch_advanced_metrics(rewards_tensor)

    return {"PnL": final_pnl, "Sharpe": sharpe, **advanced}


# =============================================================================
# 2. Autoregressive Rollout & Baselines
# =============================================================================

def vectorized_autoregressive_rollout(
    model,
    states,
    market_returns,
    target_rtg,
    context_len,
    device,
    max_timestep: int,
    *,
    rtg_rollout_mode: str = "autoregressive",
    reference_rtg: torch.Tensor | None = None,
):
    """Step-by-step autoregressive rollout for the Decision Transformer.

    Uses dynamically growing history (Algorithm 1, Chen et al. 2021) with
    left-padding to ``context_len`` so the model always receives the same
    sequence length it was trained on.  Padding uses neutral values (Flat
    action, zero state/RTG, timestep 0) rather than the previous approach
    of filling a fixed buffer with Short (action 0).

    The last state dimension is treated as **position** (as in training). It is
    overwritten with the **running position** from the DT rollout so conditioning
    matches inference semantics rather than the behaviour-policy position stored
    in offline trajectories.

    ``max_timestep`` clamps absolute time indices so they stay inside the model's
    timestep embedding table (needed for long chronological evaluations).

    Parameters
    ----------
    rtg_rollout_mode
        ``autoregressive`` — RTG token at step ``t`` is ``target_rtg`` minus the
        sum of **predicted** step rewards so far (deployment-style). If the policy
        goes flat, rewards are zero and RTG stops updating, which often locks the
        model in an all-Flat action basin even when teacher-forced accuracy is good.

        ``anchored_offline`` — RTG tokens follow the **same decay shape** as the
        offline trajectory, with the initial level pinned to ``target_rtg``:

        * **Proportional:** when ``|ref[b,0]| > ε``, use
          ``R_{b,t} = reference_rtg[b,t] * (target_rtg / reference_rtg[b,0])`` so the
          offline RTG **curve** is rescaled to hit ``target_rtg`` at ``t=0`` and still
          decay toward zero like ``reference_rtg`` (avoids additive-offset blow-up at
          late ``t``).

        * **Additive fallback** only when ``|ref[b,0]| ≤ ε``:
          ``target_rtg + reference_rtg[b,t] - reference_rtg[b,0]``.

    reference_rtg
        Required when ``rtg_rollout_mode == "anchored_offline"``: float tensor of
        shape ``(B, T)`` (same ``B, T`` as ``states``), typically
        ``traj["rtg"][:, 0]`` stacked over the batch.

    Returns
    -------
    realized_rewards   : (B, T)
    predicted_positions: (B, T)
    predicted_actions  : (B, T)  — integer action indices 0/1/2
    """
    B, T, feature_dim = states.shape
    if feature_dim < 2:
        raise ValueError("state vector must include at least one LOB feature and position")

    if rtg_rollout_mode not in ("autoregressive", "anchored_offline"):
        raise ValueError(
            f"rtg_rollout_mode must be 'autoregressive' or 'anchored_offline'; "
            f"got {rtg_rollout_mode!r}"
        )
    if rtg_rollout_mode == "anchored_offline":
        if reference_rtg is None:
            raise ValueError("anchored_offline rollout requires reference_rtg (B, T)")
        if reference_rtg.shape != (B, T):
            raise ValueError(
                f"reference_rtg must have shape ({B}, {T}); got {tuple(reference_rtg.shape)}"
            )
        reference_rtg = reference_rtg.to(device=device, dtype=torch.float32)

    conditioned_rtg_table: torch.Tensor | None = None
    if rtg_rollout_mode == "anchored_offline":
        ref0 = reference_rtg[:, 0]
        tg = torch.as_tensor(float(target_rtg), device=device, dtype=reference_rtg.dtype)
        eps = 1e-5
        safe = ref0.abs() > eps
        scale = tg / torch.where(safe, ref0, torch.ones_like(ref0))
        scaled_all = reference_rtg * scale.unsqueeze(1)
        add_all = tg + reference_rtg - ref0.unsqueeze(1)
        conditioned_rtg_table = torch.where(safe.unsqueeze(1), scaled_all, add_all)

    predicted_positions = torch.zeros((B, T), device=device)
    predicted_actions = torch.zeros((B, T), dtype=torch.long, device=device)
    realized_rewards = torch.zeros((B, T), device=device)

    current_rtg = torch.full((B, 1), target_rtg, device=device)
    current_pos = torch.zeros(B, device=device)

    t_cap = max(int(max_timestep) - 1, 0)

    # Dynamic history — grows from 1 to context_len, then stays at K.
    # No fake Short-action history; padding uses neutral Flat (action 1).
    hist_states: list[torch.Tensor] = []
    hist_actions: list[torch.Tensor] = []
    hist_rtg: list[torch.Tensor] = []
    hist_timesteps: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for t in range(T):
            state_row = states[:, t, :].clone()
            state_row[:, -1] = current_pos

            hist_states.append(state_row)
            if rtg_rollout_mode == "anchored_offline":
                hist_rtg.append(conditioned_rtg_table[:, t].clone())
            else:
                hist_rtg.append(current_rtg.squeeze(-1).clone())
            hist_timesteps.append(
                torch.full((B,), min(t, t_cap), dtype=torch.long, device=device)
            )
            # Placeholder for current action — causal mask prevents the state
            # token at position 3t+1 from seeing the action token at 3t+2, so
            # this value is invisible to the current prediction.
            hist_actions.append(torch.zeros(B, dtype=torch.long, device=device))

            # Stack the last K real entries
            L = min(len(hist_states), context_len)
            ctx_states = torch.stack(hist_states[-L:], dim=1)
            ctx_actions = torch.stack(hist_actions[-L:], dim=1)
            ctx_rtg = torch.stack(hist_rtg[-L:], dim=1).unsqueeze(-1)
            ctx_timesteps = torch.stack(hist_timesteps[-L:], dim=1)

            # Left-pad to context_len so the model always sees the same
            # sequence length as training.  Padding values: zero state,
            # Flat action (1), zero RTG, timestep 0.
            if L < context_len:
                pad = context_len - L
                ctx_states = F.pad(ctx_states, (0, 0, pad, 0))
                ctx_actions = F.pad(ctx_actions, (pad, 0), value=1)
                ctx_rtg = F.pad(ctx_rtg, (0, 0, pad, 0))
                ctx_timesteps = F.pad(ctx_timesteps, (pad, 0))

            action_preds = model(ctx_states, ctx_actions, ctx_rtg, ctx_timesteps)
            last_pred = action_preds[:, -1, :]
            action = torch.argmax(last_pred, dim=-1)

            pos = action.float() - 1.0
            step_reward = pos * market_returns[:, t]

            predicted_positions[:, t] = pos
            predicted_actions[:, t] = action
            realized_rewards[:, t] = step_reward

            # Overwrite placeholder with the real action (visible to future steps)
            hist_actions[-1] = action

            if rtg_rollout_mode == "autoregressive":
                current_rtg = current_rtg - step_reward.unsqueeze(-1)
            current_pos = pos

            # Trim lists to bound memory on long rollouts
            if len(hist_states) > context_len:
                hist_states = hist_states[-context_len:]
                hist_actions = hist_actions[-context_len:]
                hist_rtg = hist_rtg[-context_len:]
                hist_timesteps = hist_timesteps[-context_len:]

    return realized_rewards, predicted_positions, predicted_actions


def evaluate_baselines(market_returns: torch.Tensor):
    """Evaluate classic quantitative baseline policies.

    Policies: Buy & Hold, Oracle (perfect foresight), Momentum, Mean Reversion.

    Returns
    -------
    trajectories_dict : {name: cumulative-PnL array}
    metrics_dict      : {name: metrics dict}
    """
    metrics_dict = {}
    trajectories_dict = {}

    bnh_pnl = market_returns.cumsum(dim=1)
    trajectories_dict["Buy & Hold"] = bnh_pnl.mean(dim=0).cpu().numpy()
    metrics_dict["Buy & Hold"] = compute_financial_metrics(market_returns)

    oracle_pos = torch.where(
        market_returns > 0, 1.0,
        torch.where(market_returns < 0, -1.0, 0.0)
    )
    oracle_rewards = oracle_pos * market_returns
    oracle_pnl = oracle_rewards.cumsum(dim=1)
    trajectories_dict["Oracle"] = oracle_pnl.mean(dim=0).cpu().numpy()
    metrics_dict["Oracle"] = compute_financial_metrics(oracle_rewards)

    mom_pos = torch.zeros_like(market_returns)
    mom_pos[:, 1:] = torch.sign(market_returns[:, :-1])
    mom_rewards = mom_pos * market_returns
    mom_pnl = mom_rewards.cumsum(dim=1)
    trajectories_dict["Momentum"] = mom_pnl.mean(dim=0).cpu().numpy()
    metrics_dict["Momentum"] = compute_financial_metrics(mom_rewards)

    mr_pos = torch.zeros_like(market_returns)
    mr_pos[:, 1:] = -torch.sign(market_returns[:, :-1])
    mr_rewards = mr_pos * market_returns
    mr_pnl = mr_rewards.cumsum(dim=1)
    trajectories_dict["Mean Reversion"] = mr_pnl.mean(dim=0).cpu().numpy()
    metrics_dict["Mean Reversion"] = compute_financial_metrics(mr_rewards)

    return trajectories_dict, metrics_dict


# =============================================================================
# 3. Visualization Generators
# =============================================================================

def plot_sharpe_comparison(df_metrics: pd.DataFrame, save_path: Path):
    """Horizontal bar chart comparing Sharpe ratios across all policies."""
    fig, ax = plt.subplots(figsize=(10, 5.8))
    df_sorted = df_metrics.sort_values(by="Sharpe", ascending=True)
    agents = df_sorted.index.tolist()
    sharpes = df_sorted["Sharpe"].values
    colors = ["tab:green" if s > 0 else "tab:red" for s in sharpes]
    ax.barh(agents, sharpes, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Sharpe Ratio (Unannualized)")
    ax.set_title("Risk-Adjusted Performance Comparison")
    ax.grid(axis="x", alpha=0.3)
    fig.text(
        0.5,
        0.01,
        "Sharpe here = mean over episodes of (mean step reward / std step reward); "
        "near-zero DT variance ⇒ unstable ratios. Oracle is non-causal.",
        ha="center",
        fontsize=8,
        style="italic",
        color="0.35",
    )
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pnl_curves(
    trajectories_dict: dict,
    save_path: Path,
    *,
    full_title: str | None = None,
    zoom_title: str | None = None,
    norm_title: str | None = None,
):
    """Cumulative PnL: full scale (top), DT zoom (middle), optional Oracle-normalized (bottom).

    Oracle uses perfect foresight; DT does not — raw scales are not comparable. The
    normalized panel divides each series by ``|Oracle final cumulative PnL|`` so DT
    curves are visible on a common relative scale (Oracle ends at ±1).
    """
    dt_items = {k: v for k, v in trajectories_dict.items() if "DT" in k}
    oracle_key = "Oracle"
    has_oracle = oracle_key in trajectories_dict
    nrows = 3 if (has_oracle and dt_items) else 2
    height = 11 if nrows == 3 else 9
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(12, height),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.9, 0.75] if nrows == 3 else [1.0, 0.85], "hspace": 0.18},
    )
    if nrows == 2:
        ax_full, ax_dt = axes
        ax_norm = None
    else:
        ax_full, ax_dt, ax_norm = axes

    for agent, traj in trajectories_dict.items():
        if agent == "Oracle":
            ax_full.plot(traj, label=agent, linestyle="--", color="gold", alpha=0.8)
        elif "DT" in agent:
            ax_full.plot(traj, label=agent, linewidth=2.5)
        else:
            ax_full.plot(traj, label=agent, alpha=0.6)

    ax_full.axhline(0, color="black", linewidth=0.5)
    ax_full.set_title(
        full_title
        or "Cumulative PnL (full scale — DT may sit near 0 vs Oracle; see lower panels)"
    )
    ax_full.set_ylabel("Cumulative PnL")
    ax_full.legend(loc="upper left", fontsize=8)
    ax_full.grid(True, alpha=0.3)
    o_fin = None
    if has_oracle:
        o_fin = float(np.asarray(trajectories_dict[oracle_key], dtype=np.float64)[-1])
        ax_full.text(
            0.99,
            0.02,
            f"Oracle end PnL = {o_fin:.4g} (upper bound with foresight; not achievable by DT)",
            transform=ax_full.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            style="italic",
            color="0.35",
        )

    if dt_items:
        for agent, traj in dt_items.items():
            ax_dt.plot(traj, label=agent, linewidth=2.5)
        ax_dt.axhline(0, color="black", linewidth=0.5)
        stacked = np.stack(
            [np.asarray(v, dtype=np.float64) for v in dt_items.values()]
        )
        y_min, y_max = float(stacked.min()), float(stacked.max())
        span = y_max - y_min
        pad = max(span * 0.12, 1e-9)
        if span < 1e-12:
            ax_dt.set_ylim(-0.01, 0.01)
        else:
            ax_dt.set_ylim(y_min - pad, y_max + pad)
        ax_dt.set_title(
            zoom_title or "DT cumulative PnL (zoomed — same units as top panel)"
        )
        ax_dt.set_ylabel("Cumulative PnL")
        ax_dt.legend(loc="best", fontsize=8)
        ax_dt.grid(True, alpha=0.3)
    else:
        ax_dt.set_visible(False)

    if ax_norm is not None and has_oracle and o_fin is not None:
        den = max(abs(o_fin), 1e-12)
        ax_norm.axhline(0, color="black", linewidth=0.5)
        for agent, traj in trajectories_dict.items():
            y = np.asarray(traj, dtype=np.float64) / den
            if agent == "Oracle":
                ax_norm.plot(y, label=agent, linestyle="--", color="gold", alpha=0.85)
            elif "DT" in agent:
                ax_norm.plot(y, label=agent, linewidth=2.2)
            else:
                ax_norm.plot(y, label=agent, alpha=0.55)
        ax_norm.set_title(
            norm_title
            or f"Normalized by |Oracle terminal PnL| = {den:.4g} (dimensionless)"
        )
        ax_norm.set_xlabel("Time Steps")
        ax_norm.set_ylabel("Cum. PnL / |Oracle_T|")
        ax_norm.legend(loc="best", fontsize=7)
        ax_norm.grid(True, alpha=0.3)
        ax_norm.text(
            0.99,
            0.02,
            "Oracle → ±1 at T; DT shows fraction of that counterfactual ceiling.",
            transform=ax_norm.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            style="italic",
            color="0.35",
        )

    if nrows == 2:
        ax_dt.set_xlabel("Time Steps")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_drawdown_curves(
    realized_rewards_dict: dict[str, np.ndarray], save_path: Path
):
    """Plot drawdown-over-time for each agent from their mean reward series.

    Parameters
    ----------
    realized_rewards_dict : {agent_name: mean_rewards_1d_array}
    save_path             : Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(12, 5.5))

    for agent, rewards in realized_rewards_dict.items():
        rewards_arr = np.asarray(rewards, dtype=np.float64)
        cum_pnl = np.cumsum(rewards_arr)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdown = running_max - cum_pnl

        ls = "--" if agent == "Oracle" else "-"
        lw = 2.5 if "DT" in agent else 1.2
        alpha = 0.8 if "DT" in agent else 0.55
        ax.plot(drawdown, label=agent, linestyle=ls, linewidth=lw, alpha=alpha)

    ax.set_title("Drawdown Over Time")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Absolute Drawdown")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.text(
        0.5,
        0.01,
        "Oracle drawdown uses foresight; DT drawdown is on-policy. Scales differ by construction.",
        ha="center",
        fontsize=8,
        style="italic",
        color="0.35",
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_action_distribution_by_rtg(
    actions_by_rtg: dict[str, np.ndarray],
    save_path: Path,
    *,
    rollout_note: str | None = None,
):
    """Bar charts showing action (Short/Flat/Long) distributions per RTG target.

    Parameters
    ----------
    actions_by_rtg : {label: array of integer actions (0,1,2), any shape — raveled}
    save_path      : Output PNG path.
    rollout_note   : Optional second line under the main title (e.g. RTG rollout mode).
    """
    n = len(actions_by_rtg)
    if n == 0:
        return

    fig, axes = plt.subplots(2, n, figsize=(max(4 * n, 6), 7), sharex="col")
    if n == 1:
        axes = axes.reshape(2, 1)

    action_labels = ["Short", "Flat", "Long"]
    action_colors = ["tab:red", "tab:gray", "tab:green"]

    for j, (label, actions) in enumerate(actions_by_rtg.items()):
        actions_arr = np.asarray(actions, dtype=int).ravel()
        counts = np.bincount(actions_arr, minlength=3).astype(np.float64)
        total = max(float(counts.sum()), 1.0)
        pcts = counts / total * 100.0
        ent = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                ent -= p * np.log(p + 1e-12)
        ent = max(float(ent), 0.0)

        title = label if str(label).startswith("DT ") else f"DT ({label})"

        ax0 = axes[0, j]
        bars = ax0.bar(action_labels, counts, color=action_colors, edgecolor="black", linewidth=0.4)
        ax0.set_title(title, fontsize=10)
        ax0.set_ylabel("Count")
        ax0.grid(axis="y", alpha=0.3)
        ax0.set_ylim(0, max(total * 1.08, 1.0))
        for rect, cnt, pc in zip(bars, counts, pcts):
            h = rect.get_height()
            ax0.text(
                rect.get_x() + rect.get_width() / 2.0,
                h + total * 0.01,
                f"{int(cnt)}\n({pc:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax1 = axes[1, j]
        ax1.barh(action_labels, pcts, color=action_colors, edgecolor="black", linewidth=0.4)
        ax1.set_xlabel("Share of steps (%)")
        ax1.set_xlim(0, 105)
        ax1.axvline(100.0 / 3.0, color="navy", linestyle=":", linewidth=1.0, alpha=0.6)
        ax1.grid(axis="x", alpha=0.3)
        ax1.text(
            0.98,
            0.02,
            f"H={ent:.3f}\n(n={int(total)})",
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
        )

    note = ""
    if rollout_note:
        note = f"\n{rollout_note}"
    fig.suptitle(
        "DT action mix per target RTG (argmax at each step)" + note,
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.01,
        "33% line = uniform mix. Low H ⇒ near-deterministic policy. High RTG + wrong RTG map ⇒ Flat (see proportional offline RTG).",
        ha="center",
        fontsize=8,
        style="italic",
        color="0.35",
    )
    plt.tight_layout(rect=(0, 0.04, 1, 0.96))
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_advanced_metrics_comparison(
    df_metrics: pd.DataFrame, save_path: Path
):
    """Multi-panel bar charts for advanced financial metrics across all agents.

    Panels: Sortino, CVaR_95, CVaR_99, MaxDD, Calmar, ProfitFactor, HitRatio.
    """
    metric_cols = ["Sortino", "CVaR_95", "CVaR_99", "MaxDD", "Calmar", "ProfitFactor", "HitRatio"]
    available = [c for c in metric_cols if c in df_metrics.columns]
    if not available:
        return

    n = len(available)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = np.array(axes).ravel()

    agents = df_metrics.index.tolist()
    x = np.arange(len(agents))
    bar_width = 0.6

    for i, metric in enumerate(available):
        ax = axes_flat[i]
        values = df_metrics[metric].values.astype(float)
        colors = ["tab:green" if v > 0 else "tab:red" for v in values]
        if metric in ("MaxDD", "CVaR_95", "CVaR_99", "VaR_95", "VaR_99"):
            colors = ["tab:red"] * len(values)
        bars = ax.bar(x, values, width=bar_width, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=35, ha="right", fontsize=7)
        ax.set_title(metric, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.grid(axis="y", alpha=0.3)

    for j in range(len(available), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Advanced Financial Metrics Comparison", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Oracle / momentum / mean-rev use different information sets than DT; bars are not 'fair' head-to-head.",
        ha="center",
        fontsize=8,
        style="italic",
        color="0.35",
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.98))
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_inference_time(
    inference_times: dict[str, float], save_path: Path
):
    """Horizontal bar chart of per-agent inference times (milliseconds).

    Parameters
    ----------
    inference_times : {agent_name: elapsed_seconds}
    save_path       : Output PNG path.
    """
    agents = list(inference_times.keys())
    times_ms = [inference_times[a] * 1000.0 for a in agents]

    fig, ax = plt.subplots(figsize=(10, max(3.8, len(agents) * 0.5 + 1.2)))
    colors = ["tab:blue" if "DT" in a else "tab:orange" for a in agents]
    ax.barh(agents, times_ms, color=colors)
    ax.set_xlabel("Inference Time (ms)")
    ax.set_title("Agent Inference / Rollout Time Comparison")
    ax.grid(axis="x", alpha=0.3)

    for i, (val, ag) in enumerate(zip(times_ms, agents)):
        ax.text(val * 1.01, i, f"{val:.1f} ms", va="center", fontsize=8)

    fig.text(
        0.5,
        0.01,
        "Baselines share one timing slice; each DT line is a full batched rollout for one RTG target.",
        ha="center",
        fontsize=8,
        style="italic",
        color="0.35",
    )
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(save_path, dpi=300)
    plt.close()


# =============================================================================
# 4. Main Execution Pipeline
# =============================================================================

def evaluate_model(
    model_path,
    data_path,
    eval_cfg,
    model_cfg,
    plot_dir,
    state_representation: str = "raw",
    train_data_path: str | None = None,
    generator_window_size: int = 100,
    generator_price_offset: float = 10.0,
    generator_reward_type: str = "mid_price",
    generator_reward_shaping: dict[str, Any] | None = None,
):
    """Encapsulated entry point for the Hydra orchestrator.

    Evaluates the trained Decision Transformer against classic baselines and
    generates a comprehensive suite of financial visualisations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing evaluations on: {device}")

    out_path = Path(plot_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rtg_source = getattr(eval_cfg, "rtg_source", "manual")
    target_rtgs = resolve_target_rtgs(eval_cfg, train_data_path)
    if not target_rtgs:
        raise ValueError(
            "No evaluation RTG targets: set evaluation.target_rtgs (manual) or "
            "evaluation.train_rtg_percentiles with rtg_source=train_percentiles."
        )
    print(
        f"\nRTG conditioning: source={rtg_source!r} | "
        f"{len(target_rtgs)} targets: "
        f"[{', '.join(f'{v:.6g}' for v in target_rtgs[:8])}"
        f"{'…' if len(target_rtgs) > 8 else ''}]"
    )

    rtg_rollout_mode = getattr(eval_cfg, "rtg_rollout_mode", "anchored_offline")
    print(
        f"RTG rollout mode: {rtg_rollout_mode!r} — "
        "autoregressive: Rₜ₊₁ = Rₜ − rₜ (locks to Flat if rₜ≡0). "
        "anchored_offline: R_{b,t}=ref_{b,t}·(target/ref_{b,0}) if |ref_{b,0}|>ε "
        "else additive offset."
    )

    max_ts = int(getattr(model_cfg, "max_timestep", 10_000))

    # ---- 1. Load model -------------------------------------------------------
    print(f"\nLoading model from {model_path}...")
    model = build_model(model_cfg).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    raw_sd = checkpoint["model_state_dict"]
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}
    model.load_state_dict(clean_sd)

    # ---- 2. Load test data ---------------------------------------------------
    print(f"Loading testing data from {data_path}...")
    trajectories = torch.load(data_path, map_location="cpu", weights_only=False)
    if isinstance(trajectories, dict):
        trajectories = list(trajectories.values())

    trajectories = trajectories[: eval_cfg.max_eval_trajectories]
    min_len = min(len(traj["states"]) for traj in trajectories)

    states_batch = torch.stack([
        torch.tensor(traj["states"][:min_len], dtype=torch.float32)
        for traj in trajectories
    ])[:, :, : model_cfg.state_dim].to(device)

    print(f"Batch constructed: Shape {states_batch.shape}")

    reference_rtg = torch.stack([
        torch.tensor(traj["rtg"][:min_len, 0], dtype=torch.float32)
        for traj in trajectories
    ]).to(device)

    # ---- 3. Market returns ---------------------------------------------------
    market_returns = get_market_returns(
        states_batch, state_representation=state_representation
    )

    all_trajectories: dict[str, np.ndarray] = {}
    all_metrics: dict[str, dict] = {}
    all_mean_rewards: dict[str, np.ndarray] = {}
    inference_times: dict[str, float] = {}
    actions_by_rtg: dict[str, np.ndarray] = {}

    # ---- 4. Baselines --------------------------------------------------------
    print("\n--- Evaluating Baselines ---")
    t0 = time.perf_counter()
    base_trajs, base_metrics = evaluate_baselines(market_returns)
    baseline_elapsed = time.perf_counter() - t0

    all_trajectories.update(base_trajs)
    all_metrics.update(base_metrics)

    for name in base_trajs:
        inference_times[name] = baseline_elapsed / len(base_trajs)
        mr = market_returns.mean(dim=0).cpu().numpy()
        all_mean_rewards[name] = mr

    # ---- 5. Decision Transformer across RTG targets --------------------------
    print("\n--- Evaluating Decision Transformer ---")
    for rtg in target_rtgs:
        print(f"  [>] Rolling out DT with Target RTG = {rtg:.6g}...")

        t0 = time.perf_counter()
        rr_kw: dict[str, Any] = dict(
            model=model,
            states=states_batch,
            market_returns=market_returns,
            target_rtg=rtg,
            context_len=eval_cfg.context_len,
            device=device,
            max_timestep=max_ts,
            rtg_rollout_mode=rtg_rollout_mode,
        )
        if rtg_rollout_mode == "anchored_offline":
            rr_kw["reference_rtg"] = reference_rtg
        realized_rewards, predicted_pos, predicted_actions = (
            vectorized_autoregressive_rollout(**rr_kw)
        )
        elapsed = time.perf_counter() - t0

        pa = predicted_actions.long()
        n_steps = pa.numel()
        n_nonflat = int((pa != 1).sum().item())
        mean_abs_pos = float(predicted_pos.abs().mean().item())
        print(
            f"      steps={n_steps}  non-Flat={n_nonflat} ({100.0 * n_nonflat / max(n_steps, 1):.1f}%)  "
            f"mean|pos|={mean_abs_pos:.4f}"
        )

        agent_name = f"DT (RTG={rtg:.6g})"
        dt_pnl = realized_rewards.cumsum(dim=1)

        all_trajectories[agent_name] = dt_pnl.mean(dim=0).cpu().numpy()
        all_mean_rewards[agent_name] = realized_rewards.mean(dim=0).cpu().numpy()
        inference_times[agent_name] = elapsed

        m = compute_financial_metrics(realized_rewards)
        m["F1_macro"] = compute_directional_f1(predicted_actions, market_returns)
        all_metrics[agent_name] = m

        actions_by_rtg[agent_name] = predicted_actions.cpu().numpy()

    # ---- 6. Results table ----------------------------------------------------
    df_metrics = pd.DataFrame.from_dict(all_metrics, orient="index")

    print("\n" + "=" * 85)
    print("BACKTEST SUMMARY (Test Set)")
    print("=" * 85)
    cols_display = [
        "PnL", "Sharpe", "Sortino", "MaxDD", "Calmar",
        "CVaR_95", "ProfitFactor", "HitRatio",
    ]
    for _, row in df_metrics.iterrows():
        f1_s = (
            f"{row['F1_macro']:.4f}"
            if "F1_macro" in row and pd.notna(row.get("F1_macro"))
            else "—"
        )
        col_vals = "  |  ".join(
            f"{c}: {row[c]:+.4f}" for c in cols_display if c in row
        )
        print(f"  {row.name:<28}  {col_vals}  |  F1: {f1_s}")
    print("=" * 85)

    # ---- 7. Visualisations ---------------------------------------------------
    print("\nGenerating visualisation plots...")

    plot_pnl_curves(
        all_trajectories,
        out_path / "pnl_comparison_curves.png",
        full_title=(
            "Cumulative PnL — mean over random test episodes "
            f"(n={len(trajectories)}, len={min_len}); RTG source={rtg_source!r}; "
            f"rollout={rtg_rollout_mode}"
        ),
    )
    plot_sharpe_comparison(df_metrics, out_path / "sharpe_ratio_bars.png")
    plot_drawdown_curves(all_mean_rewards, out_path / "drawdown_curves.png")
    plot_action_distribution_by_rtg(
        actions_by_rtg,
        out_path / "action_distribution_by_rtg.png",
        rollout_note=(
            f"RTG rollout: {rtg_rollout_mode} (offline: proportional to ref RTG, R₀=target)"
        ),
    )
    plot_advanced_metrics_comparison(
        df_metrics, out_path / "advanced_metrics_comparison.png"
    )
    plot_inference_time(inference_times, out_path / "inference_time_comparison.png")

    # ---- 8. Per-stock chronological Day 10 episodes (optional) ---------------
    if getattr(eval_cfg, "continuous_day10_plot", False):
        print("\n--- Per-stock chronological Day 10 rollouts ---")
        try:
            X_day = _load_fi2010_test_lob_matrix()
            boundaries = detect_stock_boundaries(X_day)
            n_stocks = len(boundaries) - 1
            print(
                f"Detected {n_stocks} stocks in Day 10 test fold: "
                f"{[boundaries[i+1]-boundaries[i] for i in range(n_stocks)]} events each"
            )

            for si in range(n_stocks):
                s_start, s_end = boundaries[si], boundaries[si + 1]
                X_stock = X_day[s_start:s_end]

                tails, ret1d = precompute_chronological_lob_tails_and_returns(
                    X_stock,
                    window_size=generator_window_size,
                    state_representation=state_representation,
                    price_offset=generator_price_offset,
                    reward_type=generator_reward_type,
                    reward_shaping=generator_reward_shaping,
                )
                t_steps, f_dim = tails.shape
                state_dim = int(model_cfg.state_dim)
                chron_states = np.zeros((1, t_steps, state_dim), dtype=np.float32)
                chron_states[0, :, :f_dim] = tails
                chron_states_t = torch.from_numpy(chron_states).to(device)
                chron_returns = torch.from_numpy(ret1d.reshape(1, -1)).to(device)
                r1 = chron_returns.squeeze(0).float()
                chron_ref_rtg = torch.flip(
                    torch.cumsum(torch.flip(r1, dims=(0,)), dim=0), dims=(0,)
                ).unsqueeze(0)

                chron_trajs: dict[str, np.ndarray] = {}
                cbase_trajs, _ = evaluate_baselines(chron_returns)
                chron_trajs.update(cbase_trajs)

                for rtg in target_rtgs:
                    cr_kw: dict[str, Any] = dict(
                        model=model,
                        states=chron_states_t,
                        market_returns=chron_returns,
                        target_rtg=rtg,
                        context_len=eval_cfg.context_len,
                        device=device,
                        max_timestep=max_ts,
                        rtg_rollout_mode=rtg_rollout_mode,
                    )
                    if rtg_rollout_mode == "anchored_offline":
                        cr_kw["reference_rtg"] = chron_ref_rtg.to(device)
                    rr, _, _ = vectorized_autoregressive_rollout(**cr_kw)
                    label = f"DT (RTG={rtg:.6g})"
                    chron_trajs[label] = rr.cumsum(dim=1).squeeze(0).cpu().numpy()

                fname = f"pnl_continuous_day10_stock{si + 1}.png"
                plot_pnl_curves(
                    chron_trajs,
                    out_path / fname,
                    full_title=(
                        f"Cumulative PnL — Day 10 Stock {si + 1}/{n_stocks} "
                        f"({t_steps} steps; mid-price returns)"
                    ),
                    zoom_title=f"DT cumulative PnL — Day 10 Stock {si + 1} (zoomed)",
                )
                print(f"  Stock {si + 1}: saved {fname}")

        except Exception as exc:
            print(
                f"Skipping chronological Day 10 plots ({exc!r}). "
                "Requires kagglehub + FI-2010 cache; set continuous_day10_plot: false to silence."
            )

    print(f"Visualisations saved to '{out_path.absolute()}'.")


# =============================================================================
# CLI entry-point (used when running dt_viz.py directly)
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DT Offline Evaluation & Visualizations")
    parser.add_argument("--model_path", type=str, default="dt_model_ep15.pt")
    parser.add_argument("--data_path", type=str, default="test_trajectories.pt")
    parser.add_argument(
        "--target_rtgs", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.5]
    )
    parser.add_argument(
        "--rtg_source",
        type=str,
        default="manual",
        choices=("manual", "train_percentiles"),
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="data/train_trajectories.pt",
        help="Used when --rtg_source train_percentiles",
    )
    parser.add_argument(
        "--train_rtg_percentiles",
        type=float,
        nargs="+",
        default=[10.0, 25.0, 50.0, 75.0, 90.0],
    )
    parser.add_argument("--context_len", type=int, default=100)
    parser.add_argument("--max_eval_trajectories", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="visualizations")
    parser.add_argument(
        "--state_representation",
        type=str,
        default="raw",
        choices=("raw", "log_returns", "bps"),
    )
    parser.add_argument(
        "--continuous_day10_plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, download/load FI-2010 test LOB and plot one full chronological episode.",
    )
    parser.add_argument(
        "--rtg_rollout_mode",
        type=str,
        default="anchored_offline",
        choices=("autoregressive", "anchored_offline"),
        help="How RTG is conditioned during rollout (see evaluate_model docstring).",
    )
    parser.add_argument("--generator_window_size", type=int, default=100)
    parser.add_argument("--generator_price_offset", type=float, default=10.0)
    args = parser.parse_args()

    from types import SimpleNamespace

    cli_model_cfg = SimpleNamespace(
        architecture="transformer",
        state_dim=41,
        act_dim=3,
        d_model=128,
        n_heads=4,
        n_layers=3,
        max_timestep=10000,
        dropout=0.1,
        cnn_channels=64,
        cnn_kernel_size=3,
    )
    cli_eval_cfg = SimpleNamespace(
        target_rtgs=args.target_rtgs,
        rtg_source=args.rtg_source,
        train_rtg_percentiles=args.train_rtg_percentiles,
        context_len=args.context_len,
        max_eval_trajectories=args.max_eval_trajectories,
        continuous_day10_plot=args.continuous_day10_plot,
        rtg_rollout_mode=args.rtg_rollout_mode,
    )

    evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        eval_cfg=cli_eval_cfg,
        model_cfg=cli_model_cfg,
        plot_dir=args.out_dir,
        state_representation=args.state_representation,
        train_data_path=args.train_data_path,
        generator_window_size=args.generator_window_size,
        generator_price_offset=args.generator_price_offset,
    )
