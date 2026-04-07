#!/usr/bin/env python3
"""
Optimized Trajectories Generator for Decision Transformer LOB Trading
Optimized for high core count processors (e.g., AMD Ryzen 9 9950X)
Generates both Train (Days 1-9) and Test (Day 10) datasets.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from src.env.lob_trading_env import LOBTradingEnv

# -----------------------------------------------------------------------------
# MULTIPROCESSING OPTIMIZATIONS
# -----------------------------------------------------------------------------
# Disable numerical library multithreading in workers to prevent CPU thrashing
# Since we use 32 parallel processes, we want each process to be single-threaded.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Global variables for workers. 
# This prevents pickling the large X_data array thousands of times.
# Each worker process loads the environment and data exactly ONCE in RAM.
global_env = None
global_labels = None
global_reward_horizon = None  # None means sum to episode end (standard DT)


# -----------------------------------------------------------------------------
# FI-2010 MULTI-STOCK UTILITIES
# -----------------------------------------------------------------------------

def detect_stock_boundaries(
    lob_data: np.ndarray, n_stocks: int = 5
) -> list[int]:
    """Detect stock boundaries in FI-2010 concatenated LOB data.

    FI-2010 cross-fold files concatenate data from ``n_stocks`` Finnish
    stocks along the time axis.  This function finds the ``n_stocks - 1``
    largest absolute mid-price jumps, which correspond to the transitions
    between different equities.

    A top-K approach is used instead of a threshold because the z-scored
    data has very small tick-to-tick differences, making any fixed or
    median-relative threshold unreliable.

    Returns a sorted list of boundary indices **including** 0 and
    ``len(lob_data)`` so that ``zip(boundaries[:-1], boundaries[1:])``
    yields per-stock slices.
    """
    if n_stocks < 2:
        return [0, len(lob_data)]

    mid = (lob_data[:, 0] + lob_data[:, 2]) / 2.0
    abs_diff = np.abs(np.diff(mid))

    n_boundaries = n_stocks - 1
    if len(abs_diff) < n_boundaries:
        return [0, len(lob_data)]

    # Pick the n_boundaries largest jumps (stock transitions).
    top_indices = np.argsort(abs_diff)[-n_boundaries:]
    jump_indices = sorted((top_indices + 1).tolist())
    return [0] + jump_indices + [len(lob_data)]


def split_by_stock(
    X_data: np.ndarray,
    y_data: np.ndarray | None = None,
    n_stocks: int = 5,
) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """Split concatenated FI-2010 data into per-stock segments.

    Returns a list of ``(X_stock, y_stock)`` pairs (``y_stock`` is *None*
    when ``y_data`` is not provided).
    """
    boundaries = detect_stock_boundaries(X_data, n_stocks=n_stocks)
    segments: list[tuple[np.ndarray, np.ndarray | None]] = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        seg_y = y_data[start:end] if y_data is not None else None
        segments.append((X_data[start:end], seg_y))
    return segments


# -----------------------------------------------------------------------------
# RETURN-TO-GO COMPUTATION
# -----------------------------------------------------------------------------
def compute_rtg(rewards: np.ndarray, horizon: int | None) -> np.ndarray:
    """
    Compute return-to-go for every timestep t in an episode of length T.

    Full horizon (horizon=None or horizon<=0):
        R̂_t = sum(r_t, r_{t+1}, ..., r_{T-1})   [original DT behaviour]

    Bounded horizon H (horizon > 0):
        R̂_t = sum(r_t, r_{t+1}, ..., r_{min(t+H, T)-1})
        Rewards more than H steps in the future are excluded.

    Both cases are computed in O(T) using a prefix-sum array.
    """
    T = len(rewards)

    # Build prefix sum: prefix[i] = sum of rewards[0..i-1]
    prefix = np.empty(T + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(rewards, out=prefix[1:])

    if horizon is None or horizon <= 0:
        # Sum from t to end of episode
        end_indices = np.full(T, T, dtype=np.int64)
    else:
        # Sum at most `horizon` steps starting from t, clamped to episode end
        end_indices = np.minimum(np.arange(T, dtype=np.int64) + horizon, T)

    start_indices = np.arange(T, dtype=np.int64)
    return (prefix[end_indices] - prefix[start_indices]).astype(np.float32)


def init_worker(
    X_data,
    y_data,
    window_size,
    episode_length,
    reward_type="mid_price",
    reward_shaping=None,
    state_representation="raw",
    price_offset=10.0,
    reward_horizon=None,
):
    """
    Initializer function called ONCE per worker process when the Pool is created.
    It sets up the trading environment in the worker's local memory space.
    """
    global global_env, global_labels, global_reward_horizon
    global_reward_horizon = reward_horizon
    rs = reward_shaping or {}
    global_env = LOBTradingEnv(
        X_data,
        window_size=window_size,
        transaction_cost=0.0,
        episode_length=episode_length,
        reward_type=reward_type,
        drawdown_coef=float(rs.get("drawdown_coef", 0.0)),
        variance_coef=float(rs.get("variance_coef", 0.0)),
        time_in_market_coef=float(rs.get("time_in_market_coef", 0.0)),
        variance_window=int(rs.get("variance_window", 20)),
        state_representation=state_representation,
        price_offset=float(price_offset),
    )
    global_labels = y_data

# -----------------------------------------------------------------------------
# POLICY DEFINITIONS
# -----------------------------------------------------------------------------
def flatten_state(obs):
    """Concatenates the last LOB snapshot and the current position into a single vector."""
    return np.concatenate([obs["lob_window"][-1], obs["position"]])

# Policy 1: Random (Baseline)
def random_policy(lob_window, pos, **kwargs):
    return np.random.randint(0, 3)

# Policy 2: Imbalance (Volume based)
def imbalance_policy(lob_window, pos, **kwargs):
    ask_vol = lob_window[-1, 1]
    bid_vol = lob_window[-1, 3]
    diff = bid_vol - ask_vol
    if diff > 0.15: return 2   # Go long
    if diff < -0.15: return 0  # Go short
    return 1                   # Stay flat

# Policy 3: Momentum (Price trend)
def momentum_policy(lob_window, pos, **kwargs):
    mid = (lob_window[:, 0] + lob_window[:, 2]) / 2.0
    delta = mid[-1] - mid[-20]
    if delta > 0.003: return 2
    if delta < -0.003: return 0
    return 1

# Policy 4: Oracle (Cheating: Looks at actual next price)
def oracle_policy(lob_window, pos, env=None, **kwargs):
    t = env._current_step
    mid_now  = env.mid_prices[t]
    mid_next = env.mid_prices[t + 1]
    delta = mid_next - mid_now
    if delta > 0: return 2
    if delta < 0: return 0
    return 1

# Policy 5: Label-momentum (Follows FI-2010 predefined labels)
def label_momentum_policy(lob_window, pos, env=None, labels=None, **kwargs):
    t = env._current_step
    label = int(labels[t, 3])
    if label == 1: return 0
    if label == 3: return 2
    return 1

# Policy 6: Mean-reversion (Contrarian, intentionally bad for dataset diversity)
def mean_reversion_policy(lob_window, pos, env=None, labels=None, **kwargs):
    t = env._current_step
    label = int(labels[t, 3])
    if label == 1: return 2
    if label == 3: return 0
    return 1

POLICIES = {
    "random":     random_policy,
    "imbalance":  imbalance_policy,
    "momentum":   momentum_policy,
    "oracle":     oracle_policy,
    "label_mom":  label_momentum_policy,
    "mean_rev":   mean_reversion_policy,
}

# -----------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------

def plot_lob_features(X_data: np.ndarray, name: str, save_dir: str = "plots"):
    """Visualise raw LOB market data — mirrors the introduction notebook.

    Three panels are saved as separate files:

    1. ``{name}_lob_price_series.png``
       Best ask, best bid and mid-price over the full time horizon.

    2. ``{name}_lob_feature_distributions.png``
       Histogram distributions of all 10 Level-1 features (ask/bid prices
       and volumes across the 5 LOB levels).

    3. ``{name}_lob_autocorrelation.png``
       Autocorrelation function (ACF) of the mid-price first-difference
       (i.e., tick-by-tick returns) up to 50 lags.

    Parameters
    ----------
    X_data   : raw LOB array of shape (T, 40).
    name     : dataset split label ("Train" or "Test").
    save_dir : output directory.
    """
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    name_l = name.lower()

    # --- 1. Price series ---
    ask1 = X_data[:, 0]
    bid1 = X_data[:, 2]
    mid  = (ask1 + bid1) / 2.0

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ask1, label="Best Ask (L1)", linewidth=0.8, color="tab:red",   alpha=0.8)
    ax.plot(bid1, label="Best Bid (L1)", linewidth=0.8, color="tab:green", alpha=0.8)
    ax.plot(mid,  label="Mid Price",     linewidth=1.2, color="tab:blue",  alpha=0.9)
    ax.set_title(f"{name} — Best Bid / Ask / Mid-Price over Time", fontweight="bold")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{name_l}_lob_price_series.png", dpi=150)
    plt.close()

    # --- 2. Level-1 feature distributions ---
    # LOB columns: (ask_p, ask_v, bid_p, bid_v) repeated for 5 levels → 20 pairs
    level1_labels = [
        "Ask P L1", "Ask V L1", "Bid P L1", "Bid V L1",
        "Ask P L2", "Ask V L2", "Bid P L2", "Bid V L2",
        "Ask P L3", "Ask V L3",
    ]
    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    for i, (ax, label) in enumerate(zip(axes.ravel(), level1_labels)):
        ax.hist(X_data[:, i], bins=50, color="tab:blue", alpha=0.7, edgecolor="black")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Value", fontsize=7)
        ax.grid(True, alpha=0.25)
    plt.suptitle(
        f"{name} — Level-1 LOB Feature Distributions", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{name_l}_lob_feature_distributions.png", dpi=150)
    plt.close()

    # --- 3. Mid-price autocorrelation ---
    mid_returns = np.diff(mid)
    max_lags = min(50, len(mid_returns) - 1)
    acf_vals = np.array(
        [np.corrcoef(mid_returns[: len(mid_returns) - lag],
                     mid_returns[lag:])[0, 1]
         for lag in range(1, max_lags + 1)]
    )
    conf = 1.96 / np.sqrt(len(mid_returns))

    fig, ax = plt.subplots(figsize=(10, 4))
    lags = np.arange(1, max_lags + 1)
    ax.bar(lags, acf_vals, color="tab:blue", alpha=0.7)
    ax.axhline( conf, color="red", linestyle="--", linewidth=1.0, label="95% CI")
    ax.axhline(-conf, color="red", linestyle="--", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(
        f"{name} — Autocorrelation of Mid-Price Returns (ACF)", fontweight="bold"
    )
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{name_l}_lob_autocorrelation.png", dpi=150)
    plt.close()

    print(f"LOB feature plots for {name} saved to '{save_dir}/'.")


def plot_episode_cumulative_pnl(
    trajectories: list, dataset_name: str, save_dir: str = "plots"
):
    """One example cumulative-PnL trajectory per policy — mirrors image 2 in
    the introduction notebook.

    Parameters
    ----------
    trajectories  : list of trajectory dicts (output of ``generate_dataset``).
    dataset_name  : label ("Train" or "Test").
    save_dir      : output directory.
    """
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    by_policy: dict[str, list] = {p: [] for p in POLICIES}
    for t in trajectories:
        by_policy[t["policy"]].append(t)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    for ax, policy in zip(axes, POLICIES.keys()):
        pool = by_policy[policy]
        if not pool:
            ax.set_visible(False)
            continue
        example = pool[0]
        rewards = example["rewards"]
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.numpy()
        cum_pnl = np.cumsum(rewards)
        ax.plot(cum_pnl, linewidth=1.2, color="steelblue")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"{policy} — example episode")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumul. PnL")
        ax.grid(True, alpha=0.25)

    plt.suptitle(
        f"Cumulative PnL per policy — {dataset_name}", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        f"{save_dir}/{dataset_name.lower()}_episode_pnl_examples.png", dpi=150
    )
    plt.close()
    print(f"Episode PnL examples for {dataset_name} saved to '{save_dir}/'.")


def plot_distributions(trajectories, dataset_name, save_dir="plots"):
    """Generate and save distribution plots for a trajectory dataset.

    Files produced
    --------------
    ``{dataset_name.lower()}_policy_returns.png``
        Episode return histograms per policy.

    ``{dataset_name.lower()}_rtg_distribution.png``
        Global Return-to-Go distribution across all trajectories.

    ``{dataset_name.lower()}_action_distributions.png``
        Per-policy action (Short/Flat/Long) distribution bar charts.

    ``{dataset_name.lower()}_episode_pnl_examples.png``
        One example cumulative-PnL curve per policy.
    """
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    policy_returns: dict[str, list] = {p: [] for p in POLICIES.keys()}
    policy_actions: dict[str, list] = {p: [] for p in POLICIES.keys()}
    all_rtg: list[float] = []

    for t in trajectories:
        p = t["policy"]
        policy_returns[p].append(t["total_return"])
        actions = t["actions"]
        if isinstance(actions, torch.Tensor):
            policy_actions[p].extend(actions.tolist())
        else:
            policy_actions[p].extend(actions.tolist())
        rtg_flat = t["rtg"]
        if isinstance(rtg_flat, torch.Tensor):
            all_rtg.extend(rtg_flat.flatten().tolist())
        else:
            all_rtg.extend(rtg_flat.flatten().tolist())

    # 1. Episode return histograms per policy
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for i, (policy, rets) in enumerate(policy_returns.items()):
        ax = axes[i]
        if rets:
            ax.hist(rets, bins=30, alpha=0.7, edgecolor="black", color="tab:blue")
            mean_v = np.mean(rets)
            ax.axvline(
                mean_v, color="red", linestyle="dashed", linewidth=1.5,
                label=f"mean={mean_v:.4f}"
            )
            ax.legend(fontsize=8)
        ax.set_title(f"{policy} policy")
        ax.set_xlabel("Episode return")
        ax.set_ylabel("Count")

    plt.suptitle(
        f"Distribution of episode returns per policy", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{dataset_name.lower()}_policy_returns.png", dpi=150)
    plt.close()

    # 2. Global RTG distribution
    all_rtg_arr = np.asarray(all_rtg, dtype=np.float32)
    plt.figure(figsize=(10, 5))
    plt.hist(all_rtg_arr, bins=100, alpha=0.75, color="tab:purple", edgecolor="black")
    plt.title(
        f"Distribution of return-to-go across all trajectories", fontweight="bold"
    )
    plt.xlabel("Return-to-go")
    plt.ylabel("Count")
    global_mean = float(all_rtg_arr.mean())
    plt.axvline(
        global_mean, color="red", linestyle="dashed", linewidth=2,
        label=f"Global Mean: {global_mean:.4f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{dataset_name.lower()}_rtg_distribution.png", dpi=150)
    plt.close()

    # 3. Per-policy action distributions
    action_labels = ["Short", "Flat", "Long"]
    action_colors = ["tab:red", "tab:gray", "tab:green"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    axes = axes.ravel()
    for i, (policy, acts) in enumerate(policy_actions.items()):
        ax = axes[i]
        if acts:
            acts_arr = np.asarray(acts, dtype=int)
            counts = np.bincount(acts_arr, minlength=3)
            ax.bar(action_labels, counts, color=action_colors)
            ax.set_title(f"{policy} policy")
            ax.set_ylabel("Count")
            ax.grid(axis="y", alpha=0.3)
    plt.suptitle(
        f"Action Distribution per Policy — {dataset_name}", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{dataset_name.lower()}_action_distributions.png", dpi=150)
    plt.close()

    # 4. Episode cumulative PnL examples per policy
    plot_episode_cumulative_pnl(trajectories, dataset_name, save_dir)

    print(f"Visual plots for {dataset_name} generated in '{save_dir}/'.")

# -----------------------------------------------------------------------------
# WORKER EXECUTION
# -----------------------------------------------------------------------------
def rollout_worker(args):
    """
    Executes one full episode for a given policy and seed.
    CRITICAL: This function must return pure NumPy arrays, NOT torch tensors.
    Returning torch tensors via multiprocessing creates shared memory file descriptors (mmap),
    which leads to 'Cannot allocate memory (12)' errors when scaling to thousands of episodes.
    """
    policy_name, seed = args
    policy_fn = POLICIES[policy_name]
    
    # Access global variables initialized by init_worker
    env = global_env
    labels = global_labels
    
    obs, info = env.reset(seed=seed)
    states, actions, rewards = [], [], []
    done = False
    
    while not done:
        action = policy_fn(
            lob_window=obs["lob_window"],
            pos=obs["position"][0],
            env=env,
            labels=labels,
        )
        
        states.append(flatten_state(obs))
        actions.append(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
        
    # Convert lists to fast numpy arrays
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int64)
    rewards = np.array(rewards, dtype=np.float32)
    
    # Compute return-to-go with an optional bounded horizon
    returns_to_go = compute_rtg(rewards, global_reward_horizon)
    
    # Return standard Python/NumPy objects to avoid PyTorch shared memory leaks
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "rtg": np.expand_dims(returns_to_go, axis=1),
        "timesteps": np.arange(len(rewards), dtype=np.int64),
        "policy": policy_name,
        "total_return": float(rewards.sum())
    }

# -----------------------------------------------------------------------------
# DATASET GENERATION PIPELINE
# -----------------------------------------------------------------------------
def generate_dataset(
    X,
    y,
    num_episodes,
    output_file,
    num_workers,
    desc="Generating",
    window_size=100,
    episode_length=2000,
    reward_type="mid_price",
    reward_shaping=None,
    state_representation="raw",
    price_offset=10.0,
    reward_horizon=None,
):
    """
    Distributes the generation of trajectories across all CPU cores.
    Converts the returned NumPy arrays into PyTorch tensors safely in the main process.
    """
    horizon_str = str(reward_horizon) if reward_horizon else "full episode"
    print(f"\n--- Starting {desc} Generation ---")
    print(f"Data shape: {X.shape} | Episodes: {num_episodes} | Workers: {num_workers}")
    print(
        f"Reward: {reward_type} | state={state_representation} | "
        f"window={window_size} len={episode_length} | RTG horizon={horizon_str}"
    )
    
    # Create job list (assigning equal number of episodes to each policy)
    episodes_per_policy = num_episodes // len(POLICIES)
    jobs = []
    
    for policy_name in POLICIES.keys():
        # Offset seed to ensure unique randomness across policies
        seed_offset = list(POLICIES.keys()).index(policy_name) * 100000
        for i in range(episodes_per_policy):
            jobs.append((policy_name, seed_offset + i))
            
    all_trajectories = []
    
    # We use initializer to pass the heavy Data Arrays ONCE per worker.
    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(
            X,
            y,
            window_size,
            episode_length,
            reward_type,
            reward_shaping,
            state_representation,
            price_offset,
            reward_horizon,
        ),
    ) as pool:
        
        # Process jobs and show progress bar
        with tqdm(total=len(jobs), desc=desc, unit="ep") as pbar:
            for traj in pool.imap_unordered(rollout_worker, jobs, chunksize=16):
                
                # Convert NumPy arrays to PyTorch Tensors in the MAIN process.
                # This guarantees we do not exhaust OS file descriptors.
                torch_traj = {
                    "states": torch.from_numpy(traj["states"]),
                    "actions": torch.from_numpy(traj["actions"]),
                    "rewards": torch.from_numpy(traj["rewards"]),
                    "rtg": torch.from_numpy(traj["rtg"]),
                    "timesteps": torch.from_numpy(traj["timesteps"]),
                    "policy": traj["policy"],
                    "total_return": traj["total_return"]
                }
                
                all_trajectories.append(torch_traj)
                pbar.update()

    # Save to disk (skip when called in per-stock mode with output_file=None)
    if output_file is not None:
        torch.save(all_trajectories, output_file)
        print(f"Saved {len(all_trajectories)} trajectories to {output_file}")

    # Print short summary
    returns = [t["total_return"] for t in all_trajectories]
    print(f"Metrics -> Mean Return: {np.mean(returns):.4f} | Max Return: {np.max(returns):.4f}\n")

    return all_trajectories

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def _generate_per_stock(
    X_full: np.ndarray,
    y_full: np.ndarray,
    total_episodes: int,
    output_file: str,
    workers: int,
    split_label: str,
    plot_dir: str,
    *,
    window_size: int,
    episode_length: int,
    reward_type: str,
    reward_shaping: dict | None,
    state_representation: str,
    price_offset: float,
    reward_horizon: int | None,
) -> list:
    """Generate trajectories independently per stock, then combine.

    FI-2010 fold files concatenate 5 stocks along the time axis.
    Generating episodes from the concatenated array allows random windows
    to cross stock boundaries — mid-price jumps discontinuously and rewards
    become meaningless.  This helper splits the data first.
    """
    stocks = split_by_stock(X_full, y_full)
    n_stocks = len(stocks)
    total_events = sum(len(sX) for sX, _ in stocks)
    print(
        f"Detected {n_stocks} stocks in {split_label} data: "
        f"{[len(sX) for sX, _ in stocks]} events each "
        f"({total_events} total)"
    )

    # Visualise full concatenated LOB features (matches notebook)
    plot_lob_features(X_full, split_label, plot_dir)

    all_trajectories: list = []
    allocated = 0
    for stock_idx, (sX, sy) in enumerate(stocks):
        if len(sX) <= window_size + episode_length:
            print(
                f"  Stock {stock_idx + 1}: only {len(sX)} events — "
                f"too short for window={window_size}+episode={episode_length}, skipping."
            )
            continue

        # Proportional episode allocation; last stock absorbs rounding remainder
        if stock_idx < n_stocks - 1:
            n_eps = max(len(POLICIES), round(total_episodes * len(sX) / total_events))
        else:
            n_eps = max(len(POLICIES), total_episodes - allocated)
        allocated += n_eps

        trajs = generate_dataset(
            X=sX,
            y=sy,
            num_episodes=n_eps,
            output_file=None,
            num_workers=workers,
            desc=f"{split_label} Stock {stock_idx + 1}/{n_stocks}",
            window_size=window_size,
            episode_length=episode_length,
            reward_type=reward_type,
            reward_shaping=reward_shaping,
            state_representation=state_representation,
            price_offset=price_offset,
            reward_horizon=reward_horizon,
        )
        all_trajectories.extend(trajs)

    # Save combined dataset
    torch.save(all_trajectories, output_file)
    print(f"Saved {len(all_trajectories)} trajectories to {output_file}")

    # Distribution plots on the combined trajectories
    plot_distributions(all_trajectories, split_label, plot_dir)
    return all_trajectories


def generate_dataset_pipeline(
    train_episodes: int,
    test_episodes: int,
    workers: int,
    train_out: str,
    test_out: str,
    plot_dir: str,
    window_size: int = 100,
    episode_length: int = 2000,
    reward_type: str = "mid_price",
    reward_shaping: dict | None = None,
    state_representation: str = "raw",
    price_offset: float = 10.0,
    reward_horizon: int | None = None,
):
    """Encapsulated entry point for Hydra orchestrator."""
    # 1. Download Dataset via kagglehub
    print("Downloading/Locating FI-2010 dataset...")
    import kagglehub
    dataset_path = Path(kagglehub.dataset_download("ulfricirons/fi-2010"))

    # Locate Fold 9 files
    train_file = next(dataset_path.rglob("*NoAuction_Zscore*Training/Train*CF_9.txt"))
    test_file  = next(dataset_path.rglob("*NoAuction_Zscore*Testing/Test*CF_9.txt"))

    gen_kwargs = dict(
        window_size=window_size,
        episode_length=episode_length,
        reward_type=reward_type,
        reward_shaping=reward_shaping,
        state_representation=state_representation,
        price_offset=price_offset,
        reward_horizon=reward_horizon,
    )

    # 2. Load & generate Train Data (Days 1-9)
    print("Loading Train Data (Days 1-9)...")
    train_raw = np.loadtxt(train_file)
    X_train = train_raw[:40, :].T.astype(np.float32)
    y_train = train_raw[144:, :].T.astype(np.float32)

    _generate_per_stock(
        X_train, y_train, train_episodes, train_out, workers,
        split_label="Train", plot_dir=plot_dir, **gen_kwargs,
    )
    del train_raw, X_train, y_train

    # 3. Load & generate Test Data (Day 10)
    print("Loading Test Data (Day 10)...")
    test_raw = np.loadtxt(test_file)
    X_test = test_raw[:40, :].T.astype(np.float32)
    y_test = test_raw[144:, :].T.astype(np.float32)

    _generate_per_stock(
        X_test, y_test, test_episodes, test_out, workers,
        split_label="Test", plot_dir=plot_dir, **gen_kwargs,
    )
    del test_raw, X_test, y_test

    print(f"All processes completed successfully. Outputs saved at {train_out} and {test_out}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DT Trajectories Generator")
    parser.add_argument("--train_episodes", type=int, default=6000, help="Number of train trajectories")
    parser.add_argument("--test_episodes", type=int, default=1200, help="Number of test trajectories")
    parser.add_argument("--workers", type=int, default=32, help="CPU cores to use")
    parser.add_argument("--plot_dir", type=str, default="plots", help="Directory to save the visual distribution plots")
    args = parser.parse_args()

    # 1. Download Dataset via kagglehub
    print("Downloading/Locating FI-2010 dataset...")
    import kagglehub
    dataset_path = Path(kagglehub.dataset_download("ulfricirons/fi-2010"))
    
    # Locate Fold 9 files
    train_file = next(dataset_path.rglob("*NoAuction_Zscore*Training/Train*CF_9.txt"))
    test_file  = next(dataset_path.rglob("*NoAuction_Zscore*Testing/Test*CF_9.txt"))
    
    # 2. Load Train Data (Days 1-9)
    print("Loading Train Data (Days 1-9)...")
    train_raw = np.loadtxt(train_file)
    X_train = train_raw[:40, :].T.astype(np.float32)  # Raw LOB only
    y_train = train_raw[144:, :].T.astype(np.float32) # Labels
    
    # Generate Train Dataset
    train_trajectories = generate_dataset(
        X=X_train, 
        y=y_train, 
        num_episodes=args.train_episodes, 
        output_file="train_trajectories.pt", 
        num_workers=args.workers,
        desc="Train Dataset"
    )
    
    # Generate Train plots
    plot_distributions(train_trajectories, "Train", args.plot_dir)
    
    # Free up memory before loading test data
    del train_raw, X_train, y_train, train_trajectories
    
    # 3. Load Test Data (Day 10)
    print("Loading Test Data (Day 10)...")
    test_raw = np.loadtxt(test_file)
    X_test = test_raw[:40, :].T.astype(np.float32)
    y_test = test_raw[144:, :].T.astype(np.float32)
    
    # Generate Test Dataset
    test_trajectories = generate_dataset(
        X=X_test, 
        y=y_test, 
        num_episodes=args.test_episodes, 
        output_file="test_trajectories.pt", 
        num_workers=args.workers,
        desc="Test Dataset"
    )
    
    # Generate Test plots
    plot_distributions(test_trajectories, "Test", args.plot_dir)
    
    print("All processes completed successfully. You now have train_trajectories.pt and test_trajectories.pt.")