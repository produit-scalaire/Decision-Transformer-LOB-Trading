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
def plot_distributions(trajectories, dataset_name, save_dir="plots"):
    """
    Generates and saves the return distribution plots to provide visual feedback.
    """
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    policy_returns = {p: [] for p in POLICIES.keys()}
    all_rtg = []
    
    for t in trajectories:
        policy_returns[t["policy"]].append(t["total_return"])
        # Extract RTG tensors to a flat list for global distribution
        all_rtg.extend(t["rtg"].flatten().numpy())
        
    # 1. Episode returns distribution per policy
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, (policy, rets) in enumerate(policy_returns.items()):
        if rets:
            axes[i].hist(rets, bins=30, alpha=0.7, edgecolor='black', color='tab:blue')
            axes[i].set_title(f"Policy: {policy}")
            axes[i].axvline(np.mean(rets), color='red', linestyle='dashed', linewidth=1.5, label=f"Mean: {np.mean(rets):.2f}")
            axes[i].legend()
            
    plt.suptitle(f"Return Distributions by Policy ({dataset_name})", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{dataset_name.lower()}_policy_returns.png", dpi=150)
    plt.close()

    # 2. Global Return-to-Go distribution
    plt.figure(figsize=(10, 5))
    plt.hist(all_rtg, bins=100, alpha=0.75, color='tab:purple', edgecolor='black')
    plt.title(f"Global Return-to-Go Distribution ({dataset_name})", fontweight='bold')
    plt.xlabel("Return-to-Go")
    plt.ylabel("Frequency")
    plt.axvline(np.mean(all_rtg), color='red', linestyle='dashed', linewidth=2, label=f"Global Mean: {np.mean(all_rtg):.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{dataset_name.lower()}_rtg_distribution.png", dpi=150)
    plt.close()
    
    print(f"Visual plots for {dataset_name} have been generated in the '{save_dir}/' directory.")

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

    # Save to disk
    torch.save(all_trajectories, output_file)
    print(f"Saved {len(all_trajectories)} trajectories to {output_file}")
    
    # Print short summary
    returns = [t["total_return"] for t in all_trajectories]
    print(f"Metrics -> Mean Return: {np.mean(returns):.4f} | Max Return: {np.max(returns):.4f}\n")
    
    return all_trajectories

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
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
    
    # 2. Load Train Data (Days 1-9)
    print("Loading Train Data (Days 1-9)...")
    train_raw = np.loadtxt(train_file)
    X_train = train_raw[:40, :].T.astype(np.float32)  # Raw LOB only
    y_train = train_raw[144:, :].T.astype(np.float32) # Labels
    
    # Generate Train Dataset
    train_trajectories = generate_dataset(
        X=X_train,
        y=y_train,
        num_episodes=train_episodes,
        output_file=train_out,
        num_workers=workers,
        desc="Train Dataset",
        window_size=window_size,
        episode_length=episode_length,
        reward_type=reward_type,
        reward_shaping=reward_shaping,
        state_representation=state_representation,
        price_offset=price_offset,
        reward_horizon=reward_horizon,
    )
    
    # Generate Train plots
    plot_distributions(train_trajectories, "Train", plot_dir)
    
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
        num_episodes=test_episodes,
        output_file=test_out,
        num_workers=workers,
        desc="Test Dataset",
        window_size=window_size,
        episode_length=episode_length,
        reward_type=reward_type,
        reward_shaping=reward_shaping,
        state_representation=state_representation,
        price_offset=price_offset,
        reward_horizon=reward_horizon,
    )
    
    # Generate Test plots
    plot_distributions(test_trajectories, "Test", plot_dir)
    
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