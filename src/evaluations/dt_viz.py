import os
import time
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.models.decision_transformer import DecisionTransformer

# Disable interactive plotting to generate image files safely in the background
plt.ioff()

# =============================================================================
# 1. Financial Evaluation Metrics
# =============================================================================

def compute_financial_metrics(rewards_tensor: torch.Tensor) -> dict:
    """
    Calculates key financial metrics from a batch of simulated reward trajectories.
    - rewards_tensor: PyTorch tensor of shape (Batch_size, Timesteps)
    """
    # 1. Cumulative PnL (Profit and Loss) over the entire trajectory
    cum_pnl = torch.cumsum(rewards_tensor, dim=1)
    final_pnl = cum_pnl[:, -1].mean().item()

    # 2. Sharpe Ratio (Average Return relative to volatility/risk)
    mean_return = rewards_tensor.mean(dim=1)
    std_return = rewards_tensor.std(dim=1) + 1e-8 # Add epsilon to prevent division by zero
    sharpe = (mean_return / std_return).mean().item()
    
    # 3. Maximum Drawdown (Largest peak-to-trough drop in value)
    running_max = torch.cummax(cum_pnl, dim=1).values
    drawdown = running_max - cum_pnl
    max_dd = drawdown.max(dim=1).values.mean().item()

    return {
        "PnL": final_pnl,
        "Sharpe": sharpe,
        "MaxDD": max_dd
    }

def get_market_returns(states_batch: torch.Tensor) -> torch.Tensor:
    """
    Extracts the step-by-step market returns directly from the LOB states.
    Assumes standard FI-2010 structure where:
      - states_batch[:, :, 0] is Ask Price 1
      - states_batch[:, :, 2] is Bid Price 1
    Even if z-score normalized, the relative price differences provide an accurate 
    basis for statistical PnL comparison across policies.
    """
    ask1 = states_batch[:, :, 0]
    bid1 = states_batch[:, :, 2]
    mid_price = (ask1 + bid1) / 2.0
    
    returns = torch.zeros_like(mid_price)
    # Market return at step t is the price change from t to t+1
    returns[:, :-1] = mid_price[:, 1:] - mid_price[:, :-1]
    return returns

# =============================================================================
# 2. Autoregressive Rollout & Baselines
# =============================================================================

def vectorized_autoregressive_rollout(model, states, market_returns, target_rtg, context_len, device):
    """
    Performs a step-by-step autoregressive rollout for the Decision Transformer.
    Re-builds context iteratively to prevent looking into the future.
    """
    B, T, feature_dim = states.shape
    
    # Initialize rolling context buffers
    ctx_states = torch.zeros((B, context_len, feature_dim), device=device)
    ctx_actions = torch.zeros((B, context_len), dtype=torch.long, device=device)
    ctx_rtg = torch.zeros((B, context_len, 1), device=device)
    ctx_timesteps = torch.zeros((B, context_len), dtype=torch.long, device=device)
    
    predicted_positions = torch.zeros((B, T), device=device)
    realized_rewards = torch.zeros((B, T), device=device)
    
    current_rtg = torch.full((B, 1), target_rtg, device=device)
    
    model.eval()
    with torch.no_grad():
        for t in range(T):
            # 1. Shift context windows left (discard oldest, make room at end)
            ctx_states = torch.roll(ctx_states, shifts=-1, dims=1)
            ctx_actions = torch.roll(ctx_actions, shifts=-1, dims=1)
            ctx_rtg = torch.roll(ctx_rtg, shifts=-1, dims=1)
            ctx_timesteps = torch.roll(ctx_timesteps, shifts=-1, dims=1)
            
            # 2. Inject current step observations
            ctx_states[:, -1, :] = states[:, t, :]
            ctx_rtg[:, -1, 0] = current_rtg.squeeze(-1)
            ctx_timesteps[:, -1] = t
            
            # Zero out the action slot for the current timestep so the model predicts it
            ctx_actions[:, -1] = 0
            
            # 3. Predict action
            # model uses torch.bfloat16 implicitly via autocast if configured
            action_preds = model(ctx_states, ctx_actions, ctx_rtg, ctx_timesteps)
            
            # We only care about the prediction for the current timestep (last token)
            last_pred = action_preds[:, -1, :]  # Shape: (B, act_dim)
            action = torch.argmax(last_pred, dim=-1)  # Shape: (B,)
            
            # 4. Map action to position (must match LOBTradingEnv.action_to_position)
            #    0 -> short (-1), 1 -> flat (0), 2 -> long (+1)
            pos = action.float() - 1.0
            
            # 5. Evaluate Step Reward using the actual market return
            step_reward = pos * market_returns[:, t]
            
            # Save stats
            predicted_positions[:, t] = pos
            realized_rewards[:, t] = step_reward
            
            # 6. Update context for next iteration
            ctx_actions[:, -1] = action
            current_rtg = current_rtg - step_reward.unsqueeze(-1)
            
    return realized_rewards, predicted_positions

def evaluate_baselines(market_returns: torch.Tensor):
    """
    Evaluates classic quantitative baseline policies on the testing data.
    - Buy & Hold: Always Long
    - Momentum: Long if prev return > 0, Short if < 0
    - Mean Reversion: Short if prev return > 0, Long if < 0
    - Oracle (Perfect Foresight): Long if future return > 0, Short if < 0
    """
    metrics_dict = {}
    trajectories_dict = {}

    # 1. Buy & Hold (Pos = 1)
    bnh_pnl = market_returns.cumsum(dim=1)
    trajectories_dict['Buy & Hold'] = bnh_pnl.mean(dim=0).cpu().numpy()
    metrics_dict['Buy & Hold'] = compute_financial_metrics(market_returns)

    # 2. Oracle (Pos = sign(market_returns))
    oracle_pos = torch.where(market_returns > 0, 1.0, torch.where(market_returns < 0, -1.0, 0.0))
    oracle_rewards = oracle_pos * market_returns
    oracle_pnl = oracle_rewards.cumsum(dim=1)
    trajectories_dict['Oracle'] = oracle_pnl.mean(dim=0).cpu().numpy()
    metrics_dict['Oracle'] = compute_financial_metrics(oracle_rewards)

    # 3. Momentum (Pos = sign of prev return)
    mom_pos = torch.zeros_like(market_returns)
    mom_pos[:, 1:] = torch.sign(market_returns[:, :-1])
    mom_rewards = mom_pos * market_returns
    mom_pnl = mom_rewards.cumsum(dim=1)
    trajectories_dict['Momentum'] = mom_pnl.mean(dim=0).cpu().numpy()
    metrics_dict['Momentum'] = compute_financial_metrics(mom_rewards)

    # 4. Mean Reversion (Pos = -sign of prev return)
    mr_pos = torch.zeros_like(market_returns)
    mr_pos[:, 1:] = -torch.sign(market_returns[:, :-1])
    mr_rewards = mr_pos * market_returns
    mr_pnl = mr_rewards.cumsum(dim=1)
    trajectories_dict['Mean Reversion'] = mr_pnl.mean(dim=0).cpu().numpy()
    metrics_dict['Mean Reversion'] = compute_financial_metrics(mr_rewards)

    return trajectories_dict, metrics_dict

# =============================================================================
# 3. Visualization Generators
# =============================================================================

def plot_sharpe_comparison(df_metrics: pd.DataFrame, save_path: Path):
    """
    Generates a horizontal bar chart comparing the Sharpe ratio of all policies.
    Mimics the visual representation from the Jupyter notebook.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Sort by Sharpe for better visual hierarchy
    df_sorted = df_metrics.sort_values(by="Sharpe", ascending=True)
    
    agents = df_sorted.index.tolist()
    sharpes = df_sorted["Sharpe"].values
    colors = ["tab:green" if s > 0 else "tab:red" for s in sharpes]
    
    ax.barh(agents, sharpes, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    
    ax.set_xlabel("Sharpe Ratio (Unannualized)")
    ax.set_title("Risk-Adjusted Performance Comparison")
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_pnl_curves(trajectories_dict: dict, save_path: Path):
    """
    Plots the cumulative PnL progression over time for DT policies vs Baselines.
    Highlights Oracle and the DT models specifically.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for agent, traj in trajectories_dict.items():
        if agent == 'Oracle':
            ax.plot(traj, label=agent, linestyle='--', color='gold', alpha=0.8)
        elif 'DT' in agent:
            # Highlight DT policies with thicker lines
            ax.plot(traj, label=agent, linewidth=2.5)
        else:
            ax.plot(traj, label=agent, alpha=0.6)
            
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Cumulative PnL Comparison: Decision Transformer vs Baselines")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Cumulative PnL")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# =============================================================================
# Main Execution Pipeline
# =============================================================================
def evaluate_model(model_path, data_path, eval_cfg, model_cfg, plot_dir):
    """Encapsulated entry point for Hydra orchestrator."""
    # Hardware config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing evaluations on: {device}")

    # Ensure output directory
    out_path = Path(plot_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load the Decision Transformer
    print(f"\nLoading model from {model_path}...")
    model = DecisionTransformer(
        state_dim=model_cfg.state_dim,
        act_dim=model_cfg.act_dim,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        max_timestep=model_cfg.max_timestep
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    raw_state_dict = checkpoint['model_state_dict']
    
    # [PATCH] Remove '_orig_mod.' prefix added by torch.compile() during training
    # This allows loading a compiled checkpoint into a vanilla PyTorch module
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in raw_state_dict.items()}
    
    model.load_state_dict(clean_state_dict)
    
    # 2. Load Testing Dataset (Memory efficient partial load)
    print(f"Loading testing data from {data_path}...")
    trajectories = torch.load(data_path, map_location='cpu', weights_only=False)
    if isinstance(trajectories, dict):
        trajectories = list(trajectories.values())
        
    trajectories = trajectories[:eval_cfg.max_eval_trajectories]
    
    # Find minimum trajectory length to batch them neatly
    min_len = min(len(traj["states"]) for traj in trajectories)
    
    # Ensure correct dim matching the model (e.g. 41)
    states_batch = torch.stack([
        torch.tensor(traj["states"][:min_len], dtype=torch.float32) for traj in trajectories
    ])[:, :, :model_cfg.state_dim].to(device)

    print(f"Batch constructed: Shape {states_batch.shape}")
    
    # 3. Extract Market Environment Truths
    market_returns = get_market_returns(states_batch)

    # Master dictionaries to aggregate results for dataframe & plots
    all_trajectories = {}
    all_metrics = {}

    # 4. Evaluate Mathematical Baselines
    print("\n--- Evaluating Baselines ---")
    base_trajs, base_metrics = evaluate_baselines(market_returns)
    all_trajectories.update(base_trajs)
    all_metrics.update(base_metrics)

    # 5. Evaluate Decision Transformer across Target RTGs
    print("\n--- Evaluating Decision Transformer ---")
    for rtg in eval_cfg.target_rtgs:
        print(f"  [>] Rolling out DT with Target RTG = {rtg}...")
        
        realized_rewards, predicted_pos = vectorized_autoregressive_rollout(
            model=model,
            states=states_batch,
            market_returns=market_returns,
            target_rtg=rtg,
            context_len=eval_cfg.context_len,
            device=device
        )
        
        agent_name = f"DT (RTG={rtg})"
        dt_pnl = realized_rewards.cumsum(dim=1)
        
        all_trajectories[agent_name] = dt_pnl.mean(dim=0).cpu().numpy()
        all_metrics[agent_name] = compute_financial_metrics(realized_rewards)

    # 6. Consolidate Results & Print Table
    df_metrics = pd.DataFrame.from_dict(all_metrics, orient='index')
    
    print("\n" + "=" * 65)
    print("BACKTEST SUMMARY (Test Set)")
    print("=" * 65)
    for _, row in df_metrics.iterrows():
        print(f"  {row.name:<20} | PnL: {row['PnL']:+.6f} | Sharpe: {row['Sharpe']:+.4f} | MaxDD: {row['MaxDD']:.6f}")
    print("=" * 65)

    # 7. Generate Enhanced Visualizations
    print("\nGenerating comparative visualization plots...")
    plot_pnl_curves(all_trajectories, out_path / "pnl_comparison_curves.png")
    plot_sharpe_comparison(df_metrics, out_path / "sharpe_ratio_bars.png")
    
    print(f"Success. Visualizations saved to '{out_path.absolute()}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DT Offline Evaluation & Visualizations")
    parser.add_argument("--model_path", type=str, default="dt_model_ep15.pt", help="Path to trained DT model checkpoint")
    parser.add_argument("--data_path", type=str, default="test_trajectories.pt", help="Path to testing dataset")
    parser.add_argument("--target_rtgs", type=float, nargs='+', default=[0.5, 1.0, 2.0], help="List of RTG targets to test")
    parser.add_argument("--context_len", type=int, default=100, help="Autoregressive sliding window size")
    parser.add_argument("--max_eval_trajectories", type=int, default=32, help="Number of test batch trajectories to evaluate")
    parser.add_argument("--out_dir", type=str, default="visualizations", help="Output directory for plots")
    args = parser.parse_args()

    # Hardware config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing evaluations on: {device}")

    # Ensure output directory
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load the Decision Transformer
    print(f"\nLoading model from {args.model_path}...")
    model = DecisionTransformer(
        state_dim=40,
        act_dim=3,
        d_model=128,  # Match training architecture
        n_heads=4,
        n_layers=3,
        max_ep_len=10000
    ).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    raw_state_dict = checkpoint['model_state_dict']
    
    # [PATCH] Remove '_orig_mod.' prefix added by torch.compile() during training
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in raw_state_dict.items()}
    
    model.load_state_dict(clean_state_dict)
    
    # 2. Load Testing Dataset (Memory efficient partial load)
    print(f"Loading testing data from {args.data_path}...")
    trajectories = torch.load(args.data_path, map_location='cpu', weights_only=False)
    if isinstance(trajectories, dict):
        trajectories = list(trajectories.values())
        
    trajectories = trajectories[:args.max_eval_trajectories]
    
    # Find minimum trajectory length to batch them neatly
    min_len = min(len(traj["states"]) for traj in trajectories)
    
    # Strip behavioral position if present in raw data (ensure dim is 40)
    states_batch = torch.stack([
        torch.tensor(traj["states"][:min_len], dtype=torch.float32) for traj in trajectories
    ])[:, :, :40].to(device)

    print(f"Batch constructed: Shape {states_batch.shape}")
    
    # 3. Extract Market Environment Truths
    market_returns = get_market_returns(states_batch)

    # Master dictionaries to aggregate results for dataframe & plots
    all_trajectories = {}
    all_metrics = {}

    # 4. Evaluate Mathematical Baselines
    print("\n--- Evaluating Baselines ---")
    base_trajs, base_metrics = evaluate_baselines(market_returns)
    all_trajectories.update(base_trajs)
    all_metrics.update(base_metrics)

    # 5. Evaluate Decision Transformer across Target RTGs
    print("\n--- Evaluating Decision Transformer ---")
    for rtg in args.target_rtgs:
        print(f"  [>] Rolling out DT with Target RTG = {rtg}...")
        
        realized_rewards, predicted_pos = vectorized_autoregressive_rollout(
            model=model,
            states=states_batch,
            market_returns=market_returns,
            target_rtg=rtg,
            context_len=args.context_len,
            device=device
        )
        
        agent_name = f"DT (RTG={rtg})"
        dt_pnl = realized_rewards.cumsum(dim=1)
        
        all_trajectories[agent_name] = dt_pnl.mean(dim=0).cpu().numpy()
        all_metrics[agent_name] = compute_financial_metrics(realized_rewards)

    # 6. Consolidate Results & Print Table
    df_metrics = pd.DataFrame.from_dict(all_metrics, orient='index')
    
    print("\n" + "=" * 65)
    print("BACKTEST SUMMARY (Test Set)")
    print("=" * 65)
    for _, row in df_metrics.iterrows():
        print(f"  {row.name:<20} | PnL: {row['PnL']:+.6f} | Sharpe: {row['Sharpe']:+.4f} | MaxDD: {row['MaxDD']:.6f}")
    print("=" * 65)

    # 7. Generate Enhanced Visualizations
    print("\nGenerating comparative visualization plots...")
    plot_pnl_curves(all_trajectories, out_path / "pnl_comparison_curves.png")
    plot_sharpe_comparison(df_metrics, out_path / "sharpe_ratio_bars.png")
    
    print(f"Success. Visualizations saved to '{out_path.absolute()}'.")