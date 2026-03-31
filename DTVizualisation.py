import os
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from DecisionTransformer import DecisionTransformer

# -----------------------------------------------------------------------------
# 1. Financial Evaluation Metrics
# -----------------------------------------------------------------------------

def compute_financial_metrics(rewards_tensor: torch.Tensor) -> dict:
    """
    Computes rigorous financial metrics on a batch of simulated reward trajectories.
    rewards_tensor shape: (Batch, Timesteps)
    """
    # 1. Cumulative PnL
    cum_pnl = torch.cumsum(rewards_tensor, dim=1)
    final_pnl = cum_pnl[:, -1].mean().item()

    # 2. Sharpe Ratio (cross-sectional mean across the batch)
    mean_return = rewards_tensor.mean(dim=1)
    std_return = rewards_tensor.std(dim=1) + 1e-8
    sharpe = (mean_return / std_return).mean().item()
    
    # 3. Maximum Drawdown (MaxDD)
    # Equation: MaxDD_T = max_{tau} ( max_{t <= tau} PnL_t - PnL_tau )
    running_max = torch.cummax(cum_pnl, dim=1).values
    drawdown = running_max - cum_pnl
    max_dd = drawdown.max(dim=1).values.mean().item()

    return {
        "cumulative_pnl": final_pnl,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "mean_step_reward": mean_return.mean().item()
    }

# -----------------------------------------------------------------------------
# 2. Offline Evaluation (Action Distributions & Confusion Matrix)
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_offline_metrics(model, dataloader, device):
    """
    Teacher-forcing evaluation to extract confusion matrix and action distributions.
    O(1) temporal complexity per batch (no autoregressive loop).
    """
    model.eval()
    all_preds = []
    all_targets = []

    start_t = time.time()
    for states, actions, rtg, timesteps in dataloader:
        states = states.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        rtg = rtg.to(device, non_blocking=True)
        timesteps = timesteps.to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(states, actions, rtg, timesteps)
            preds = torch.argmax(logits, dim=-1)

        # We only care about valid predictions (ignore padding if any)
        all_preds.append(preds.cpu().flatten())
        all_targets.append(actions.cpu().flatten())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
    
    # Compute discrete probability distributions
    target_dist = np.bincount(all_targets, minlength=3) / len(all_targets)
    pred_dist = np.bincount(all_preds, minlength=3) / len(all_preds)

    print(f"Offline metrics computed in {time.time() - start_t:.2f}s")
    return cm, target_dist, pred_dist

# -----------------------------------------------------------------------------
# 3. Online Autoregressive Rollout (Vectorized LOB Simulation)
# -----------------------------------------------------------------------------

@torch.no_grad()
def vectorized_autoregressive_rollout(model, test_states: torch.Tensor, target_rtg: float, context_len: int, device: torch.device):
    """
    Highly optimized parallel generation. Simulates the LOB environment natively on GPU.
    test_states shape: (B, T_max, 40) -> LOB features without the position.
    """
    model.eval()
    B, T_max, _ = test_states.shape
    
    # Pre-allocate trajectory tensors on VRAM
    states = torch.zeros((B, T_max, 41), dtype=torch.float32, device=device)
    actions = torch.ones((B, T_max), dtype=torch.long, device=device) * 1 # Init flat
    rtg = torch.zeros((B, T_max, 1), dtype=torch.float32, device=device)
    timesteps = torch.arange(T_max, dtype=torch.long, device=device).unsqueeze(0).expand(B, T_max)
    rewards = torch.zeros((B, T_max), dtype=torch.float32, device=device)

    # Initial state conditions
    states[:, 0, :40] = test_states[:, 0, :40]
    states[:, 0, 40] = 0.0 # Initial position: flat
    rtg[:, 0, 0] = target_rtg
    
    start_t = time.time()
    
    # Strict causal autoregressive loop
    for t in range(T_max - 1):
        # Context slicing (Sliding Window attention optimization)
        start_idx = max(0, t + 1 - context_len)
        ctx_states = states[:, start_idx:t+1, :]
        ctx_actions = actions[:, start_idx:t+1]
        ctx_rtg = rtg[:, start_idx:t+1, :]
        ctx_timesteps = timesteps[:, start_idx:t+1]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(ctx_states, ctx_actions, ctx_rtg, ctx_timesteps)
            # Take the prediction for the very last timestep in the context
            next_action_logits = logits[:, -1, :] 
            next_action = torch.argmax(next_action_logits, dim=-1)

        actions[:, t] = next_action
        
        # --- Environment Dynamics Vectorized Simulation ---
        current_pos = states[:, t, 40]
        next_pos = next_action.float() - 1.0 # Map [0, 1, 2] -> [-1.0, 0.0, 1.0]
        
        # Extract mid prices: (ask_price + bid_price) / 2
        # Index 0 is ask_price_1, Index 2 is bid_price_1
        mid_now = (test_states[:, t, 0] + test_states[:, t, 2]) / 2.0
        mid_next = (test_states[:, t+1, 0] + test_states[:, t+1, 2]) / 2.0
        
        # Step PnL computation
        step_pnl = next_pos * (mid_next - mid_now)
        
        # Strict transaction cost penalty mapping using boolean masks
        transaction_mask = (next_pos != current_pos).float()
        step_reward = step_pnl - (0.005 * transaction_mask)
        
        rewards[:, t] = step_reward
        
        # State & RTG Update for t+1
        states[:, t+1, :40] = test_states[:, t+1, :40]
        states[:, t+1, 40] = next_pos
        rtg[:, t+1, 0] = rtg[:, t, 0] - step_reward

    print(f"Vectorized rollout of {B} trajectories ({T_max} steps) completed in {time.time() - start_t:.2f}s")
    return compute_financial_metrics(rewards)

# -----------------------------------------------------------------------------
# 4. Orchestration & Visualization
# -----------------------------------------------------------------------------

def plot_action_distribution(train_dist, test_dist, save_path="action_dist.png"):
    labels = ['Short (0)', 'Flat (1)', 'Long (2)']
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, train_dist, width, label='Train Dataset (Behavior)')
    ax.bar(x + width/2, test_dist, width, label='Test Output (Policy)')

    ax.set_ylabel('Empirical Probability')
    ax.set_title('Action Distribution: Behavioral Support vs Learned Policy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Action distribution plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_weights", type=str, required=True, help="Path to .pt model weights")
    parser.add_argument("--test_data", type=str, required=True, help="Path to offline_dataset.pt for testing")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--context_len", type=int, default=1024)
    parser.add_argument("--target_rtg", type=float, default=2.0, help="Target optimal Return-to-Go condition")
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Initialization
    model = DecisionTransformer(
        state_dim=41, 
        act_dim=3,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers
    ).to(device)
    
    checkpoint = torch.load(args.model_weights, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Inductor compilation for Blackwell Tensor Cores
    model = torch.compile(model)
    
    # 2. Data Loading (Taking a subset for fast eval)
    print(f"Loading test data from {args.test_data}...")
    trajectories = torch.load(args.test_data, weights_only=False)
    
    # Create simple dummy dataloader equivalent for offline metrics
    test_batch = [
        torch.stack([t['states'] for t in trajectories[:256]]),
        torch.stack([t['actions'] for t in trajectories[:256]]),
        torch.stack([t['rtg'] for t in trajectories[:256]]),
        torch.stack([t['timesteps'] for t in trajectories[:256]])
    ]
    
    # 3. Offline Metrics (Confusion & Distribution)
    print("\n" + "="*50 + "\n[1] OFFLINE METRICS\n" + "="*50)
    cm, target_dist, pred_dist = evaluate_offline_metrics(model, [test_batch], device)
    
    print("Confusion Matrix (Rows: True, Cols: Pred):")
    print(cm)
    print("\nAction Distribution (Train vs Test):")
    for i, a in enumerate(["Short", "Flat", "Long "]):
        print(f"  {a} -> True: {target_dist[i]*100:.1f}% | Pred: {pred_dist[i]*100:.1f}%")
        
    plot_action_distribution(target_dist, pred_dist)
    
    # 4. Online Vectorized Rollout
    print("\n" + "="*50 + "\n[2] ONLINE METRICS (VECTORIZED ROLLOUT)\n" + "="*50)
    test_states = test_batch[0][:, :, :40].to(device) # Strip behavioral position
    
    metrics = vectorized_autoregressive_rollout(
        model=model,
        test_states=test_states,
        target_rtg=args.target_rtg,
        context_len=args.context_len,
        device=device
    )
    
    print("\nFINAL FINANCIAL SUMMARY:")
    print(f"  Cumulative PnL   : {metrics['cumulative_pnl']:+.6f}")
    print(f"  Sharpe Ratio     : {metrics['sharpe_ratio']:+.4f} (Unannualized)")
    print(f"  Max Drawdown     : {metrics['max_drawdown']:.6f}")

if __name__ == "__main__":
    main()