import os
import time
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from src.models.model_factory import build_model
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

plt.ioff()

# -----------------------------------------------------------------------------
# 1. Dataset Class (Optimized for Huge Datasets)
# -----------------------------------------------------------------------------

class OptimizedTrajectoryDataset(Dataset):
    """
    Optimized sliding-window dataset for millions of samples.
    - Keeps data in System RAM (CPU) to prevent GPU VRAM crashes.
    - Uses 1D PyTorch Tensors for indexing to avoid Python list memory leaks.
    """
    def __init__(self, data_path: str, context_len: int = 100):
        print(f"Loading trajectories from {data_path} into System RAM (CPU)...")
        start_time = time.time()
        
        # 1. Load data onto CPU (map_location='cpu' is critical here)
        trajectories = torch.load(data_path, map_location='cpu', weights_only=False)
        if isinstance(trajectories, dict):
            trajectories = list(trajectories.values())
            
        self.context_len = context_len
        self.states = []
        self.actions = []
        self.rtg = []
        
        # 2. Calculate the total number of valid windows to pre-allocate memory
        total_windows = 0
        for traj in trajectories:
            T = len(traj["rewards"])
            if T > context_len:
                total_windows += (T - context_len)
                
        # 3. Create 1D PyTorch Int32 arrays for indices instead of Python Lists.
        # A Python list of 94 million tuples uses > 5GB of System RAM.
        # Two 1D Int32 tensors use only ~750 MB and are infinitely faster to read.
        self.traj_indices = torch.zeros(total_windows, dtype=torch.int32)
        self.time_indices = torch.zeros(total_windows, dtype=torch.int32)
        
        current_idx = 0
        for i, traj in enumerate(trajectories):
            # Append tensors to our lists (still on CPU)
            self.states.append(traj["states"].to(torch.float32))
            self.actions.append(traj["actions"].to(torch.long))
            self.rtg.append(traj["rtg"].to(torch.float32))
            
            # Fill the index arrays
            T = len(traj["rewards"])
            num_windows = T - context_len
            if num_windows > 0:
                self.traj_indices[current_idx : current_idx + num_windows] = i
                self.time_indices[current_idx : current_idx + num_windows] = torch.arange(num_windows, dtype=torch.int32)
                current_idx += num_windows
                
        print(f"Dataset ready in {time.time() - start_time:.2f} seconds.")
        print(f"Total sliding windows created (K={context_len}): {total_windows:,}")
        print("Data is successfully stored in CPU RAM. It will be streamed to the GPU in the background.")

    def __len__(self):
        return len(self.traj_indices)

    def __getitem__(self, idx):
        # Read the indices from our fast 1D tensors
        traj_idx = self.traj_indices[idx].item()
        t = self.time_indices[idx].item()
        K = self.context_len
        
        # Slice the context window directly from CPU memory
        states = self.states[traj_idx][t : t + K]
        actions = self.actions[traj_idx][t : t + K]
        rtg = self.rtg[traj_idx][t : t + K]
        
        # Generate timesteps
        timesteps = torch.arange(t, t + K, dtype=torch.long)
        
        # Return the slices (they are still on CPU at this point)
        return states, actions, rtg, timesteps

# -----------------------------------------------------------------------------
# 2. Optimizer Configuration
# -----------------------------------------------------------------------------

def configure_optimizers(model: nn.Module, weight_decay: float, learning_rate: float, betas: tuple):
    """
    Separates weights requiring Weight Decay (Linear layers) from those 
    that don't (Biases, LayerNorms, Embeddings) following GPT design.
    """
    decay, no_decay = set(), set()
    # Conv1d weights should also decay (same reasoning as Linear weights).
    whitelist_modules = (nn.Linear, nn.Conv1d)
    blacklist_modules = (nn.LayerNorm, nn.Embedding)
    
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f'{mn}.{pn}' if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_modules):
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    
    # Fused AdamW avoids multiple GPU memory reads/writes, highly recommended for RTX 5090
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=True)

# -----------------------------------------------------------------------------
# 3. Training Curve Plotting
# -----------------------------------------------------------------------------

def plot_training_curves(history: dict, save_path: "str | Path"):
    """Save a 3-panel figure of Loss, Accuracy and Learning Rate over training.

    Parameters
    ----------
    history   : dict with keys:
                  ``step_loss``  — list of (global_step, loss) pairs
                  ``step_lr``    — list of (global_step, lr) pairs
                  ``epoch_acc``  — list of (epoch, accuracy) pairs
    save_path : destination PNG path.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # --- Loss ---
    if history.get("step_loss"):
        steps, losses = zip(*history["step_loss"])
        axes[0].plot(steps, losses, linewidth=1.0, color="tab:blue", alpha=0.85)
        if history.get("epoch_loss"):
            ep_steps, ep_losses = zip(*history["epoch_loss"])
            axes[0].plot(
                ep_steps, ep_losses, "o-", color="navy", linewidth=1.8,
                markersize=5, label="Epoch avg"
            )
            axes[0].legend(fontsize=8)
    axes[0].set_title("Training Loss", fontweight="bold")
    axes[0].set_xlabel("Gradient Step")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, alpha=0.3)

    # --- Accuracy ---
    if history.get("epoch_acc"):
        epochs, accs = zip(*history["epoch_acc"])
        axes[1].plot(epochs, [a * 100 for a in accs], "o-", color="tab:green",
                     linewidth=1.8, markersize=6)
    axes[1].set_title("Training Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3)

    # --- Learning Rate ---
    if history.get("step_lr"):
        steps, lrs = zip(*history["step_lr"])
        axes[2].plot(steps, lrs, linewidth=1.0, color="tab:orange")
    axes[2].set_title("Learning Rate Schedule", fontweight="bold")
    axes[2].set_xlabel("Gradient Step")
    axes[2].set_ylabel("Learning Rate")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Training Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to '{save_path}'.")


# -----------------------------------------------------------------------------
# 4. Training Loop
# -----------------------------------------------------------------------------

def train_model(
    train_data_path, model_dir, model_cfg, train_cfg, hardware_cfg,
    plot_dir: "str | None" = None
):
    """Encapsulated entry point for Hydra orchestrator.

    Parameters
    ----------
    plot_dir : Optional directory to save the training-curves PNG.
               Falls back to ``model_dir`` when *None*.
    """
    device = torch.device(hardware_cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"--- Hardware target: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ---")

    # Hydra resolves YAML numerics; plain OmegaConf/yaml loads may leave 1e-4 as str.
    lr = float(train_cfg.lr)
    weight_decay = float(train_cfg.weight_decay)
    context_len = int(train_cfg.context_len)
    batch_size = int(train_cfg.batch_size)
    n_epochs = int(train_cfg.epochs)

    # 1. Load Dataset on CPU
    dataset = OptimizedTrajectoryDataset(
        data_path=train_data_path,
        context_len=context_len,
    )
    
    # 2. Optimized DataLoader Setup
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(hardware_cfg.workers),
        pin_memory=True,
        prefetch_factor=2,
    )

    # 3. Build the model selected in config (transformer or cnn).
    model = build_model(model_cfg).to(device)

    # 4. Compile Model (Triton Kernels for RTX 5090 Blackwell architecture)
    if hardware_cfg.compile_model:
        print("Compiling model via torch.compile (this takes ~1 min)...")
        model = torch.compile(model, mode="max-autotune")
    
    # 5. Optimizer & Scheduler
    optimizer = configure_optimizers(model, weight_decay, lr, (0.9, 0.95))
    total_steps = len(dataloader) * n_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    loss_fn = nn.CrossEntropyLoss()

    print("=" * 80)
    print(f"Starting Training: {n_epochs} Epochs | Batch: {batch_size} | K: {context_len}")
    print(f"Total gradient steps per epoch: {len(dataloader):,}")
    print("=" * 80)

    # 6. Training history buffers (for learning-curve plots)
    history: dict[str, list] = {
        "step_loss":  [],   # (global_step, loss)
        "step_lr":    [],   # (global_step, lr)
        "epoch_loss": [],   # (global_step_at_epoch_end, avg_loss)
        "epoch_acc":  [],   # (epoch, accuracy)
    }
    global_step = 0
    log_every = max(1, len(dataloader) // 10)   # ~10 loss points per epoch

    # 7. Training Loop
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        start_time = time.time()

        for step, (states, actions, rtg, timesteps) in enumerate(dataloader):

            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            rtg = rtg.to(device, non_blocking=True)
            timesteps = timesteps.to(device, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(states, actions, rtg, timesteps)

                flat_logits = logits.reshape(-1, logits.size(-1))
                flat_actions = actions.reshape(-1)

                loss = loss_fn(flat_logits, flat_actions)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            global_step += 1

            batch_size_actual = actions.size(0)
            epoch_loss += loss.item() * batch_size_actual
            epoch_correct += (flat_logits.argmax(dim=-1) == flat_actions).sum().item()
            epoch_total += flat_actions.numel()

            # Record step-level metrics at regular intervals
            if global_step % log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                history["step_loss"].append((global_step, loss.item()))
                history["step_lr"].append((global_step, current_lr))

            if (step + 1) % 500 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch:02d} | Step {step+1:05d}/{len(dataloader)} | Loss: {loss.item():.4f} | LR: {current_lr:.2e}")

        # Epoch Profiling
        avg_loss = epoch_loss / len(dataset)
        avg_acc = epoch_correct / epoch_total
        epoch_time = time.time() - start_time
        samples_per_sec = len(dataset) / epoch_time

        history["epoch_loss"].append((global_step, avg_loss))
        history["epoch_acc"].append((epoch, avg_acc))

        print("-" * 80)
        print(f"Epoch {epoch:02d} Completed in {epoch_time:.1f}s | Throughput: {samples_per_sec:,.0f} windows/s")
        print(f"Average Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.2%}")
        print("-" * 80)

        # Save Checkpoint
        checkpoint_path = f"{model_dir}/dt_model_ep{epoch:02d}.pt"
        state_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'acc': avg_acc,
        }
        torch.save(state_dict, checkpoint_path)
        # Always save the latest as final to easily load it during evaluation
        torch.save(state_dict, f"{model_dir}/dt_model_final.pt")

    # 8. Save training curves
    curves_dir = Path(plot_dir) if plot_dir else Path(model_dir)
    curves_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history, curves_dir / "training_curves.png")


def train(args):
    # This is kept for backward compatibility if you run the script directly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description="Optimized DT Training for RTX 5090")
    parser.add_argument("--data_path", type=str, default="train_trajectories.pt", help="Path to generated trajectories")
    parser.add_argument("--epochs", type=int, default=15)
    
    # DEFAULT BATCH SIZE MASSIVELY INCREASED
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size (Huge batches for RTX 5090)")
    
    parser.add_argument("--context_len", type=int, default=100, help="Sliding window size (K)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    
    args = parser.parse_args()
    
    train(args)