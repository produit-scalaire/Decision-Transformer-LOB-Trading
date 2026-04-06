import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from src.models.decision_transformer import DecisionTransformer

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
    whitelist_modules = (nn.Linear,)
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
# 3. Training Loop
# -----------------------------------------------------------------------------

def train_model(train_data_path, model_dir, model_cfg, train_cfg, hardware_cfg):
    """Encapsulated entry point for Hydra orchestrator."""
    device = torch.device(hardware_cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"--- Hardware target: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ---")

    # 1. Load Dataset on CPU
    dataset = OptimizedTrajectoryDataset(
        data_path=train_data_path, 
        context_len=train_cfg.context_len
    )
    
    # 2. Optimized DataLoader Setup
    dataloader = DataLoader(
        dataset, 
        batch_size=train_cfg.batch_size, 
        shuffle=True, 
        num_workers=hardware_cfg.workers, 
        pin_memory=True, 
        prefetch_factor=2
    )

    # 3. Initialize Model
    model = DecisionTransformer(
        state_dim=model_cfg.state_dim, 
        act_dim=model_cfg.act_dim,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        max_timestep=model_cfg.max_timestep,
        dropout=model_cfg.dropout
    ).to(device)

    # 4. Compile Model (Triton Kernels for RTX 5090 Blackwell architecture)
    if hardware_cfg.compile_model:
        print("Compiling model via torch.compile (this takes ~1 min)...")
        model = torch.compile(model, mode="max-autotune")
    
    # 5. Optimizer & Scheduler
    optimizer = configure_optimizers(model, train_cfg.weight_decay, train_cfg.lr, (0.9, 0.95))
    total_steps = len(dataloader) * train_cfg.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    loss_fn = nn.CrossEntropyLoss()

    print("=" * 80)
    print(f"Starting Training: {train_cfg.epochs} Epochs | Batch: {train_cfg.batch_size} | K: {train_cfg.context_len}")
    print(f"Total gradient steps per epoch: {len(dataloader):,}")
    print("=" * 80)

    # 6. Training Loop
    for epoch in range(1, train_cfg.epochs + 1):
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

            batch_size_actual = actions.size(0)
            epoch_loss += loss.item() * batch_size_actual
            epoch_correct += (flat_logits.argmax(dim=-1) == flat_actions).sum().item()
            epoch_total += flat_actions.numel()

            if (step + 1) % 500 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch:02d} | Step {step+1:05d}/{len(dataloader)} | Loss: {loss.item():.4f} | LR: {current_lr:.2e}")

        # Epoch Profiling
        avg_loss = epoch_loss / len(dataset)
        avg_acc = epoch_correct / epoch_total
        epoch_time = time.time() - start_time
        samples_per_sec = len(dataset) / epoch_time
        
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