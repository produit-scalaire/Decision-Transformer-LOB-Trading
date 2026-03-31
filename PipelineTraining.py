import os
import math
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from DecisionTransformer import DecisionTransformer

# -----------------------------------------------------------------------------
# 1. Dataset Class (Zero-copy GPU loading strategy)
# -----------------------------------------------------------------------------

class OfflineRLDataset(Dataset):
    """
    Dataset optimisé pour stocker les trajectoires en RAM (64Go disponibles)
    et les distribuer au Dataloader.
    """
    def __init__(self, data_path: str):
        print(f"Loading trajectories from {data_path}...")
        start = time.time()
        self.trajectories = torch.load(data_path, weights_only=False)
        print(f"Loaded {len(self.trajectories)} trajectories in {time.time() - start:.2f}s.")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        # Cast to contiguous memory blocks for strict PCIe transfer
        return (
            traj['states'].float(),   # (K, state_dim) -> cast to float32 for safety before amp
            traj['actions'].long(),   # (K,)
            traj['rtg'].float(),      # (K, 1)
            traj['timesteps'].long()  # (K,)
        )

# -----------------------------------------------------------------------------
# 2. Optimizer & Scheduler Configuration
# -----------------------------------------------------------------------------

def configure_optimizers(model: torch.nn.Module, weight_decay: float, learning_rate: float, betas: tuple):
    """
    Sépare les poids nécessitant un Weight Decay (matrices linéaires) 
    de ceux ne le nécessitant pas (biais, LayerNorm, Embeddings 1D).
    Règle canonique introduite par Karpathy pour les architectures GPT/Transformer.
    """
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

    # Validation stricte
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter = decay & no_decay
    union = decay | no_decay
    assert len(inter) == 0, "Erreur topologique : paramètres intersectés."
    assert len(param_dict.keys() - union) == 0, "Erreur topologique : paramètres non assignés."

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer

# -----------------------------------------------------------------------------
# 3. Training Loop (Ultra-Optimized for Blackwell RTX 5090)
# -----------------------------------------------------------------------------

def train_decision_transformer(
    data_path: str,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    warmup_steps: int = 1000,
    seq_len: int = 1024,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 6
):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', "GPU CUDA est requis (RTX 5090 attendue)."
    print(f"Hardware target: {torch.cuda.get_device_name(0)}")

    # 1. Dataset & DataLoader (CPU -> PCIe -> GPU async pipeline)
    dataset = OfflineRLDataset(data_path)
    
    # cpu_count() returns logical cores. Use max 16 to avoid saturating I/O overhead.
    num_workers = min(16, os.cpu_count() or 8) 
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,  # Crucial for fast Host-to-Device transfer
        drop_last=True
    )

    # 2. Model Initialization
    # Dimension état = 41 (40 features + 1 position LOB)
    model = DecisionTransformer(
        state_dim=41, 
        act_dim=3,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_timestep=10000, # Large bound to prevent embedding overflow
    ).to(device)

    # JIT Compilation via TorchInductor. Generates specific Triton kernels for RTX 5090
    print("Compiling model via torch.compile (this may take 1-2 minutes)...")
    model = torch.compile(model)
    print("Model compiled.")

    # 3. Optimizer Configuration
    optimizer = configure_optimizers(model, weight_decay, lr, (0.9, 0.95))
    
    # 4. Learning Rate Scheduler (Cosine with Warmup)
    total_steps = len(dataloader) * epochs
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)) # Min LR = 10% of base LR
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 5. Training Epochs
    print("=" * 60)
    print("Starting Training Loop")
    print(f"Total Epochs: {epochs} | Steps per Epoch: {len(dataloader)}")
    print(f"Batch Size: {batch_size} | Base LR: {lr}")
    print("=" * 60)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for step, (states, actions, rtg, timesteps) in enumerate(dataloader):
            # Asynchronous non-blocking transfer to VRAM
            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            rtg = rtg.to(device, non_blocking=True)
            timesteps = timesteps.to(device, non_blocking=True)

            # --- Forward Pass with Automatic Mixed Precision (BFloat16) ---
            # BFloat16 natively prevents underflows, no GradScaler needed.
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(states, actions, rtg, timesteps)
                
                # Cross-Entropy Loss computation
                # Flatten the Batch and Sequence dimensions: (B*K, act_dim)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), 
                    actions.reshape(-1)
                )

            # --- Backward Pass ---
            optimizer.zero_grad(set_to_none=True) # More efficient than zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if step % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1:02d} | Step {step:04d}/{len(dataloader)} | Loss: {loss.item():.4f} | LR: {current_lr:.2e}")

        # Epoch Profiling
        avg_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - start_time
        samples_per_sec = (len(dataloader) * batch_size) / epoch_time
        
        print("-" * 60)
        print(f"Epoch {epoch+1:02d} Completed | Avg Loss: {avg_loss:.4f}")
        print(f"Throughput: {samples_per_sec:.2f} samples/s | Time: {epoch_time:.2f}s")
        print("-" * 60)
        
        # Checkpoint mapping
        checkpoint_path = f"dt_model_ep{epoch+1:02d}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="offline_dataset.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64) # Adjust depending on VRAM usage
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    train_decision_transformer(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )