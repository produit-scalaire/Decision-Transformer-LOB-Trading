# Reinforcement Learning via Sequence Modeling for Market Data

**Authors:** Côme Genet & Clément Callaert  
**Date:** February 2026

## Project Overview

Traditional Reinforcement Learning (RL) methods utilizing Temporal Difference (TD) learning or policy gradients often suffer from severe instability in highly noisy, non-stationary financial environments.

This repository explores a paradigm shift by framing the High-Frequency Trading (HFT) RL problem as a conditional sequence modeling task. Instead of explicitly estimating value functions ($Q$-learning) or computing policy gradients to maximize expected return, we implement an Offline RL agent based on the **Decision Transformer (DT)** architecture. By modeling trajectories of past states, actions, and returns-to-go (RTG), the agent leverages causal self-attention mechanisms to autoregressively predict optimal actions that align with a specific target return.

> **Note:** The accompanying Jupyter Notebook serves strictly as an exploratory introduction to the dataset and the core attention mechanisms. The actual training, generation, and evaluation pipelines are orchestrated via the production-ready Python modules described below.

## Hardware & Performance Profiling

This pipeline has been specifically architected and optimized for high-end local compute clusters. The default `config.yaml` is tuned for the following hardware constraints:

| Component | Specification |
|-----------|---------------|
| **CPU** | AMD Ryzen 9 9950X (Data generation scaling via 32 parallel workers, OMP/MKL thread pinning to prevent dataloader thrashing) |
| **GPU** | NVIDIA RTX 5090 32GB VRAM (`torch.set_float32_matmul_precision('high')` enabled to leverage Ada/Blackwell Tensor Cores) |
| **RAM** | 64 GB DDR5 |
| **Optimization** | `torch.compile` (Torch 2.x max-autotune) for kernel fusion, and massive batch sizing (`batch_size: 2048`) to maximize SM occupancy on the RTX 5090 |

## Problem Formulation

We formalize the trading process as a Markov Decision Process (MDP) defined by the tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$, adapted for an offline autoregressive sequence modeling approach:

### State Space ($\mathcal{S}$)
A temporal window of normalized Limit Order Book (LOB) data (depth of 10 levels) is stacked with the scalar position to form a **41-dimensional** flattened state (40 LOB features + 1). By default (**`raw`**) the LOB channels are used as loaded (e.g. z-scored snapshots). Optionally, **stationary price channels** can be enabled so only ask/bid **prices** are transformed per tick; **volumes are unchanged**:

| Mode (`generator.state_representation`) | Price columns (20 levels) |
|----------------------------------------|---------------------------|
| **`raw`** | Snapshot values as in the dataset |
| **`log_returns`** | \(\log(p_t + \delta) - \log(p_{t-1} + \delta)\) with configurable **`generator.price_offset`** \(\delta\) (stabilizes z-scored levels) |
| **`bps`** | \((p_t - p_{t-1}) / (\|p_{t-1}\| + \delta) \times 10^4\) |

The environment applies this inside `LOBTradingEnv` (`src/env/lob_trading_env.py`); trajectory workers and Hydra use the same flags. Evaluation rollouts (`src/evaluations/market_returns.py`, wired from `main.py`) must use the same `state_representation` as the test trajectories. State dimensionality stays **41** across modes so the Decision Transformer input size does not change.

### Action Space ($\mathcal{A}$)
$$\mathcal{A} = \{-1, 0, 1\} \quad \text{(Sell, Hold, Buy)}$$

### Reward Function ($\mathcal{R}$)
$r_t$ evaluates the mark-to-market simulated profit. To penalize excessive churn and enforce strict microstructural realism, a transaction cost $c > 0$ is subtracted for any non-zero $\Delta$ in the agent's inventory.

**Configurable reward modes** (`configs/config.yaml` → `generator.reward_type`):

| Mode | Definition |
|------|------------|
| **`mid_price`** (default) | Step reward is base PnL only: position $\times$ change in mid price minus transaction costs. This is what the Decision Transformer conditions on when learning from RTGs. |
| **`shaped`** | Same base PnL, minus optional penalties controlled by `generator.reward_shaping`: drawdown from the running peak of **cumulative base PnL**, rolling variance of recent **base** step rewards (window `variance_window`), and a **time-in-market** term proportional to $\| \text{position} \|$. Use this when you want offline data or experiments that encode risk aversion beyond raw mid moves. |

Trajectory generation (`src/data/trajectories_generator.py`) and `main.py` pass reward and state-representation settings into `LOBTradingEnv` so workers and the Hydra pipeline stay aligned.

### Trajectory Representation
The causal transformer operates on context windows of length $K$:

$$\tau = (\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots, \hat{R}_K, s_K, a_K)$$

where $\hat{R}_{t}$ is the **Return-to-Go**, computed with a configurable horizon $H$:

$$\hat{R}_t = \sum_{t'=t}^{\min(t+H,\, T)-1} r_{t'}$$

| `generator.reward_horizon` | Behaviour |
|----------------------------|-----------|
| `null` (default) | Sum all rewards from $t$ to the episode end $T$ (original DT paper) |
| positive integer $H$ | Sum only the next $H$ steps; rewards further than $H$ steps ahead are excluded |

A short horizon focuses the agent on near-term outcomes and can reduce the variance of RTG estimates in long episodes. See [Configuring the RTG horizon](#configuring-the-rtg-horizon) below.

## Pipeline Architecture & Execution

The repository relies on `hydra-core` for hierarchical configuration management (`config.yaml`). The pipeline is divided into three distinct phases:

- **Data Generation**
- **Training**
- **Evaluation**

### FI-2010 Multi-Stock Handling

FI-2010 cross-validation fold files concatenate all **5 Finnish stocks** along the time axis into a single matrix. Naively sampling random episodes from this concatenated array lets windows cross stock boundaries, where the mid-price jumps discontinuously between unrelated equities — producing garbage transitions and meaningless rewards.

The data-generation pipeline therefore **detects stock boundaries** automatically (via large discontinuities in the mid-price series) and generates episodes **independently per stock**. Episode counts are allocated proportionally to each stock's data length. The continuous Day 10 evaluation (`continuous_day10_plot`) likewise evaluates per stock, producing one PnL plot per equity rather than a single plot that conflates five different instruments.

## Model Architectures

Two model architectures are available and controlled entirely by the `model.architecture` key in `configs/config.yaml` (or via Hydra CLI override). Both models share the same interface, training loop, and evaluation code — only the state embedding module differs.

### Transformer (default)

```
architecture: "transformer"
```

The original Decision Transformer. The state vector `(state_dim,)` is embedded with a single linear projection (`nn.Linear(state_dim, d_model)`). This is the fast baseline.

```
State (B, K, state_dim)
       │
       ▼
  nn.Linear(state_dim → d_model)
       │
       ▼
  + time positional embedding
       │
       ▼
  Causal Transformer blocks  ×  n_layers
       │
       ▼
  Action head  →  logits (B, K, act_dim)
```

### CNN Encoder

```
architecture: "cnn"
```

Replaces the linear state embedding with a two-layer 1D CNN encoder (`CNNStateEncoder`). The state vector is treated as a 1D signal of length `state_dim` with one input channel. Two `Conv1d` layers (same-padding, GELU activations) extract local feature patterns, an `AdaptiveAvgPool1d(1)` collapses the spatial dimension to a fixed-size vector regardless of `state_dim`, and a final linear layer projects to `d_model`. Everything downstream (RTG/action embeddings, causal Transformer blocks, action head) is identical to the base model.

```
State (B, K, state_dim)
       │
       ▼  reshape → (B*K, 1, state_dim)
  Conv1d(1 → cnn_channels, kernel=cnn_kernel_size)  →  GELU
  Conv1d(cnn_channels → cnn_channels, kernel=cnn_kernel_size)  →  GELU
       │
       ▼
  AdaptiveAvgPool1d(1)  →  squeeze  →  (B*K, cnn_channels)
       │
       ▼
  nn.Linear(cnn_channels → d_model)  →  reshape → (B, K, d_model)
       │
       ▼
  + time positional embedding
       │
       ▼
  Causal Transformer blocks  ×  n_layers
       │
       ▼
  Action head  →  logits (B, K, act_dim)
```

The `AdaptiveAvgPool1d` makes the encoder **state-dim agnostic**: it works identically for `raw` (41-dim), `log_returns` (41-dim), `bps` (41-dim), or any future state space with a different dimensionality — no code change required.

#### CNN-specific config keys

| Key | Default | Description |
|-----|---------|-------------|
| `model.cnn_channels` | `64` | Number of hidden channels in both Conv1d layers |
| `model.cnn_kernel_size` | `3` | Kernel size for both Conv1d layers |

These keys are ignored when `architecture: "transformer"`.

### Switching architectures

**From the command line (Hydra override):**

```bash
# Use the original transformer (default)
python main.py model.architecture=transformer

# Use the CNN encoder
python main.py model.architecture=cnn

# CNN with custom encoder width and kernel
python main.py model.architecture=cnn model.cnn_channels=128 model.cnn_kernel_size=5
```

**In `configs/config.yaml`:**

```yaml
model:
  architecture: "cnn"   # change this line
  cnn_channels: 64
  cnn_kernel_size: 3
```

> **Important:** a checkpoint saved with one architecture cannot be loaded by the other. If you switch architectures, you must retrain from scratch or keep separate `paths.model_dir` directories.

### 1. Standard Execution

To run the full pipeline using the default parameters specified in `configs/config.yaml`:

```bash
python main.py
```

### 2. Specific Commands & Hydra Overrides

You can selectively run phases or dynamically override hyperparameters directly from the CLI without modifying the source code.

**Run only the training phase** with a specific learning rate and context length:

```bash
python main.py pipeline.run_generation=false pipeline.run_evaluation=false training.lr=5e-5 training.context_len=150
```

**Run generation** with higher CPU concurrency:

```bash
python main.py pipeline.run_training=false pipeline.run_evaluation=false hardware.workers=48 generator.train_episodes=100000
```

**Switch to shaped rewards** for data generation (override coefficients as needed):

```bash
python main.py pipeline.run_training=false pipeline.run_evaluation=false generator.reward_type=shaped
```

**Use stationary LOB prices** (log-returns on price levels; regenerate data and retrain—the model is not interchangeable with `raw` checkpoints):

```bash
python main.py generator.state_representation=log_returns
```

**Evaluate a specific checkpoint** across different target Returns-to-Go:

```bash
python main.py pipeline.run_generation=false pipeline.run_training=false 'evaluation.target_rtgs=[0.5, 1.0, 3.0, 5.0]'
```

### Configuring the RTG horizon

By default `generator.reward_horizon: null` (from `configs/config.yaml`) uses the full episode length to compute $\hat{R}_t$. Set it to a positive integer to cap the look-ahead window.

**In `configs/config.yaml`:**

```yaml
generator:
  reward_horizon: 200   # only sum the next 200 reward steps
```

**Via Hydra CLI override** (regenerate data only, keep existing model):

```bash
# Short horizon of 50 steps
python main.py pipeline.run_training=false pipeline.run_evaluation=false \
  generator.reward_horizon=50

# Restore to full-episode RTG (null must be passed as a string)
python main.py pipeline.run_training=false pipeline.run_evaluation=false \
  generator.reward_horizon=null
```

> **Important:** the horizon only affects data generation (the stored `.pt` files). A model trained on data generated with `reward_horizon=50` conditions on short-horizon RTGs, so you must pass comparable target-RTG values at evaluation time. Changing this setting after training requires regenerating the dataset and retraining from scratch.

**Context horizon sweep (train one checkpoint per $K$, then profile):** use a separate `paths.model_dir` per context length so `dt_model_final.pt` files are not overwritten, then aggregate:

```bash
for K in 50 100 250 500; do
  python main.py pipeline.run_generation=false \
    training.context_len=$K evaluation.context_len=$K \
    paths.model_dir=models/context_K${K}
done
python scripts/context_horizon_profile.py \
  --checkpoints \
    50=models/context_K50/dt_model_final.pt \
    100=models/context_K100/dt_model_final.pt \
    250=models/context_K250/dt_model_final.pt \
    500=models/context_K500/dt_model_final.pt \
  --state_representation=raw
```

Evaluation summaries from `main.py` also report **F1_macro** for Decision Transformer rows (agreement with the same oracle used in the profile script).

## Tests

The suite uses [pytest](https://docs.pytest.org/) and exercises: the LOB trading environment (mid vs. shaped rewards, stationary state modes, market-return helpers), trajectory dataset indexing, Decision Transformer forward pass and causality, rollout worker behavior, and the CNN encoder / model factory. Install dependencies (including `pytest`) from the repo root:

```bash
pip install -r requirements.txt
```

Run all tests:

```bash
pytest
```

Run only the main test module with verbose output:

```bash
pytest tests/test.py -v
```

Run a single test by name:

```bash
pytest tests/test.py::test_dt_causality -v
```

The project root must be the current working directory so imports resolve (`src.*` packages). `pytest.ini` configures collection so `tests/test.py` is discovered even though its name is not `test_*.py`.

## To Do / Future Work

### Core Experimentation

- [x] **Reward Shaping Formulation:** Parameterize the reward function to penalize drawdowns, variance, or time-in-market, moving beyond simple mid-price PnL. Implemented via `generator.reward_type` / `generator.reward_shaping` in `configs/config.yaml`, `LOBTradingEnv` in `src/env/lob_trading_env.py`, and the trajectory generator worker wiring; covered by tests in `tests/test.py`.
- [x] **State Space Engineering:** Switchable observation construction for ask/bid **price** levels only—**`raw`** (dataset layout), **`log_returns`**, or **`bps`**—with **`generator.price_offset`** for numerical stability on z-scored data; volumes unchanged; dim stays 41. Implemented in `src/env/lob_trading_env.py`, config keys `generator.state_representation` / `generator.price_offset`, trajectory `init_worker` wiring in `src/data/trajectories_generator.py`, evaluation helpers in `src/evaluations/market_returns.py` and `evaluate_model(..., state_representation=...)` in `src/evaluations/dt_viz.py`; covered in `tests/test.py`.
- [x] **Context Horizon Profiling:** Benchmark Sharpe ratio and macro-F1 (vs instantaneous mid-proxy oracle) across attention windows $K \in \{50, 100, 250, 500\}$ via per-$K$ training checkpoints and `scripts/context_horizon_profile.py`. DeepLOB (Zhang et al., 2018) FI-2010 movement F1 at horizons $k \in \{10,50,100\}$ is cited in the script plot as a qualitative reference only (different dataset and label definition).
- [ ] **Architectural Scaling:** Conduct ablation studies on the transformer depth and width (`d_model`, `n_heads`, `n_layers`) relative to the available 32GB VRAM.
- [x] **CNN architecture:** A CNN-based state encoder (`CNNDecisionTransformer`) is available as a drop-in replacement for the base transformer. A `model.architecture` config key switches between the two paths at runtime. See [Model Architectures](#model-architectures) below for details.
- [x] **Return-to-go horizon ($H$):** `generator.reward_horizon` in `configs/config.yaml` caps the future-reward sum: `null` keeps the original full-episode behaviour; a positive integer $H$ restricts the sum to the next $H$ steps. Implemented via `compute_rtg()` in `src/data/trajectories_generator.py`, threaded through `init_worker` / `rollout_worker` / `generate_dataset` / `generate_dataset_pipeline` / `main.py`; covered by tests in `tests/test.py`.

### Dataset & Environment Expansion

- [ ] **Traditional Equities:** Port the environment and evaluate the model on standard equity ticks (e.g., LSE dataset) to compare cross-asset microstructural dynamics.
- [ ] **Alternative LOB Datasets:** Test out-of-sample robustness on LOBSTER data or proprietary crypto tick data.
- [ ] **FI-2010 Feature Engineering:** Integrate and benchmark the handcrafted temporal features (e.g., moving averages, volatility windows) historically provided in the FI-2010 dataset.

### Advanced Research (DeepLOB Integration)

Inspired by Zhang et al. (2018):

- [ ] **Spatial-Temporal State Encoding:** The current Decision Transformer reads flattened LOB states. Implement a DeepLOB-style Convolutional frontend (Inception modules with asymmetric $1\times2$ and $1\times10$ kernels) to extract micro-price and imbalance features before feeding the embeddings to the causal transformer.
- [ ] **Adaptive Smoothing Labels:** Evaluate utilizing DeepLOB's smoothed future mid-price prediction horizons ($k=10, 50, 100$) as an auxiliary loss term or as the basis for the intrinsic reward function to reduce signal-to-noise ratio in the RTG computation.
- [ ] **Execution Latency Simulation:** Introduce queue position estimation and order execution probability (e.g., Hawkes processes) into the MDP, replacing the naive assumption of immediate L1 fill.

## References

1. Chen, L., et al. (2021). **Decision Transformer: Reinforcement Learning via Sequence Modeling.** *NeurIPS*.

2. Ntakaris, A., et al. (2018). **Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods.** *Journal of Forecasting*.

3. Zhang, Z., et al. (2018). **DeepLOB: Deep Convolutional Neural Networks for Limit Order Books.** *IEEE Transactions on Signal Processing*.