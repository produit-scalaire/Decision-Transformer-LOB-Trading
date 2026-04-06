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
A temporal window of normalized Limit Order Book (LOB) data (depth of 10 levels, yielding a 41-dimensional vector including time/features).

### Action Space ($\mathcal{A}$)
$$\mathcal{A} = \{-1, 0, 1\} \quad \text{(Sell, Hold, Buy)}$$

### Reward Function ($\mathcal{R}$)
$r_t$ evaluates the mark-to-market simulated profit. To penalize excessive churn and enforce strict microstructural realism, a transaction cost $c > 0$ is subtracted for any non-zero $\Delta$ in the agent's inventory.

### Trajectory Representation
The causal transformer operates on context windows of length $K$:

$$\tau = (\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots, \hat{R}_K, s_K, a_K)$$

where $\hat{R}_{t} = \sum_{t'=t}^{T} r_{t'}$ is the **Return-to-Go**.

## Pipeline Architecture & Execution

The repository relies on `hydra-core` for hierarchical configuration management (`config.yaml`). The pipeline is divided into three distinct phases:

- **Data Generation**
- **Training**
- **Evaluation**

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

**Evaluate a specific checkpoint** across different target Returns-to-Go:

```bash
python main.py pipeline.run_generation=false pipeline.run_training=false 'evaluation.target_rtgs=[0.5, 1.0, 3.0, 5.0]'
```


## To Do / Future Work

### Core Experimentation

- [ ] **Reward Shaping Formulation:** Parameterize the reward function to penalize drawdowns, variance, or time-in-market, moving beyond simple mid-price PnL.
- [ ] **State Space Engineering:** Experiment with alternative state representations (e.g., transforming absolute prices into stationary log-returns or relative basis points).
- [ ] **Context Horizon Profiling:** Benchmark model performance (Sharpe Ratio, F1-Score) across varying attention window sizes ($K \in \{50, 100, 250, 500\}$) to evaluate memory decay vs. predictive power.
- [ ] **Architectural Scaling:** Conduct ablation studies on the transformer depth and width (`d_model`, `n_heads`, `n_layers`) relative to the available 32GB VRAM.

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