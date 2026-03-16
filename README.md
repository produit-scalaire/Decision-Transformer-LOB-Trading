# Reinforcement Learning via Sequence Modeling for Market Data

**Authors**: Côme Genet & Clément Callaer  
**Date**: February 2026  

## Project Overview

Traditional Reinforcement Learning (RL) methods utilizing Temporal Difference (TD) learning or policy gradients often suffer from instability in highly noisy and non-stationary environments, such as financial markets. 

This project explores a paradigm shift by framing the Reinforcement Learning problem as a **conditional sequence modeling task**. Instead of explicitly estimating value functions or computing policy gradients to maximize expected return, we implement an **Offline RL agent based on the Decision Transformer architecture**. By modeling trajectories of past states, actions, and returns-to-go, the agent leverages self-attention mechanisms to autoregressively predict optimal actions that align with a target return.

## Problem Formulation

We formalize the trading process as a Markov Decision Process (MDP) defined by the tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$, adapted here for an offline sequence modeling approach:

* **State Space ($\mathcal{S}$):** Represents a specific time window of a Limit Order Book (LOB). Features include the best available bid and ask prices, volumes, price gaps between buyers and sellers, and volume imbalances.
* **Action Space ($\mathcal{A}$):** A discrete action space defined as $\mathcal{A} = \{-1, 0, 1\}$, corresponding respectively to Sell, Hold (Do Nothing), and Buy.
* **Reward Function ($\mathcal{R}$):** The reward $r_t$ evaluates the simulated profit of the agent's position. To penalize excessive trading and enforce realism, a transaction cost $c > 0$ is strictly subtracted whenever the agent transitions between different market positions.
* **Trajectory Representation:** The Decision Transformer operates on sequences of length $K$:

    $$\tau = (\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots, \hat{R}_K, s_K, a_K)$$

    where $\hat{R}_{t} = \sum_{t'=t}^{T} r_{t'}$ is the Return-to-Go.

## Methodology & Technologies

* **Core Architecture:** **Decision Transformer**. The model processes the sequence of Returns-to-Go, states, and actions using a causal GPT-like Transformer architecture.
* **Framework:** The entire architecture is implemented **from scratch using PyTorch**.
* **Objective:** At timestep $t$, the model receives the context of the last $K$ timesteps and predicts the immediate action $a_t$ required to achieve the target return $\hat{R}_t$.

## Dataset

The environment simulates a financial market using the public benchmark dataset **FI-2010**. It provides high-frequency normalized Limit Order Book (LOB) data, allowing the agent to observe microstructural market dynamics.

## References

The theoretical foundation and the data preprocessing pipelines of this project are based on the following literature:

1.  **Chen, L., et al. (2021).** *Decision Transformer: Reinforcement Learning via Sequence Modeling.* Advances in Neural Information Processing Systems.
2.  **Ntakaris, A., et al. (2018).** *Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods.* Journal of Forecasting.
3.  **Zhang, Z., et al. (2019).** *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books.* IEEE Transactions on Signal Processing.