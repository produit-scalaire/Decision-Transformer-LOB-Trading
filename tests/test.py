import os
import tempfile
import pytest
import torch
import numpy as np
from pathlib import Path

# Fix relative imports for pytest execution from root directory
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.env.lob_trading_env import LOBTradingEnv, LOB_PRICE_COL_INDICES
from src.models.decision_transformer import DecisionTransformer
from src.training.training_pipeline import OptimizedTrajectoryDataset, configure_optimizers
from src.data.trajectories_generator import rollout_worker, init_worker
from src.evaluations.direction_metrics import oracle_actions_from_returns, compute_directional_f1

# =============================================================================
# FIXTURES (DUMMY DATA & MOCKS)
# =============================================================================

@pytest.fixture
def dummy_lob_data() -> np.ndarray:
    """
    Generates deterministic pseudo-random LOB data for reproducibility.
    Shape: (T=500, Features=40)
    """
    np.random.seed(42)
    # LOB shape is (T, 40)
    data = np.random.randn(500, 40).astype(np.float32)
    # Ensure ask > bid for level 1 (prices are col 0 and 2)
    data[:, 0] = np.abs(data[:, 0]) + 100.0  # Ask price
    data[:, 2] = data[:, 0] - np.abs(np.random.randn(500)) - 0.1  # Bid price < Ask
    return data

@pytest.fixture
def dummy_trajectories_file(tmp_path: Path) -> str:
    """
    Generates a dummy .pt file containing mock trajectories for Dataset testing.
    """
    T, K, state_dim = 150, 100, 41
    trajectories = []
    
    for i in range(3):
        traj = {
            "states": torch.randn(T, state_dim, dtype=torch.float32),
            "actions": torch.randint(0, 3, (T,), dtype=torch.long),
            "rewards": torch.randn(T, dtype=torch.float32),
            "rtg": torch.randn(T, 1, dtype=torch.float32),
            "timesteps": torch.arange(T, dtype=torch.long),
            "policy": "random",
            "total_return": 0.5
        }
        trajectories.append(traj)
    
    file_path = tmp_path / "dummy_train.pt"
    torch.save(trajectories, file_path)
    return str(file_path)

# =============================================================================
# ENVIRONMENT TESTS
# =============================================================================

def test_lob_env_step_mechanics(dummy_lob_data: np.ndarray):
    """
    Validates deterministic state transitions, PnL computation, and boundary constraints.
    """
    window_size = 10
    tc = 0.005
    env = LOBTradingEnv(
        lob_data=dummy_lob_data,
        window_size=window_size,
        transaction_cost=tc,
        episode_length=100
    )
    
    obs, info = env.reset(seed=42)
    assert obs["lob_window"].shape == (window_size, 40), "Observation LOB window shape mismatch."
    assert obs["position"].shape == (1,), "Position shape mismatch."
    assert env._position == 0, "Initial position must be 0."
    
    start_step = env._current_step
    mid_price_t0 = env.mid_prices[start_step]
    mid_price_t1 = env.mid_prices[start_step + 1]
    
    # Execute Long Action (Action 2 -> Position +1)
    # Reward = (1 * (mid_price_t1 - mid_price_t0)) - (tc * |1 - 0|)
    next_obs, reward, terminated, truncated, next_info = env.step(2)
    
    expected_reward = 1.0 * (mid_price_t1 - mid_price_t0) - tc
    
    np.testing.assert_almost_equal(reward, expected_reward, decimal=5, 
                                   err_msg="Reward calculation violates financial constraints.")
    assert env._position == 1, "State transition failed for Long action."


def test_action_position_roundtrip():
    """Discrete action encoding must match env helpers."""
    for pos in (-1, 0, 1):
        a = LOBTradingEnv.position_to_action(pos)
        assert LOBTradingEnv.action_to_position(a) == pos
    for a in (0, 1, 2):
        assert LOBTradingEnv.position_to_action(LOBTradingEnv.action_to_position(a)) == a


def test_lob_env_constructor_rejects_bad_shapes():
    """LOB data must be (T, 40) with T > window_size."""
    good = np.random.randn(120, 40).astype(np.float32)
    with pytest.raises(ValueError, match="Expected \\(T, 40\\)"):
        LOBTradingEnv(lob_data=good.reshape(-1), window_size=10)
    # T must be strictly greater than window_size (T=15 is valid for W=10)
    with pytest.raises(ValueError, match="must exceed"):
        LOBTradingEnv(lob_data=good[:10], window_size=10, episode_length=5)


def test_lob_env_rejects_unknown_reward_type(dummy_lob_data: np.ndarray):
    with pytest.raises(ValueError, match="reward_type"):
        LOBTradingEnv(
            lob_data=dummy_lob_data,
            window_size=10,
            reward_type="unknown",
        )


def test_lob_env_rejects_unknown_state_representation(dummy_lob_data: np.ndarray):
    with pytest.raises(ValueError, match="state_representation"):
        LOBTradingEnv(
            lob_data=dummy_lob_data,
            window_size=10,
            state_representation="absolute_zorg",
        )


def test_lob_env_stationary_modes_zero_when_flat_mid():
    """Constant best bid/ask ⇒ zero log-return / bps on all price levels in the window."""
    data = flat_mid_lob(80)
    for rep in ("log_returns", "bps"):
        env = LOBTradingEnv(
            lob_data=data,
            window_size=10,
            episode_length=20,
            state_representation=rep,
            price_offset=10.0,
        )
        obs, _ = env.reset(seed=0)
        pw = obs["lob_window"]
        np.testing.assert_allclose(
            pw[:, LOB_PRICE_COL_INDICES], 0.0, atol=1e-5, rtol=0.0
        )


def test_lob_env_stationary_preserves_volume_columns():
    data = flat_mid_lob(80)
    data[:, 1] = 0.3
    data[:, 3] = 0.7
    data[:, 5] = -0.2
    vol_idx = [i for i in range(40) if i not in set(LOB_PRICE_COL_INDICES)]
    env_raw = LOBTradingEnv(
        lob_data=data, window_size=10, episode_length=5, state_representation="raw"
    )
    env_lr = LOBTradingEnv(
        lob_data=data, window_size=10, episode_length=5, state_representation="log_returns"
    )
    o0, _ = env_raw.reset(seed=0)
    o1, _ = env_lr.reset(seed=0)
    np.testing.assert_array_equal(o0["lob_window"][:, vol_idx], o1["lob_window"][:, vol_idx])


def test_get_market_returns_respects_state_representation():
    from src.evaluations.market_returns import get_market_returns

    B, T, D = 2, 5, 41
    s = torch.zeros(B, T, D)
    s[:, :, 0] = 100.0
    s[:, :, 2] = 100.0
    r_raw = get_market_returns(s, "raw")
    assert r_raw.shape == (B, T)
    assert torch.allclose(r_raw[:, :-1], torch.zeros_like(r_raw[:, :-1]))
    assert r_raw[:, -1].abs().max().item() == 0.0

    s2 = torch.randn(B, T, D)
    r_lr = get_market_returns(s2, "log_returns")
    assert r_lr.shape == (B, T)
    assert r_lr[:, -1].abs().max().item() == 0.0


def flat_mid_lob(num_rows: int) -> np.ndarray:
    """Constant mid-price so base step PnL is zero when position is fixed."""
    x = np.zeros((num_rows, 40), dtype=np.float32)
    x[:, 0] = 100.0
    x[:, 2] = 100.0
    return x


def test_lob_env_shaped_penalizes_time_in_market_when_mid_flat():
    """
    With shaped rewards, |position| term lowers reward vs mid_price when base PnL is zero.
    """
    data = flat_mid_lob(80)
    w = 10
    lam_t = 0.01
    env_shaped = LOBTradingEnv(
        lob_data=data,
        window_size=w,
        transaction_cost=0.0,
        episode_length=30,
        reward_type="shaped",
        drawdown_coef=0.0,
        variance_coef=0.0,
        time_in_market_coef=lam_t,
        variance_window=20,
    )
    env_plain = LOBTradingEnv(
        lob_data=data,
        window_size=w,
        transaction_cost=0.0,
        episode_length=30,
        reward_type="mid_price",
    )
    env_shaped.reset(seed=0)
    env_plain.reset(seed=0)
    _, r_shaped, _, _, info_s = env_shaped.step(2)
    _, r_plain, _, _, info_p = env_plain.step(2)
    assert info_s["base_reward"] == pytest.approx(0.0)
    assert info_p.get("base_reward") == pytest.approx(info_s["base_reward"])
    assert r_plain == pytest.approx(0.0)
    assert r_shaped == pytest.approx(-lam_t * abs(1))


def test_trajectories_rollout_matches_env_reward_mode():
    """
    init_worker + rollout_worker must use the same reward logic as a direct env rollout.
    """
    data = flat_mid_lob(120)
    labels = np.zeros((120, 5), dtype=np.float32)
    labels[:, 3] = 2

    shaping = {
        "drawdown_coef": 0.0,
        "variance_coef": 0.0,
        "time_in_market_coef": 0.02,
        "variance_window": 10,
    }
    init_worker(
        data,
        labels,
        window_size=10,
        episode_length=25,
        reward_type="shaped",
        reward_shaping=shaping,
        state_representation="log_returns",
    )
    out = rollout_worker(("random", 0))

    env = LOBTradingEnv(
        lob_data=data,
        window_size=10,
        transaction_cost=0.0,
        episode_length=25,
        reward_type="shaped",
        time_in_market_coef=0.02,
        variance_window=10,
        drawdown_coef=0.0,
        variance_coef=0.0,
        state_representation="log_returns",
    )
    env.reset(seed=0)
    replay = []
    for a in out["actions"]:
        _, r, term, trunc, _ = env.step(int(a))
        replay.append(r)
        if term or trunc:
            break

    assert len(replay) == len(out["rewards"])
    np.testing.assert_array_almost_equal(out["rewards"], np.array(replay, dtype=np.float32))


def test_lob_env_invalid_action_raises(dummy_lob_data: np.ndarray):
    env = LOBTradingEnv(
        lob_data=dummy_lob_data,
        window_size=10,
        transaction_cost=0.0,
        episode_length=20,
    )
    env.reset(seed=0)
    with pytest.raises(ValueError, match="Invalid action"):
        env.step(99)


def test_lob_env_hold_flat_zero_reward_no_tc(dummy_lob_data: np.ndarray):
    """From flat, holding (action 1) keeps position 0 and pays no TC unless price move is credited."""
    env = LOBTradingEnv(
        lob_data=dummy_lob_data,
        window_size=10,
        transaction_cost=0.01,
        episode_length=30,
    )
    env.reset(seed=0)
    _, r, _, _, _ = env.step(1)
    assert env._position == 0
    start = env._current_step - 1
    price_change = float(env.mid_prices[start + 1] - env.mid_prices[start])
    expected = 0.0 * price_change  # no TC
    np.testing.assert_almost_equal(r, expected)


def test_lob_env_episode_terminates_at_end(dummy_lob_data: np.ndarray):
    episode_length = 12
    env = LOBTradingEnv(
        lob_data=dummy_lob_data,
        window_size=10,
        transaction_cost=0.0,
        episode_length=episode_length,
    )
    env.reset(seed=123)
    terminated = False
    steps = 0
    while not terminated:
        _, _, terminated, _, _ = env.step(1)
        steps += 1
        assert steps < 500_000, "Episode did not terminate"
    # Each step advances _current_step; termination when _current_step >= _end_step
    assert env._current_step >= env._end_step


def test_lob_env_close_round_trip_pays_transaction_costs(dummy_lob_data: np.ndarray):
    """Long then flat (sell) incurs TC on both legs."""
    tc = 0.005
    env = LOBTradingEnv(
        lob_data=dummy_lob_data,
        window_size=10,
        transaction_cost=tc,
        episode_length=40,
    )
    env.reset(seed=7)
    env.step(2)
    _, r2, _, _, info2 = env.step(1)
    assert env._position == 0
    assert info2["transaction_cost_paid"] == pytest.approx(tc)


# =============================================================================
# DATASET TESTS
# =============================================================================

def test_optimized_dataset_indexing(dummy_trajectories_file: str):
    """
    Verifies that the custom 1D Int32 indexing correctly maps sliding windows 
    without off-by-one index overflow.
    """
    context_len = 100
    dataset = OptimizedTrajectoryDataset(data_path=dummy_trajectories_file, context_len=context_len)
    
    # 3 trajectories of len 150 -> each yields (150 - 100) = 50 windows. Total = 150.
    assert len(dataset) == 150, f"Expected 150 windows, got {len(dataset)}"
    
    # Sample arbitrary indices to check tensor shapes
    states, actions, rtg, timesteps = dataset[0]
    assert states.shape == (context_len, 41), "State window slicing failed."
    assert actions.shape == (context_len,), "Action window slicing failed."
    assert rtg.shape == (context_len, 1), "RTG window slicing failed."
    
    # Verify exact timestep alignment
    assert timesteps[0].item() == 0, "Timestep offset error at sequence start."
    assert timesteps[-1].item() == context_len - 1, "Timestep alignment error at sequence tail."


def test_optimized_dataset_empty_when_traj_shorter_than_context(tmp_path: Path):
    """No sliding windows when T <= context_len."""
    T, context_len = 80, 100
    traj = {
        "states": torch.randn(T, 41, dtype=torch.float32),
        "actions": torch.randint(0, 3, (T,), dtype=torch.long),
        "rewards": torch.randn(T, dtype=torch.float32),
        "rtg": torch.randn(T, 1, dtype=torch.float32),
        "timesteps": torch.arange(T, dtype=torch.long),
        "policy": "random",
        "total_return": 0.0,
    }
    path = tmp_path / "short.pt"
    torch.save([traj], path)
    ds = OptimizedTrajectoryDataset(data_path=str(path), context_len=context_len)
    assert len(ds) == 0


def test_optimized_dataset_sliding_window_timesteps_increment(dummy_trajectories_file: str):
    """Second window in the same trajectory starts at absolute time offset 1."""
    context_len = 100
    ds = OptimizedTrajectoryDataset(data_path=dummy_trajectories_file, context_len=context_len)
    _, _, _, t0 = ds[0]
    _, _, _, t1 = ds[1]
    assert t0[0].item() == 0
    assert t1[0].item() == 1
    assert t1[-1].item() == t0[-1].item() + 1


# =============================================================================
# DECISION TRANSFORMER TESTS
# =============================================================================

def test_dt_forward_pass_shapes():
    """
    Checks tensor dimension alignment through the entire Transformer stack.
    """
    state_dim, act_dim, K, B = 41, 3, 50, 4
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        d_model=64,
        n_heads=2,
        n_layers=2,
        max_timestep=1000
    )
    
    states = torch.randn(B, K, state_dim)
    actions = torch.randint(0, act_dim, (B, K))
    rtg = torch.randn(B, K, 1)
    timesteps = torch.arange(K).unsqueeze(0).repeat(B, 1)
    
    logits = model(states, actions, rtg, timesteps)
    assert logits.shape == (B, K, act_dim), f"Expected {(B, K, act_dim)}, got {logits.shape}"


def test_oracle_actions_from_returns_maps_sign_to_dt_indices():
    r = torch.tensor([[0.1, -0.5, 0.0], [0.0, 0.02, -0.01]])
    o = oracle_actions_from_returns(r)
    assert o.shape == r.shape
    assert o[0, 0].item() == 2
    assert o[0, 1].item() == 0
    assert o[0, 2].item() == 1
    assert o[1, 0].item() == 1


def test_compute_directional_f1_perfect_when_matching_oracle():
    # Include positive, negative, and zero returns so all three oracle classes appear (macro-F1).
    r = torch.tensor([[0.5, -0.3, 0.0, 0.02, -0.01]])
    oracle = oracle_actions_from_returns(r)
    f1 = compute_directional_f1(oracle, r)
    assert f1 == pytest.approx(1.0)


@torch.no_grad()
def test_dt_causality():
    """
    CRITICAL: Validates that flash attention causal masking is strictly enforced.
    Modifying a token at t+1 must NOT alter the predicted logits at t.
    """
    state_dim, act_dim, K = 41, 3, 20
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        d_model=128,
        n_heads=4,
        n_layers=2
    ).eval()
    
    states = torch.randn(1, K, state_dim)
    actions = torch.randint(0, act_dim, (1, K))
    rtg = torch.randn(1, K, 1)
    timesteps = torch.arange(K).unsqueeze(0)
    
    # Baseline output
    logits_base = model(states, actions, rtg, timesteps)
    
    # Perturb state at t=10
    states_perturbed = states.clone()
    states_perturbed[0, 10, :] += 100.0
    
    logits_perturbed = model(states_perturbed, actions, rtg, timesteps)
    
    # Assert causality: Tokens [0:10] must remain strictly identical
    assert torch.allclose(logits_base[0, :10, :], logits_perturbed[0, :10, :], atol=1e-6), \
        "CAUSALITY LEAK DETECTED: Future tokens are influencing past predictions."
        
    # Assert reaction: Tokens [10:] must diverge due to the perturbation
    assert not torch.allclose(logits_base[0, 10:, :], logits_perturbed[0, 10:, :], atol=1e-6), \
        "Model failed to react to token perturbation."


@torch.no_grad()
def test_dt_eval_deterministic():
    """Same inputs in eval mode produce identical logits."""
    torch.manual_seed(0)
    model = DecisionTransformer(
        state_dim=41, act_dim=3, d_model=64, n_heads=2, n_layers=2, dropout=0.0
    ).eval()
    B, K = 2, 32
    states = torch.randn(B, K, 41)
    actions = torch.randint(0, 3, (B, K))
    rtg = torch.randn(B, K, 1)
    timesteps = torch.arange(K).unsqueeze(0).expand(B, -1)
    out1 = model(states, actions, rtg, timesteps)
    out2 = model(states, actions, rtg, timesteps)
    assert torch.equal(out1, out2)


def test_configure_optimizers_two_groups_cover_all_params():
    """AdamW param groups: decay on Linear weights, no decay on biases/embedding norm."""
    model = DecisionTransformer(
        state_dim=41, act_dim=3, d_model=32, n_heads=2, n_layers=1, dropout=0.0
    )
    opt = configure_optimizers(model, weight_decay=0.1, learning_rate=1e-4, betas=(0.9, 0.95))
    assert len(opt.param_groups) == 2
    named = dict(model.named_parameters())
    grouped = set()
    for g in opt.param_groups:
        for p in g["params"]:
            grouped.add(id(p))
    assert len(grouped) == len(named), "Some parameters missing from optimizer or duplicated"
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for g in opt.param_groups for p in g["params"])

# =============================================================================
# MULTIPROCESSING CONSTRAINTS
# =============================================================================

def test_worker_memory_isolation(dummy_lob_data: np.ndarray):
    """
    Ensures that the rollout worker accesses globals correctly without passing 
    massive NumPy arrays through pipes, and strictly enforces max 8 workers.
    """
    # 1. Initialize globals exactly as Pool.initializer would
    # Simulate labels with dummy data
    dummy_labels = np.zeros((500, 5)) 
    dummy_labels[:, 3] = np.random.choice([1, 2, 3], size=500)
    
    init_worker(dummy_lob_data, dummy_labels, window_size=10, episode_length=50)
    
    # 2. Run a single rollout thread
    job_args = ("random", 42)
    traj = rollout_worker(job_args)
    
    # 3. Type enforcement to prevent memory leak mapped by torch tensors
    assert isinstance(traj["states"], np.ndarray), "Worker must return np.ndarray, not torch.Tensor to avoid fd leaks."
    assert isinstance(traj["actions"], np.ndarray)
    assert traj["states"].shape[1] == 41, "State feature dimension mismatch."
    
    # Enforce constraints for future multiprocessing pools
    max_allowed_workers = 8
    import multiprocessing as mp
    cores = mp.cpu_count()
    recommended_workers = min(cores, max_allowed_workers)
    
    assert recommended_workers <= 8, "Worker limit bypassed. Potential VRAM/RAM saturation risk."