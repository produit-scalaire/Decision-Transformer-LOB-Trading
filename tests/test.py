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
from src.models.cnn_decision_transformer import CNNDecisionTransformer, CNNStateEncoder
from src.models.model_factory import build_model, SUPPORTED_ARCHITECTURES
from src.training.training_pipeline import OptimizedTrajectoryDataset, configure_optimizers
from src.data.trajectories_generator import rollout_worker, init_worker, compute_rtg
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


# =============================================================================
# CNN DECISION TRANSFORMER TESTS
# =============================================================================

# Minimal model config used by the factory tests below.
# Using types.SimpleNamespace so we don't depend on Hydra in unit tests.
from types import SimpleNamespace

def _make_model_cfg(architecture: str = "transformer") -> SimpleNamespace:
    return SimpleNamespace(
        architecture=architecture,
        state_dim=41,
        act_dim=3,
        d_model=64,
        n_heads=2,
        n_layers=2,
        max_timestep=1000,
        dropout=0.0,
        cnn_channels=32,
        cnn_kernel_size=3,
    )


def test_cnn_state_encoder_output_shape_various_state_dims():
    """CNNStateEncoder must produce (B, K, d_model) for any state_dim."""
    d_model = 64
    for state_dim in [10, 41, 100, 200]:
        encoder = CNNStateEncoder(state_dim=state_dim, d_model=d_model, channels=32, kernel_size=3)
        x = torch.randn(2, 5, state_dim)
        out = encoder(x)
        assert out.shape == (2, 5, d_model), (
            f"CNNStateEncoder output shape wrong for state_dim={state_dim}: got {out.shape}"
        )


def test_cnn_dt_forward_pass_shapes():
    """CNNDecisionTransformer forward pass produces correct output shapes for various state_dims."""
    for state_dim in [10, 41, 80]:
        model = CNNDecisionTransformer(
            state_dim=state_dim, act_dim=3, d_model=64,
            n_heads=2, n_layers=2, max_timestep=1000,
            cnn_channels=32, cnn_kernel_size=3,
        )
        B, K = 4, 20
        states = torch.randn(B, K, state_dim)
        actions = torch.randint(0, 3, (B, K))
        rtg = torch.randn(B, K, 1)
        timesteps = torch.arange(K).unsqueeze(0).expand(B, -1)
        logits = model(states, actions, rtg, timesteps)
        assert logits.shape == (B, K, 3), (
            f"CNNDecisionTransformer output shape wrong for state_dim={state_dim}: got {logits.shape}"
        )


def test_model_factory_returns_transformer():
    """build_model returns a DecisionTransformer when architecture='transformer'."""
    model = build_model(_make_model_cfg("transformer"))
    assert isinstance(model, DecisionTransformer)


def test_model_factory_returns_cnn():
    """build_model returns a CNNDecisionTransformer when architecture='cnn'."""
    model = build_model(_make_model_cfg("cnn"))
    assert isinstance(model, CNNDecisionTransformer)


def test_model_factory_invalid_architecture_raises():
    """build_model raises ValueError for unknown architecture names."""
    cfg = _make_model_cfg()
    cfg.architecture = "lstm"
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model(cfg)


def test_model_factory_supported_architectures_constant():
    """SUPPORTED_ARCHITECTURES must contain exactly 'transformer' and 'cnn'."""
    assert "transformer" in SUPPORTED_ARCHITECTURES
    assert "cnn" in SUPPORTED_ARCHITECTURES


@torch.no_grad()
def test_cnn_dt_causality():
    """
    Causal masking must hold for CNNDecisionTransformer:
    modifying token at t=10 must not change logits at t < 10.
    """
    state_dim, act_dim, K = 41, 3, 20
    model = CNNDecisionTransformer(
        state_dim=state_dim, act_dim=act_dim, d_model=64,
        n_heads=2, n_layers=2, cnn_channels=32,
    ).eval()

    states = torch.randn(1, K, state_dim)
    actions = torch.randint(0, act_dim, (1, K))
    rtg = torch.randn(1, K, 1)
    timesteps = torch.arange(K).unsqueeze(0)

    logits_base = model(states, actions, rtg, timesteps)

    states_perturbed = states.clone()
    states_perturbed[0, 10, :] += 100.0

    logits_perturbed = model(states_perturbed, actions, rtg, timesteps)

    assert torch.allclose(logits_base[0, :10, :], logits_perturbed[0, :10, :], atol=1e-6), \
        "CAUSALITY LEAK in CNNDecisionTransformer: future tokens affect past predictions."
    assert not torch.allclose(logits_base[0, 10:, :], logits_perturbed[0, 10:, :], atol=1e-6), \
        "CNNDecisionTransformer failed to react to perturbation at t=10."


@torch.no_grad()
def test_cnn_dt_eval_deterministic():
    """Same inputs in eval mode produce identical logits (no dropout noise)."""
    torch.manual_seed(0)
    model = CNNDecisionTransformer(
        state_dim=41, act_dim=3, d_model=64, n_heads=2, n_layers=2,
        cnn_channels=32, dropout=0.0,
    ).eval()
    B, K = 2, 32
    states = torch.randn(B, K, 41)
    actions = torch.randint(0, 3, (B, K))
    rtg = torch.randn(B, K, 1)
    timesteps = torch.arange(K).unsqueeze(0).expand(B, -1)
    out1 = model(states, actions, rtg, timesteps)
    out2 = model(states, actions, rtg, timesteps)
    assert torch.equal(out1, out2)


def test_configure_optimizers_covers_cnn_model_params():
    """
    All CNNDecisionTransformer parameters must be in exactly one AdamW group.
    Conv1d weights go to the decay group; biases/norms go to no-decay.
    """
    model = CNNDecisionTransformer(
        state_dim=41, act_dim=3, d_model=32, n_heads=2, n_layers=1,
        cnn_channels=16, dropout=0.0,
    )
    opt = configure_optimizers(model, weight_decay=0.1, learning_rate=1e-4, betas=(0.9, 0.95))
    assert len(opt.param_groups) == 2

    grouped_ids = set()
    for g in opt.param_groups:
        for p in g["params"]:
            grouped_ids.add(id(p))

    named = dict(model.named_parameters())
    assert len(grouped_ids) == len(named), \
        "CNN model: some parameters are missing from or duplicated in optimizer groups."
    assert (
        sum(p.numel() for p in model.parameters())
        == sum(p.numel() for g in opt.param_groups for p in g["params"])
    )


def test_training_pipeline_builds_cnn_from_config(dummy_trajectories_file: str):
    """
    End-to-end check: build_model with cnn config + real dataset batch flows through
    the CNN DT without errors.
    """
    cfg = _make_model_cfg("cnn")
    model = build_model(cfg)
    assert isinstance(model, CNNDecisionTransformer)

    ds = OptimizedTrajectoryDataset(data_path=dummy_trajectories_file, context_len=100)
    states, actions, rtg, timesteps = ds[0]

    # Add the batch dimension expected by the model.
    with torch.no_grad():
        logits = model(
            states.unsqueeze(0),
            actions.unsqueeze(0),
            rtg.unsqueeze(0),
            timesteps.unsqueeze(0),
        )

    assert logits.shape == (1, 100, 3), f"Unexpected output shape: {logits.shape}"


# =============================================================================
# RETURN-TO-GO HORIZON TESTS
# =============================================================================

def test_compute_rtg_full_horizon_matches_reverse_cumsum():
    """
    With horizon=None, compute_rtg must equal the original reverse-cumsum formula.
    The two implementations should produce bit-identical float32 results.
    """
    np.random.seed(0)
    rewards = np.random.randn(50).astype(np.float32)

    # Original formula used before this feature was added
    expected = np.flip(np.cumsum(np.flip(rewards))).astype(np.float32)
    result = compute_rtg(rewards, horizon=None)

    np.testing.assert_array_almost_equal(
        result, expected, decimal=5,
        err_msg="Full-horizon RTG does not match the original reverse-cumsum formula."
    )


def test_compute_rtg_horizon_zero_treated_as_full():
    """
    horizon=0 is treated the same as horizon=None (full episode sum).
    This keeps the interface forgiving for default/sentinel values.
    """
    rewards = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    full = compute_rtg(rewards, horizon=None)
    zero = compute_rtg(rewards, horizon=0)
    np.testing.assert_array_equal(full, zero)


def test_compute_rtg_horizon_one_equals_reward_itself():
    """
    With H=1 each R̂_t is just r_t (only the current step's reward counts).
    """
    rewards = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
    result = compute_rtg(rewards, horizon=1)
    np.testing.assert_array_almost_equal(result, rewards, decimal=6)


def test_compute_rtg_bounded_horizon_hand_example():
    """
    Manual check: rewards=[1, 2, 3, 4, 5], H=2
    t=0: r_0+r_1 = 3
    t=1: r_1+r_2 = 5
    t=2: r_2+r_3 = 7
    t=3: r_3+r_4 = 9
    t=4: r_4      = 5  (only one step left, window clamps to episode end)
    """
    rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    expected = np.array([3.0, 5.0, 7.0, 9.0, 5.0], dtype=np.float32)
    result = compute_rtg(rewards, horizon=2)
    np.testing.assert_array_almost_equal(result, expected, decimal=6)


def test_compute_rtg_horizon_larger_than_episode_equals_full():
    """
    When H >= T (episode length), the result must equal the full-horizon RTG.
    No out-of-bounds access should occur.
    """
    rewards = np.array([0.5, -0.1, 0.3], dtype=np.float32)
    full = compute_rtg(rewards, horizon=None)
    large = compute_rtg(rewards, horizon=1000)
    np.testing.assert_array_almost_equal(full, large, decimal=6)


def test_compute_rtg_output_dtype_is_float32():
    """RTG array must always come back as float32 regardless of input dtype."""
    rewards_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    for h in (None, 1, 2):
        result = compute_rtg(rewards_f64, horizon=h)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype} for horizon={h}"


def test_compute_rtg_first_element_is_total_return_when_full_horizon():
    """
    The first element of the full-horizon RTG is the episode total return.
    This is the target_rtg used at inference time.
    """
    rewards = np.array([1.0, -0.5, 2.0, 0.3], dtype=np.float32)
    rtg = compute_rtg(rewards, horizon=None)
    assert rtg[0] == pytest.approx(float(rewards.sum()), rel=1e-5)


def test_rollout_worker_uses_horizon_via_global(dummy_lob_data: np.ndarray):
    """
    When init_worker is called with reward_horizon=H, rollout_worker must produce
    RTG values consistent with compute_rtg(..., horizon=H).

    We use H=5 and verify that each R̂_t == sum of the next 5 rewards (clamped).
    """
    dummy_labels = np.zeros((500, 5), dtype=np.float32)
    dummy_labels[:, 3] = 2  # label column used by label_mom policy

    H = 5
    init_worker(dummy_lob_data, dummy_labels, window_size=10, episode_length=50, reward_horizon=H)
    traj = rollout_worker(("random", 99))

    rewards = traj["rewards"]           # shape (T,)
    rtg_worker = traj["rtg"].squeeze()  # shape (T,)  [squeezed from (T, 1)]

    expected_rtg = compute_rtg(rewards, horizon=H)
    np.testing.assert_array_almost_equal(
        rtg_worker, expected_rtg, decimal=5,
        err_msg="Worker RTG does not match compute_rtg with the same horizon."
    )


# =============================================================================
# FINANCIAL METRICS TESTS
# =============================================================================

from src.evaluations.financial_metrics import (
    compute_sortino_ratio,
    compute_var,
    compute_cvar,
    compute_var_cvar,
    compute_max_drawdown,
    compute_calmar_ratio,
    compute_hit_ratio,
    compute_profit_factor,
    compute_advanced_metrics,
    compute_batch_advanced_metrics,
)


class TestSortinoRatio:
    def test_zero_downside_returns_high_sortino(self):
        """All-positive rewards → sigma_d ≈ 0 → very large Sortino."""
        rewards = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        sortino = compute_sortino_ratio(rewards, tau=0.0)
        assert sortino > 10.0, f"Expected high Sortino for all-positive returns, got {sortino}"

    def test_negative_only_rewards_low_sortino(self):
        """All-negative rewards below tau → Sortino < 0."""
        rewards = np.array([-0.1, -0.2, -0.3], dtype=np.float32)
        sortino = compute_sortino_ratio(rewards, tau=0.0)
        assert sortino < 0.0

    def test_all_positive_rewards_higher_sortino_than_sharpe(self):
        """For all-positive rewards the Sortino denominator sigma_d=0, making it
        much larger than Sharpe (which has a non-zero std denominator)."""
        rewards = np.array([0.1, 0.2, 0.15, 0.12, 0.08], dtype=np.float32)
        mean_r = float(np.mean(rewards))
        std_r = float(np.std(rewards)) + 1e-8
        sharpe = mean_r / std_r
        sortino = compute_sortino_ratio(rewards, tau=0.0)
        # sigma_d = 0 so sortino >> sharpe for all-positive rewards
        assert sortino > sharpe

    def test_tau_shifts_baseline(self):
        """Increasing tau reduces the Sortino ratio."""
        rewards = np.array([0.05] * 20, dtype=np.float32)
        s0 = compute_sortino_ratio(rewards, tau=0.0)
        s1 = compute_sortino_ratio(rewards, tau=0.03)
        assert s0 >= s1, "Higher tau should not increase Sortino"

    def test_output_is_scalar_float(self):
        rewards = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        result = compute_sortino_ratio(rewards)
        assert isinstance(result, float)


class TestVaRCVaR:
    def test_var_quantile_correctness(self):
        """VaR_0.9 on uniform losses [0,1] should be close to 0.9."""
        np.random.seed(42)
        losses = np.random.uniform(0, 1, 100_000).astype(np.float32)
        rewards = -losses
        var = compute_var(rewards, alpha=0.9)
        assert abs(var - 0.9) < 0.02, f"VaR_0.9 should be ~0.9, got {var}"

    def test_cvar_exceeds_var(self):
        """CVaR must be >= VaR for the same alpha (tail average >= quantile)."""
        np.random.seed(1)
        rewards = np.random.randn(5000).astype(np.float32)
        for alpha in (0.9, 0.95, 0.99):
            var = compute_var(rewards, alpha)
            cvar = compute_cvar(rewards, alpha)
            assert cvar >= var - 1e-6, f"CVaR < VaR at alpha={alpha}"

    def test_var_cvar_tuple_consistency(self):
        """compute_var_cvar must return the same values as individual calls."""
        np.random.seed(2)
        rewards = np.random.randn(1000).astype(np.float32)
        var_t, cvar_t = compute_var_cvar(rewards, alpha=0.95)
        assert abs(var_t - compute_var(rewards, 0.95)) < 1e-6
        assert abs(cvar_t - compute_cvar(rewards, 0.95)) < 1e-6

    def test_all_positive_rewards_var_negative(self):
        """All-positive rewards → losses all negative → VaR < 0 (gain scenario)."""
        rewards = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        var = compute_var(rewards, alpha=0.95)
        assert var < 0.0

    def test_cvar_99_ge_cvar_95(self):
        """CVaR_99 should be >= CVaR_95 (higher confidence = worse tail)."""
        np.random.seed(3)
        rewards = np.random.randn(5000).astype(np.float32)
        _, cvar95 = compute_var_cvar(rewards, 0.95)
        _, cvar99 = compute_var_cvar(rewards, 0.99)
        assert cvar99 >= cvar95 - 1e-6


class TestMaxDrawdownCalmar:
    def test_monotone_gains_zero_mdd(self):
        """Strictly increasing PnL has zero drawdown."""
        rewards = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        mdd = compute_max_drawdown(rewards)
        assert mdd == pytest.approx(0.0, abs=1e-7)

    def test_known_drawdown_sequence(self):
        """Rewards: [1, -2, 1] → PnL: [1, -1, 0] → peak=1, trough=-1 → MDD=2."""
        rewards = np.array([1.0, -2.0, 1.0], dtype=np.float32)
        mdd = compute_max_drawdown(rewards)
        assert mdd == pytest.approx(2.0, abs=1e-6)

    def test_mdd_nonnegative(self):
        np.random.seed(10)
        rewards = np.random.randn(200).astype(np.float32)
        assert compute_max_drawdown(rewards) >= 0.0

    def test_calmar_positive_when_net_positive(self):
        """Net-positive trajectory with some drawdown → Calmar > 0."""
        rewards = np.array([0.1, -0.05, 0.2, -0.03, 0.15], dtype=np.float32)
        calmar = compute_calmar_ratio(rewards)
        assert calmar > 0.0

    def test_calmar_negative_when_net_negative(self):
        """Net-negative trajectory → Calmar < 0."""
        rewards = np.array([-0.1, 0.02, -0.15], dtype=np.float32)
        calmar = compute_calmar_ratio(rewards)
        assert calmar < 0.0


class TestProfitFactorHitRatio:
    def test_all_wins_hit_ratio_one(self):
        rewards = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        assert compute_hit_ratio(rewards) == pytest.approx(1.0)

    def test_all_losses_hit_ratio_zero(self):
        rewards = np.array([-0.1, -0.2, -0.3], dtype=np.float32)
        assert compute_hit_ratio(rewards) == pytest.approx(0.0)

    def test_hit_ratio_partial(self):
        """3 wins out of 5 steps → hit ratio = 0.6."""
        rewards = np.array([0.1, -0.2, 0.3, -0.1, 0.05], dtype=np.float32)
        hr = compute_hit_ratio(rewards)
        assert hr == pytest.approx(0.6, abs=1e-6)

    def test_profit_factor_greater_than_one_for_net_positive(self):
        """Gross profit > gross loss → PF > 1."""
        rewards = np.array([2.0, -0.5, 1.0, -0.3], dtype=np.float32)
        pf = compute_profit_factor(rewards)
        assert pf > 1.0

    def test_profit_factor_less_than_one_for_net_negative(self):
        """Gross loss > gross profit → PF < 1."""
        rewards = np.array([-2.0, 0.3, -1.5, 0.1], dtype=np.float32)
        pf = compute_profit_factor(rewards)
        assert pf < 1.0

    def test_profit_factor_all_wins_very_large(self):
        """All positive → denominator ≈ 0 → PF is very large."""
        rewards = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        pf = compute_profit_factor(rewards)
        assert pf > 1e6


class TestAdvancedMetrics:
    def test_compute_advanced_metrics_returns_all_keys(self):
        """compute_advanced_metrics must return all expected keys."""
        rewards = np.random.randn(100).astype(np.float32)
        result = compute_advanced_metrics(rewards)
        expected_keys = {
            "Sortino", "VaR_95", "CVaR_95", "VaR_99", "CVaR_99",
            "MaxDD", "Calmar", "ProfitFactor", "HitRatio",
        }
        assert expected_keys == set(result.keys())

    def test_compute_advanced_metrics_all_floats(self):
        rewards = np.random.randn(50).astype(np.float32)
        result = compute_advanced_metrics(rewards)
        for k, v in result.items():
            assert isinstance(v, float), f"Key {k} is not a float: {type(v)}"

    def test_compute_batch_advanced_metrics_numpy(self):
        """batch helper works with plain numpy (B, T) array."""
        rewards = np.random.randn(4, 200).astype(np.float32)
        result = compute_batch_advanced_metrics(rewards)
        assert "Sortino" in result
        assert "CVaR_99" in result
        assert isinstance(result["HitRatio"], float)

    def test_compute_batch_advanced_metrics_torch(self):
        """batch helper works with PyTorch (B, T) tensor."""
        rewards = torch.randn(4, 200)
        result = compute_batch_advanced_metrics(rewards)
        assert "ProfitFactor" in result
        assert 0.0 <= result["HitRatio"] <= 1.0

    def test_cvar_99_ge_cvar_95_batch(self):
        """Batch CVaR_99 must be >= CVaR_95 on average."""
        np.random.seed(42)
        rewards = np.random.randn(8, 500).astype(np.float32)
        result = compute_batch_advanced_metrics(rewards)
        assert result["CVaR_99"] >= result["CVaR_95"] - 1e-5

    def test_hit_ratio_in_unit_interval(self):
        rewards = np.random.randn(10, 100).astype(np.float32)
        result = compute_batch_advanced_metrics(rewards)
        assert 0.0 <= result["HitRatio"] <= 1.0

    def test_mdd_nonnegative_batch(self):
        rewards = np.random.randn(5, 300).astype(np.float32)
        result = compute_batch_advanced_metrics(rewards)
        assert result["MaxDD"] >= 0.0


# =============================================================================
# TRAINING CURVE PLOTTING TESTS
# =============================================================================

from src.training.training_pipeline import plot_training_curves


class TestPlotTrainingCurves:
    def test_creates_png_file(self, tmp_path):
        """plot_training_curves saves a valid PNG to the given path."""
        history = {
            "step_loss":  [(100, 1.2), (200, 1.0), (300, 0.9)],
            "step_lr":    [(100, 1e-4), (200, 9e-5), (300, 8e-5)],
            "epoch_loss": [(300, 1.0)],
            "epoch_acc":  [(1, 0.55)],
        }
        save_path = tmp_path / "curves.png"
        plot_training_curves(history, save_path)
        assert save_path.exists(), "training_curves PNG was not created."
        assert save_path.stat().st_size > 0, "training_curves PNG is empty."

    def test_empty_history_does_not_raise(self, tmp_path):
        """Calling with empty history must not crash (panels stay blank)."""
        history = {"step_loss": [], "step_lr": [], "epoch_loss": [], "epoch_acc": []}
        save_path = tmp_path / "curves_empty.png"
        plot_training_curves(history, save_path)
        assert save_path.exists()

    def test_multi_epoch_history(self, tmp_path):
        """Multiple epochs of data plot without error."""
        history = {
            "step_loss":  [(i * 50, 2.0 / (1 + i * 0.1)) for i in range(1, 21)],
            "step_lr":    [(i * 50, 1e-4 * (0.99 ** i)) for i in range(1, 21)],
            "epoch_loss": [(i * 100, 1.5 / (1 + i * 0.2)) for i in range(1, 6)],
            "epoch_acc":  [(i, 0.3 + i * 0.08) for i in range(1, 6)],
        }
        save_path = tmp_path / "curves_multi.png"
        plot_training_curves(history, save_path)
        assert save_path.exists()


# =============================================================================
# DATASET VISUALISATION TESTS
# =============================================================================

from src.data.trajectories_generator import (
    plot_lob_features,
    plot_episode_cumulative_pnl,
    plot_distributions,
    detect_stock_boundaries,
    split_by_stock,
    POLICIES,
)


@pytest.fixture
def dummy_x_data() -> np.ndarray:
    """Minimal synthetic LOB matrix (T=200, 40 features) for viz tests."""
    np.random.seed(55)
    X = np.random.randn(200, 40).astype(np.float32)
    X[:, 0] = np.abs(X[:, 0]) + 100.0   # ask1 > 0
    X[:, 2] = X[:, 0] - 0.05             # bid1 < ask1
    return X


@pytest.fixture
def dummy_trajectories_for_viz() -> list:
    """Small synthetic trajectory list covering all 6 policies."""
    np.random.seed(77)
    trajectories = []
    for policy in POLICIES:
        T = 60
        traj = {
            "states":       torch.randn(T, 41, dtype=torch.float32),
            "actions":      torch.randint(0, 3, (T,), dtype=torch.long),
            "rewards":      torch.randn(T, dtype=torch.float32) * 0.01,
            "rtg":          torch.randn(T, 1, dtype=torch.float32),
            "timesteps":    torch.arange(T, dtype=torch.long),
            "policy":       policy,
            "total_return": float(torch.randn(1).item()),
        }
        trajectories.append(traj)
    return trajectories


class TestPlotLobFeatures:
    def test_creates_all_three_png_files(self, tmp_path, dummy_x_data):
        """plot_lob_features must create three PNG files."""
        plot_lob_features(dummy_x_data, "Test", str(tmp_path))
        expected = [
            "test_lob_price_series.png",
            "test_lob_feature_distributions.png",
            "test_lob_autocorrelation.png",
        ]
        for fname in expected:
            fpath = tmp_path / fname
            assert fpath.exists(), f"Expected file not created: {fname}"
            assert fpath.stat().st_size > 0, f"File is empty: {fname}"

    def test_name_prefix_lower_cased(self, tmp_path, dummy_x_data):
        """File names must use the lowercase of the name argument."""
        plot_lob_features(dummy_x_data, "Train", str(tmp_path))
        assert (tmp_path / "train_lob_price_series.png").exists()


class TestPlotEpisodeCumulativePnL:
    def test_creates_png_file(self, tmp_path, dummy_trajectories_for_viz):
        plot_episode_cumulative_pnl(
            dummy_trajectories_for_viz, "Train", str(tmp_path)
        )
        assert (tmp_path / "train_episode_pnl_examples.png").exists()

    def test_handles_empty_trajectories_gracefully(self, tmp_path):
        """Empty list must not crash, even if no file is produced."""
        try:
            plot_episode_cumulative_pnl([], "Test", str(tmp_path))
        except Exception as exc:
            pytest.fail(f"plot_episode_cumulative_pnl raised {exc!r} on empty input")


class TestPlotDistributions:
    def test_creates_all_expected_files(self, tmp_path, dummy_trajectories_for_viz):
        plot_distributions(dummy_trajectories_for_viz, "Train", str(tmp_path))
        expected_files = [
            "train_policy_returns.png",
            "train_rtg_distribution.png",
            "train_action_distributions.png",
            "train_episode_pnl_examples.png",
        ]
        for fname in expected_files:
            fpath = tmp_path / fname
            assert fpath.exists(), f"Expected file not found: {fname}"


# =============================================================================
# STOCK BOUNDARY DETECTION TESTS
# =============================================================================

def _make_multi_stock_lob(n_stocks: int = 3, events_per_stock: int = 200) -> np.ndarray:
    """Synthetic concatenated LOB matrix mimicking FI-2010 multi-stock layout."""
    segments = []
    np.random.seed(42)
    for s in range(n_stocks):
        base_ask = 100.0 + s * 50.0
        data = np.random.randn(events_per_stock, 40).astype(np.float32) * 0.01
        data[:, 0] = base_ask + np.cumsum(np.random.randn(events_per_stock) * 0.001)
        data[:, 2] = data[:, 0] - 0.05
        segments.append(data)
    return np.concatenate(segments, axis=0)


class TestDetectStockBoundaries:
    def test_detects_correct_number_of_boundaries(self):
        X = _make_multi_stock_lob(n_stocks=5, events_per_stock=300)
        boundaries = detect_stock_boundaries(X, n_stocks=5)
        assert boundaries[0] == 0
        assert boundaries[-1] == len(X)
        assert len(boundaries) == 5 + 1

    def test_single_stock_has_no_interior_boundaries(self):
        X = _make_multi_stock_lob(n_stocks=1, events_per_stock=500)
        boundaries = detect_stock_boundaries(X, n_stocks=1)
        assert boundaries == [0, 500]

    def test_boundary_positions_close_to_expected(self):
        X = _make_multi_stock_lob(n_stocks=3, events_per_stock=200)
        boundaries = detect_stock_boundaries(X, n_stocks=3)
        assert abs(boundaries[1] - 200) <= 1
        assert abs(boundaries[2] - 400) <= 1

    def test_split_by_stock_returns_correct_count(self):
        X = _make_multi_stock_lob(n_stocks=4, events_per_stock=150)
        y = np.random.randn(len(X), 5).astype(np.float32)
        segments = split_by_stock(X, y, n_stocks=4)
        assert len(segments) == 4

    def test_split_by_stock_preserves_total_rows(self):
        X = _make_multi_stock_lob(n_stocks=3, events_per_stock=200)
        segments = split_by_stock(X, n_stocks=3)
        total = sum(len(sX) for sX, _ in segments)
        assert total == len(X)

    def test_split_by_stock_no_y(self):
        X = _make_multi_stock_lob(n_stocks=2, events_per_stock=100)
        segments = split_by_stock(X, n_stocks=2)
        assert all(sy is None for _, sy in segments)


# =============================================================================
# DT_VIZ UNIT TESTS (new plots)
# =============================================================================

import pandas as pd
from src.evaluations.dt_viz import (
    compute_financial_metrics,
    plot_drawdown_curves,
    plot_action_distribution_by_rtg,
    plot_advanced_metrics_comparison,
    plot_inference_time,
)


class TestComputeFinancialMetrics:
    def test_returns_all_required_keys(self):
        """compute_financial_metrics must include PnL, Sharpe and all advanced keys."""
        rewards = torch.randn(4, 200)
        result = compute_financial_metrics(rewards)
        required = {
            "PnL", "Sharpe", "Sortino", "VaR_95", "CVaR_95",
            "VaR_99", "CVaR_99", "MaxDD", "Calmar", "ProfitFactor", "HitRatio",
        }
        missing = required - set(result.keys())
        assert not missing, f"Missing keys in compute_financial_metrics: {missing}"

    def test_sharpe_positive_for_strictly_positive_rewards(self):
        rewards = torch.ones(2, 50) * 0.01
        m = compute_financial_metrics(rewards)
        assert m["Sharpe"] > 0.0

    def test_pnl_additive_sanity(self):
        """PnL should equal sum of step rewards (averaged over batch)."""
        rewards = torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], dtype=torch.float32)
        m = compute_financial_metrics(rewards)
        assert m["PnL"] == pytest.approx(0.6, abs=1e-5)


class TestNewPlots:
    def test_plot_drawdown_curves_creates_png(self, tmp_path):
        rewards_dict = {
            "Oracle": np.array([0.1, 0.1, -0.05, 0.1]),
            "DT (R=0.5)": np.array([0.02, -0.01, 0.01, 0.02]),
        }
        save_path = tmp_path / "drawdown.png"
        plot_drawdown_curves(rewards_dict, save_path)
        assert save_path.exists()

    def test_plot_action_distribution_by_rtg_creates_png(self, tmp_path):
        actions = {
            "R=0.0": np.array([0, 1, 2, 0, 0, 2, 1]),
            "R=0.5": np.array([2, 2, 1, 0, 2, 2, 2]),
        }
        save_path = tmp_path / "action_dist.png"
        plot_action_distribution_by_rtg(actions, save_path)
        assert save_path.exists()

    def test_plot_action_distribution_empty_does_not_crash(self, tmp_path):
        save_path = tmp_path / "empty_action_dist.png"
        plot_action_distribution_by_rtg({}, save_path)

    def test_plot_advanced_metrics_comparison_creates_png(self, tmp_path):
        data = {
            "Oracle":     {"PnL": 0.5, "Sharpe": 1.2, "Sortino": 2.0, "CVaR_95": 0.01,
                           "CVaR_99": 0.02, "MaxDD": 0.05, "Calmar": 5.0,
                           "ProfitFactor": 2.5, "HitRatio": 0.65},
            "DT (R=0.5)": {"PnL": 0.1, "Sharpe": 0.5, "Sortino": 0.8, "CVaR_95": 0.03,
                           "CVaR_99": 0.05, "MaxDD": 0.1, "Calmar": 1.0,
                           "ProfitFactor": 1.2, "HitRatio": 0.52},
        }
        df = pd.DataFrame.from_dict(data, orient="index")
        save_path = tmp_path / "advanced.png"
        plot_advanced_metrics_comparison(df, save_path)
        assert save_path.exists()

    def test_plot_inference_time_creates_png(self, tmp_path):
        times = {"Oracle": 0.002, "DT (R=0.0)": 0.15, "DT (R=0.5)": 0.16}
        save_path = tmp_path / "inference.png"
        plot_inference_time(times, save_path)
        assert save_path.exists()