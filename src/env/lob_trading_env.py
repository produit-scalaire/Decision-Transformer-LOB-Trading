"""
Gymnasium environment for Limit Order Book (LOB) trading.

Follows the DeepLOB paper setting:
  - Observation: window of W consecutive LOB snapshots (each with 40 raw features)
                 plus the agent's current inventory position.
  - Action:      desired position in {-1 (short), 0 (flat), +1 (long)}.
  - Reward:      position * delta_mid_price  -  transaction_cost * |delta_position|,
                 optionally with additive shaping (drawdown / variance / time-in-market).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


_REWARD_TYPES = frozenset({"mid_price", "shaped"})
_STATE_REPRESENTATIONS = frozenset({"raw", "log_returns", "bps"})
# Per FI-2010 / DeepLOB layout: level k uses columns 4*k+0 .. 4*k+3 for ask_p, ask_v, bid_p, bid_v.
LOB_PRICE_COL_INDICES = tuple(4 * k + j for k in range(10) for j in (0, 2))


class LOBTradingEnv(gym.Env):
    """
    Trading environment on FI-2010 raw LOB data (40 features per snapshot).

    Parameters
    ----------
    lob_data : np.ndarray, shape (T, 40)
        Matrix of LOB snapshots (z-score normalised).
        Column layout per level i (i = 1 … 10):
            [ask_price_i, ask_volume_i, bid_price_i, bid_volume_i]
    window_size : int
        Number of past snapshots in each observation (default 100, as in DeepLOB).
    transaction_cost : float
        Cost per unit of absolute position change.
    episode_length : int or None
        Fixed episode length (random start). If None the full dataset is one episode.
    reward_type : str
        ``"mid_price"`` — PnL from mid moves and transaction costs only.
        ``"shaped"`` — same base PnL minus penalties (drawdown from peak equity on base PnL,
        rolling variance of base step returns, and |position| for time in market).
    drawdown_coef : float
        Weight on (peak_base_equity - current_base_equity) when ``reward_type=="shaped"``.
    variance_coef : float
        Weight on rolling variance of recent base step rewards when shaped.
    time_in_market_coef : float
        Weight on |position| after the action when shaped.
    variance_window : int
        Max history length for the rolling variance term (>= 1).
    state_representation : str
        ``"raw"`` — return LOB windows as stored (e.g. z-scored snapshots).
        ``"log_returns"`` — replace each ask/bid **price** level with
        ``log(p_t + price_offset) - log(p_{t-1} + price_offset)`` (first row of the
        window uses the previous global tick when available, else zeros).
        ``"bps"`` — relative change in basis points:
        ``(p_t - p_{t-1}) / (|p_{t-1}| + price_offset) * 1e4``.
        Volume columns are always left unchanged.
    price_offset : float
        Non-negative shift for log/bps transforms (stabilizes z-scored or small prices).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        lob_data: np.ndarray,
        window_size: int = 100,
        transaction_cost: float = 0.0,
        episode_length: int | None = None,
        reward_type: str = "mid_price",
        drawdown_coef: float = 0.0,
        variance_coef: float = 0.0,
        time_in_market_coef: float = 0.0,
        variance_window: int = 20,
        state_representation: str = "raw",
        price_offset: float = 10.0,
    ):
        super().__init__()

        if lob_data.ndim != 2 or lob_data.shape[1] != 40:
            raise ValueError(f"Expected (T, 40) array, got {lob_data.shape}")
        if lob_data.shape[0] <= window_size:
            raise ValueError(
                f"Data length ({lob_data.shape[0]}) must exceed "
                f"window_size ({window_size})"
            )
        if reward_type not in _REWARD_TYPES:
            raise ValueError(
                f"reward_type must be one of {sorted(_REWARD_TYPES)}, got {reward_type!r}"
            )
        if variance_window < 1:
            raise ValueError("variance_window must be >= 1")
        if state_representation not in _STATE_REPRESENTATIONS:
            raise ValueError(
                f"state_representation must be one of {sorted(_STATE_REPRESENTATIONS)}, "
                f"got {state_representation!r}"
            )
        if price_offset < 0.0:
            raise ValueError("price_offset must be non-negative")

        self.lob_data = lob_data.astype(np.float32)
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.episode_length = episode_length
        self.reward_type = reward_type
        self.drawdown_coef = float(drawdown_coef)
        self.variance_coef = float(variance_coef)
        self.time_in_market_coef = float(time_in_market_coef)
        self.variance_window = int(variance_window)
        self.state_representation = state_representation
        self.price_offset = float(price_offset)

        # Mid-price: (best_ask + best_bid) / 2  (columns 0 and 2)
        self.mid_prices = (self.lob_data[:, 0] + self.lob_data[:, 2]) / 2.0

        self.n_features = 40
        self.n_timesteps = lob_data.shape[0]

        # 3 discrete actions: 0 -> short (-1), 1 -> flat (0), 2 -> long (+1)
        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Dict(
            {
                "lob_window": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.window_size, self.n_features),
                    dtype=np.float32,
                ),
                "position": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

        self._current_step: int = 0
        self._position: int = 0
        self._start_step: int = 0
        self._end_step: int = 0

        # Shaped reward: equity and risk stats use base (mid-price) step PnL only.
        self._equity_base: float = 0.0
        self._peak_equity: float = 0.0
        self._base_reward_buf: list[float] = []

    # ------------------------------------------------------------------
    # Action <-> position helpers
    # ------------------------------------------------------------------

    @staticmethod
    def action_to_position(action: int) -> int:
        """0 -> -1 (short), 1 -> 0 (flat), 2 -> +1 (long)."""
        return action - 1

    @staticmethod
    def position_to_action(position: int) -> int:
        """-1 -> 0, 0 -> 1, +1 -> 2."""
        return position + 1

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _stationary_price_features(
        self,
        window: np.ndarray,
        prev_row: np.ndarray | None,
    ) -> np.ndarray:
        """Return a copy of ``window`` with price columns replaced by temporal transforms."""
        out = window.astype(np.float32, copy=True)
        cols = LOB_PRICE_COL_INDICES
        w = window.astype(np.float64, copy=False)
        c = np.asarray(cols, dtype=int)

        if prev_row is None:
            out[0, c] = 0.0
            if window.shape[0] < 2:
                return out
            curr = w[1:, c]
            prev = w[:-1, c]
        else:
            p0 = prev_row.astype(np.float64, copy=False)[c].reshape(1, -1)
            curr = w[:, c]
            prev = p0 if w.shape[0] == 1 else np.vstack([p0, w[:-1, c]])

        off = self.price_offset
        if self.state_representation == "log_returns":
            lc = np.log(np.maximum(curr + off, 1e-8))
            lp = np.log(np.maximum(prev + off, 1e-8))
            transformed = lc - lp
        else:  # bps
            denom = np.maximum(np.abs(prev) + off, 1e-8)
            transformed = (curr - prev) / denom * 10000.0

        if prev_row is None:
            out[1:, c] = transformed.astype(np.float32)
        else:
            out[:, c] = transformed.astype(np.float32)
        return out

    def _transform_lob_window(self, start: int, lob_window: np.ndarray) -> np.ndarray:
        if self.state_representation == "raw":
            return lob_window
        prev = self.lob_data[start - 1] if start > 0 else None
        return self._stationary_price_features(lob_window, prev)

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict:
        start = self._current_step - self.window_size
        lob_window = self.lob_data[start : self._current_step]
        lob_window = self._transform_lob_window(start, lob_window)
        return {
            "lob_window": lob_window,
            "position": np.array([self._position], dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.episode_length is not None:
            max_start = max(
                self.window_size,
                self.n_timesteps - self.episode_length - 1,
            )
            self._start_step = int(
                self.np_random.integers(self.window_size, max_start + 1)
            )
            self._end_step = min(
                self._start_step + self.episode_length,
                self.n_timesteps - 1,
            )
        else:
            self._start_step = self.window_size
            self._end_step = self.n_timesteps - 1

        self._current_step = self._start_step
        self._position = 0
        self._equity_base = 0.0
        self._peak_equity = 0.0
        self._base_reward_buf = []

        return self._get_obs(), {
            "mid_price": float(self.mid_prices[self._current_step]),
        }

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        new_position = self.action_to_position(int(action))
        delta_pos = abs(new_position - self._position)

        price_now = self.mid_prices[self._current_step]
        price_next = self.mid_prices[self._current_step + 1]
        price_change = price_next - price_now

        base_reward = float(
            new_position * price_change - self.transaction_cost * delta_pos
        )

        if self.reward_type == "mid_price":
            reward = base_reward
            drawdown_pen = 0.0
            var_pen = 0.0
            time_pen = 0.0
        else:
            self._equity_base += base_reward
            self._peak_equity = max(self._peak_equity, self._equity_base)
            drawdown = self._peak_equity - self._equity_base
            drawdown_pen = self.drawdown_coef * drawdown

            self._base_reward_buf.append(base_reward)
            if len(self._base_reward_buf) > self.variance_window:
                self._base_reward_buf.pop(0)
            step_var = float(np.var(self._base_reward_buf))
            var_pen = self.variance_coef * step_var

            time_pen = self.time_in_market_coef * abs(new_position)
            reward = base_reward - drawdown_pen - var_pen - time_pen

        self._position = new_position
        self._current_step += 1

        terminated = self._current_step >= self._end_step
        truncated = False

        obs = self._get_obs()
        info = {
            "mid_price": float(self.mid_prices[self._current_step]),
            "position": self._position,
            "price_change": float(price_change),
            "transaction_cost_paid": float(self.transaction_cost * delta_pos),
            "base_reward": base_reward,
            "reward_shaping_penalty": float(drawdown_pen + var_pen + time_pen),
        }
        return obs, reward, terminated, truncated, info
