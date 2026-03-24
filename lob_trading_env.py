"""
Gymnasium environment for Limit Order Book (LOB) trading.

Follows the DeepLOB paper setting:
  - Observation: window of W consecutive LOB snapshots (each with 40 raw features)
                 plus the agent's current inventory position.
  - Action:      desired position in {-1 (short), 0 (flat), +1 (long)}.
  - Reward:      position * delta_mid_price  -  transaction_cost * |delta_position|.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


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
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        lob_data: np.ndarray,
        window_size: int = 100,
        transaction_cost: float = 0.0,
        episode_length: int | None = None,
    ):
        super().__init__()

        if lob_data.ndim != 2 or lob_data.shape[1] != 40:
            raise ValueError(f"Expected (T, 40) array, got {lob_data.shape}")
        if lob_data.shape[0] <= window_size:
            raise ValueError(
                f"Data length ({lob_data.shape[0]}) must exceed "
                f"window_size ({window_size})"
            )

        self.lob_data = lob_data.astype(np.float32)
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.episode_length = episode_length

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
    # Core Gymnasium API
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict:
        start = self._current_step - self.window_size
        lob_window = self.lob_data[start : self._current_step]
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

        reward = float(
            new_position * price_change
            - self.transaction_cost * delta_pos
        )

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
        }
        return obs, reward, terminated, truncated, info
