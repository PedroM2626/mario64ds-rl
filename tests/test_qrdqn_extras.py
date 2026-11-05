"""HoldQRDQN, DemoRefreshCallback and demo retention, on a tiny fake env (no emulator)."""
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest

from src.qrdqn_extras import DemoRefreshCallback, HoldQRDQN


class Tiny(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (4,), np.float32)
    action_space = gym.spaces.Discrete(12)

    def reset(self, seed=None, options=None):
        return np.zeros(4, np.float32), {}

    def step(self, a):
        return np.zeros(4, np.float32), 0.0, False, False, {}


def _model(hold_max, n_envs=1):
    from stable_baselines3.common.vec_env import DummyVecEnv
    return HoldQRDQN("MlpPolicy", DummyVecEnv([Tiny] * n_envs), hold_max=hold_max, buffer_size=2000,
                     learning_starts=10, device="cpu", seed=0)


def _runs(actions):
    out, n = [], 1
    for a, b in zip(actions, actions[1:]):
        if a == b:
            n += 1
        else:
            out.append(n)
            n = 1
    return out + [n]


def test_hold_exploration_repeats_actions_and_keeps_the_random_share():
    obs = np.zeros((1, 4), np.float32)
    shares = {}
    for hold in (1, 10):
        m = _model(hold)
        m.exploration_rate = 0.5
        greedy = int(m.policy.predict(obs, deterministic=True)[0][0])
        np.random.seed(0)
        acts = [int(m.predict(obs, deterministic=False)[0][0]) for _ in range(4000)]
        shares[hold] = np.mean([a != greedy for a in acts])
        if hold == 10:
            assert max(_runs(acts)) >= 8          # a held random action is repeated
        else:
            assert np.mean(_runs(acts)) < 3        # plain epsilon-greedy: short runs
    # the expected share of non-greedy decisions stays near epsilon (0.5 * 11/12) in both modes
    assert shares[1] == pytest.approx(0.46, abs=0.06)
    assert shares[10] == pytest.approx(0.46, abs=0.06)


def test_deterministic_predict_ignores_holds():
    m = _model(10)
    m.exploration_rate = 1.0
    obs = np.zeros((1, 4), np.float32)
    acts = {int(m.predict(obs, deterministic=True)[0][0]) for _ in range(50)}
    assert len(acts) == 1


def test_prefilled_demos_survive_learn():
    from src.demos import prefill_replay_buffer
    m = _model(1)
    t = {"obs": np.zeros((1, 4), np.float32), "next_obs": np.zeros((1, 4), np.float32),
         "action": np.array([[3]]), "reward": np.array([1.0]), "done": np.array([False]), "info": {}, "exact": True}
    assert prefill_replay_buffer(m, [t] * 25) == 25
    m.learn(total_timesteps=100)
    assert m.replay_buffer.size() >= 25 + 100 - 5


def test_refresh_callback_readds_on_schedule():
    calls = []
    cb = DemoRefreshCallback(["t"], every=100, prefill_fn=lambda model, tr: calls.append(model))
    cb.model = MagicMock()
    for step in (50, 99, 100, 150, 199, 200, 350):
        cb.num_timesteps = step
        cb._on_step()
    assert len(calls) == 3                      # at 100, 200 and 350 (next due 300)
