"""ValueWarmup on a tiny fake env (no emulator)."""
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.ppo_extras import ValueWarmup


class Tiny(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (4,), np.float32)
    action_space = gym.spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        return np.random.uniform(-1, 1, 4).astype(np.float32), {}

    def step(self, a):
        return np.random.uniform(-1, 1, 4).astype(np.float32), float(a), False, False, {}


def _model():
    return PPO("MlpPolicy", DummyVecEnv([Tiny]), n_steps=64, batch_size=32, n_epochs=2, device="cpu", seed=0)


def _snapshot(m):
    return {k: v.detach().clone() for k, v in m.policy.state_dict().items()}


def _changed(a, b, prefix):
    return any(not torch.equal(a[k], b[k]) for k in a if k.startswith(prefix))


def test_only_the_value_head_trains_during_warmup_and_everything_after():
    m = _model()
    before = _snapshot(m)
    m.learn(256, callback=ValueWarmup(10_000))        # warm-up longer than the run
    mid = _snapshot(m)
    assert _changed(before, mid, "value_net") or _changed(before, mid, "mlp_extractor.value_net")
    assert not _changed(before, mid, "action_net")
    assert not _changed(before, mid, "mlp_extractor.policy_net")
    m.learn(256, callback=ValueWarmup(64), reset_num_timesteps=False)   # then thaws
    after = _snapshot(m)
    assert _changed(mid, after, "action_net")
    assert all(p.requires_grad for p in m.policy.parameters())


def test_zero_warmup_trains_everything():
    m = _model()
    before = _snapshot(m)
    m.learn(128, callback=ValueWarmup(0))
    assert _changed(before, _snapshot(m), "action_net")
