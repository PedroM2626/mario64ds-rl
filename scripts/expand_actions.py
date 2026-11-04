"""Grow a legacy6 PPO checkpoint into a compound12 one without forgetting what it knows.

compound12 keeps the six legacy actions in the same indices (src/env.py
ACTION_SETS), so every network weight is copied unchanged and only the action
head gains six rows. Each new action starts from the row of the legacy action
it extends (up+left/up+right/up+Y from up, the jumps from B) with its bias
lowered by --bias-offset, so the expanded policy initially behaves like the
original one and only explores the combinations occasionally.

Usage:
    python scripts/expand_actions.py --model models/curriculum_flow25_r3_best.zip \
        --out models/curriculum_flow25_r3_c12.zip
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# compound12 index -> legacy action it extends (noop, left, right, up, down, B = 0..5)
BASE = {6: 3, 7: 3, 8: 5, 9: 5, 10: 5, 11: 3}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bias-offset", type=float, default=2.0)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    m6 = PPO.load(os.path.join(ROOT_DIR, args.model), device="cpu")
    if int(m6.action_space.n) != 6:
        raise SystemExit(f"expected a 6-action checkpoint, got {m6.action_space.n}")

    class Spaces(gym.Env):
        observation_space = m6.observation_space
        action_space = gym.spaces.Discrete(12)

        def reset(self, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    m12 = PPO(m6.policy_class, DummyVecEnv([Spaces]), learning_rate=m6.learning_rate, n_steps=m6.n_steps,
              batch_size=m6.batch_size, n_epochs=m6.n_epochs, gamma=m6.gamma, gae_lambda=m6.gae_lambda,
              clip_range=m6.clip_range, ent_coef=m6.ent_coef, vf_coef=m6.vf_coef,
              max_grad_norm=m6.max_grad_norm, policy_kwargs=m6.policy_kwargs, device="cpu")
    sd6, sd12 = m6.policy.state_dict(), m12.policy.state_dict()
    for k in sd12:
        if k.startswith("action_net."):
            new = torch.empty_like(sd12[k])
            new[:6] = sd6[k]
            for i, b in BASE.items():
                new[i] = sd6[k][b]
                if k.endswith("bias"):
                    new[i] -= args.bias_offset
            sd12[k] = new
        else:
            sd12[k] = sd6[k]
    m12.policy.load_state_dict(sd12)
    m12.num_timesteps = m6.num_timesteps

    # The six legacy logits must be unchanged.
    obs = torch.as_tensor(np.stack([m6.observation_space.sample() for _ in range(8)])).float()
    with torch.no_grad():
        l6 = m6.policy.action_net(m6.policy.mlp_extractor.forward_actor(m6.policy.extract_features(obs, m6.policy.pi_features_extractor)))
        l12 = m12.policy.action_net(m12.policy.mlp_extractor.forward_actor(m12.policy.extract_features(obs, m12.policy.pi_features_extractor)))
    err = float((l12[:, :6] - l6).abs().max())
    p_new = float(torch.softmax(l12, -1)[:, 6:].sum(-1).mean())
    m12.save(os.path.join(ROOT_DIR, args.out))
    print(f"saved {args.out}: legacy logits max diff {err:.2e}, mean probability of the 6 new actions {p_new:.3f}")


if __name__ == "__main__":
    main()
