"""Frame-stack / human-action pairs for src.ppo_extras.BCAnchor.

The continuous replay of a human run from the track start, plus, for each savestate (route checkpoints
`step-NNNNNN.dst`, dense `dense-NNNN.dst`; the number is the decision), the next --window human
actions from a fresh stack (training resets have no frame history), as scripts/clone_demo.py does.

Usage:
    python scripts/build_bc_set.py --demo runs/human-demonstrations/<run>/completed/attempt-000001 \
        --states data/curriculum/ds3_human_v1 data/curriculum/ds3_human_dense --out data/curriculum/ds3_human_bc.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import deque

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np  # noqa: E402


def fresh_stack(obs):
    return deque([np.zeros((84, 84), np.uint8)] * 3 + [obs[:, :, 0]], maxlen=4)


def collect(env, actions, state):
    env.state_path = state
    obs, _ = env.reset()
    dq = fresh_stack(obs)
    X, Y = [], []
    for a in actions:
        X.append(np.stack(list(dq)))
        Y.append(int(a))
        obs, _, term, _, _ = env.step(int(a))
        dq.append(obs[:, :, 0])
        if term:
            break
    return X, Y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--states", nargs="+", required=True, help="directories of savestates named *-<decision>.dst")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from src.demos import load_demo
    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path
    demo = load_demo(os.path.join(ROOT_DIR, args.demo))
    actions = [int(a) for a in demo["actions"]]
    env = Mario64DSEnv(rom_path=rom_path(), state_path=savestate_path("ds3"), max_steps=2000, flow_weight=0.0,
                       dist_reward_enabled=False, track_id="ds3", action_set="compound12", black_screen_death=False)
    X, Y = collect(env, actions, savestate_path("ds3"))
    n_cont = len(Y)
    for d in args.states:
        for path in sorted(glob.glob(os.path.join(ROOT_DIR, d, "*.dst"))):
            t = int(re.search(r"-(\d+)\.dst$", path).group(1))
            if t >= len(actions):
                continue
            x, y = collect(env, actions[t:t + args.window], path)
            X += x
            Y += y
    env.close()
    X, Y = np.stack(X), np.asarray(Y)
    os.makedirs(os.path.dirname(os.path.join(ROOT_DIR, args.out)), exist_ok=True)
    np.savez_compressed(os.path.join(ROOT_DIR, args.out), X=X, Y=Y)
    print(f"{len(Y)} samples ({n_cont} continuous); actions {np.bincount(Y, minlength=12).tolist()} -> {args.out}")


if __name__ == "__main__":
    main()
