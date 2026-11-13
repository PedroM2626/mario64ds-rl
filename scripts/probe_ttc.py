"""How high does a BTR checkpoint get on Tick Tock Clock (ds8), with no training on it?

Runs the greedy policy for N episodes with the ref10 episode rules and logs, per episode, the
start height, the highest height, the final height, coins, decisions and how the episode ended.

Usage:
    python scripts/probe_ttc.py --model models/btr_goal20/step_1000005.pt --episodes 10 --out results/ttc_probe_btr_goal20.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--track", default="ds8")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--deterministic", action="store_true", help="fixed quantile levels: the policy is then a deterministic function of the observation")
    args = ap.parse_args()

    from scripts.train_btr import ref_env_kwargs
    from src.btr import BTRAgent, checkpoint_actions
    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path

    n_act = checkpoint_actions(os.path.join(ROOT_DIR, args.model))
    action_set = {10: "ref10", 13: "ref13"}[n_act]
    agent = BTRAgent(n_act, device="cpu")
    agent.load(os.path.join(ROOT_DIR, args.model))
    env = Mario64DSEnv(rom_path=rom_path(), state_path=savestate_path(args.track), max_steps=args.max_steps,
                       track_id=args.track, **ref_env_kwargs(args.track, action_set=action_set))
    rows = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        dq = deque([np.zeros((84, 84), np.uint8)] * 3 + [obs[:, :, 0]], maxlen=4)
        c0, y0 = env.ram.get_coins(), float(env.ram.get_pos_3d()[1])
        ymax, end, info, t = y0, "horizon", {}, 0
        for t in range(args.max_steps):
            a = int(agent.act(np.stack(list(dq))[None], noisy=False, fixed_taus=args.deterministic)[0])
            obs, _, term, _, info = env.step(a)
            dq.append(obs[:, :, 0])
            ymax = max(ymax, float(env.ram.get_pos_3d()[1]))
            if term:
                end = "star" if info.get("success") else "died"
                break
        rows.append({"episode": ep, "start_y": round(y0, 1), "max_y": round(ymax, 1), "final_y": round(float(env.ram.get_pos_3d()[1]), 1),
                     "route_s": round(float(info.get("route_progress") or 0.0), 1),
                     "coins": int(env.ram.get_coins() - c0), "decisions": t + 1, "end": end})
        print(rows[-1], flush=True)
    env.close()
    if args.out:
        with open(os.path.join(ROOT_DIR, args.out), "w", encoding="utf-8") as fh:
            json.dump({"model": args.model, "track": args.track, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
