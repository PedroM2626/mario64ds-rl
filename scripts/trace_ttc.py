"""Per-decision position trace of a BTR checkpoint on ds8 (greedy), to see where it leaves the route.

Usage:
    python scripts/trace_ttc.py --model models/btr_ttc_s3/final.pt --episodes 3 --out results/ttc_trace_btr_ttc_s3_final.json

Each episode stops at the horizon, on death/star, or after --idle decisions without new route progress
(default 800; 0 = never). The trace holds action, position and route arc length for every decision.
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
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--idle", type=int, default=800)
    ap.add_argument("--out", required=True)
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
    episodes = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        dq = deque([np.zeros((84, 84), np.uint8)] * 3 + [obs[:, :, 0]], maxlen=4)
        trace, best, last_gain, end = [], -1.0, 0, "horizon"
        for t in range(args.max_steps):
            a = int(agent.act(np.stack(list(dq))[None], noisy=False, fixed_taus=args.deterministic)[0])
            obs, _, term, _, info = env.step(a)
            dq.append(obs[:, :, 0])
            pos = env.ram.get_pos_3d()
            s = float(info.get("route_progress") or 0.0)
            if s > best:
                best, last_gain = s, t
            trace.append({"d": t + 1, "a": a, "pos": [round(float(v), 1) for v in pos], "s": round(s, 1)})
            if term:
                end = "star" if info.get("success") else "died"
                break
            if args.idle and t - last_gain >= args.idle:
                end = "idle"
                break
        episodes.append({"episode": ep, "end": end, "decisions": len(trace), "trace": trace})
        print(ep, end, len(trace), round(best, 1), flush=True)
    env.close()
    with open(os.path.join(ROOT_DIR, args.out), "w", encoding="utf-8") as fh:
        json.dump({"model": args.model, "track": args.track, "episodes": episodes}, fh)


if __name__ == "__main__":
    main()
