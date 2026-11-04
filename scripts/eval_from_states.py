"""Greedy PPO from a set of savestates: does it get past a hard spot?

For each savestate: run the deterministic policy for --max-steps decisions and
report how the episode ended (finish / died / horizon), coins gained (RAM
counter) and the closest distance to the finish. Used to test whether training
from savestates just before a hard spot teaches the agent to pass it.

Usage:
    python scripts/eval_from_states.py --model models/curriculum_flow25_r3_best.zip \
        --states "data/curriculum/ds3_gap/*.dst"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--states", required=True, help="glob of savestates")
    ap.add_argument("--track", default="ds3")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--goal-pos", default=None, help="x,y,z of an intermediate goal (as in train_curriculum)")
    ap.add_argument("--goal-radius", type=float, default=600.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from scripts.eval_finish_line import make_policy
    from src.env import Mario64DSEnv
    from src.envs import rom_path
    from src.ram_state import TRACK_GOALS
    goal = TRACK_GOALS[args.track]["target_pos"]
    from stable_baselines3 import PPO
    paths = sorted(glob.glob(os.path.join(ROOT_DIR, args.states)))
    n_actions = int(PPO.load(os.path.join(ROOT_DIR, args.model), device="cpu").action_space.n)
    target = goal if not args.goal_pos else np.array([float(v) for v in args.goal_pos.split(",")])
    env = Mario64DSEnv(rom_path=rom_path(), state_path=paths[0], max_steps=args.max_steps, flow_weight=0.0,
                       dist_reward_enabled=False, track_id=args.track,
                       action_set={6: "legacy6", 12: "compound12"}[n_actions],
                       goal_override=None if not args.goal_pos else
                       {"pos": target.tolist(), "radius": args.goal_radius})
    reset, act = make_policy("ppo", os.path.join(ROOT_DIR, args.model), None, "cpu")
    rows = []
    for p in paths:
        env.state_path = p
        obs, info = env.reset()
        reset(obs)
        c0, dmin = env.ram.get_coins(), np.inf
        for t in range(args.max_steps):
            obs, _, term, trunc, info = env.step(act(obs))
            dmin = min(dmin, float(np.linalg.norm(info["pos_3d"] - target)))
            if term or trunc:
                break
        end = "finish" if info.get("reached_goal") else "died" if info.get("died") else "horizon"
        rows.append({"state": os.path.basename(p), "end": end, "decisions": t + 1,
                     "coins_gained": int(env.ram.get_coins() - c0), "min_dist": round(dmin, 1)})
        print(json.dumps(rows[-1]), flush=True)
    env.close()
    summary = {"model": args.model, "states": args.states, "max_steps": args.max_steps,
               "finished": sum(r["end"] == "finish" for r in rows),
               "survived": sum(r["end"] != "died" for r in rows), "n": len(rows), "rows": rows}
    print(f"SUMMARY finished {summary['finished']}/{summary['n']}, not dead {summary['survived']}/{summary['n']}")
    if args.out:
        with open(os.path.join(ROOT_DIR, args.out), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
