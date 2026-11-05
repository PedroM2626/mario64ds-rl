"""Turn a route found by scripts/explore_slide.py into a demonstration folder.

The folder has the layout src/demos.load_demo reads (the human recorder's):
demo.npz with the per-decision env action indices, and manifest.json with the
source savestate / action set / hashes. src/train_qrdqn.py --demo then replays it
under the training reward and puts its transitions into the replay buffer, exactly
as it does for a human recording. Every decision is `exact` (the actions are
indices of the env's own action set).

If a goal is given (--goal-pos/--goal-radius, as in train_curriculum), the route is
cut at the first decision that enters it, so that the replayed episode ends with the
goal bonus instead of being terminated "early".

Usage:
    python scripts/route_to_demo.py --search runs/explore/ds3-spot3b/search.json --upto 720 \
        --goal-pos 6432,-11263,-11499 --goal-radius 600 --out data/curriculum/ds3_route_demo
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np  # noqa: E402

from src.demos import sha256_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", required=True, help="search.json of an explore_slide run")
    ap.add_argument("--upto", type=int, required=True)
    ap.add_argument("--goal-pos", default=None)
    ap.add_argument("--goal-radius", type=float, default=600.0)
    ap.add_argument("--action-set", default="compound12", choices=["legacy6", "compound12"])
    ap.add_argument("--frameskip", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path
    spec = json.load(open(args.search, encoding="utf-8"))
    track, actions = spec["track"], [int(a) for a in spec["actions"][:args.upto]]
    start = savestate_path(track)
    goal = None if not args.goal_pos else {"pos": [float(v) for v in args.goal_pos.split(",")],
                                           "radius": args.goal_radius}
    env = Mario64DSEnv(rom_path=rom_path(), state_path=start, max_steps=len(actions) + 1,
                       frameskip=args.frameskip, flow_weight=0.0, dist_reward_enabled=False,
                       track_id=track, action_set=args.action_set, goal_override=goal)
    env.reset()
    positions, coins, used, reached = [env.ram.get_pos_3d().copy()], [env.ram.get_coins()], [], False
    for t, a in enumerate(actions):
        _, _, term, _, info = env.step(a)
        used.append(a)
        positions.append(info["pos_3d"].copy())
        coins.append(env.ram.get_coins())
        if term:
            if info.get("died"):
                raise SystemExit(f"the route dies at decision {t + 1}")
            reached = bool(info.get("reached_goal"))
            break
    env.close()

    out = os.path.join(ROOT_DIR, args.out, "attempt-000001")
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "demo.npz"), actions=np.asarray(used, np.int16),
                        exact=np.ones(len(used), bool), positions=np.stack(positions))
    manifest = {
        "recorder_version": 0, "state": "route_from_search",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "source_state": os.path.relpath(start, ROOT_DIR).replace(os.sep, "/"),
        "source_state_sha256": sha256_file(start), "track_id": track, "action_set": args.action_set,
        "frameskip": args.frameskip, "decisions": len(used), "reached_goal": reached,
        "coins": int(coins[-1]), "start_coins": int(coins[0]), "search": args.search.replace("\\", "/"),
        "goal": goal, "files_sha256": {"demo.npz": sha256_file(os.path.join(out, "demo.npz"))},
        "note": "Found by search (scripts/explore_slide.py + sweep_moves.py), not a human recording."}
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"{len(used)} decisions, goal reached={reached}, coins {coins[0]} -> {coins[-1]} -> {args.out}/attempt-000001")


if __name__ == "__main__":
    main()
