"""Random-shooting search for the closest approach to the ds3 finish from the route's savestates.

explore_slide.py only keeps branches that survive, so it cannot discover a dive that passes
through the finish region and then falls. This one runs random rollouts (an action held
1..--hold-max decisions) from every route savestate and records the closest approach to the
finish (3D) and whether the finish region was entered, with no survival requirement.

Usage:
    python scripts/dive_search.py --states "data/curriculum/ds3_route/route-0[67]*.dst" --rollouts 64
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np  # noqa: E402

_ENV = None


def _env(path, action_set):
    global _ENV
    if _ENV is None:
        from src.env import Mario64DSEnv
        from src.envs import rom_path
        _ENV = Mario64DSEnv(rom_path=rom_path(), state_path=path, max_steps=100000, flow_weight=0.0,
                            dist_reward_enabled=False, track_id="ds3", action_set=action_set,
                            use_ram_death=False, black_screen_death=False)
    return _ENV


def rollout(job):
    path, seed, horizon, hold_max, action_set, prefix = job
    from src.ram_state import TRACK_GOALS, Sm64DsState
    g = TRACK_GOALS["ds3"]
    env = _env(path, action_set)
    env.state_path = path
    env.reset()
    rng = np.random.default_rng(seed)
    n = env.action_space.n
    acts, best, entered, d, best_at = [], 1e18, False, 0, 0
    hold_left, a = 0, 0
    for a0 in (prefix or []):
        env.step(int(a0))
        acts.append(int(a0))
        d += 1
        horizon += 1
    base = d
    while d < horizon:
        if hold_left == 0:
            a, hold_left = int(rng.integers(n)), int(rng.integers(1, hold_max + 1))
        env.step(a)
        acts.append(a)
        hold_left -= 1
        d += 1
        pos = env.ram.get_pos_3d()
        dist = float(np.linalg.norm(pos - g["target_pos"]))
        if dist < best:
            best, best_at = dist, d
        if Sm64DsState.in_finish_region(pos, g["target_pos"], g["finish_region"]):
            entered = True
            break
        if Sm64DsState.fell_out_of_course(pos, g["kill_y"]):
            break
    return {"state": os.path.basename(path), "seed": seed, "best": best, "best_at": best_at, "finish": entered, "actions": acts}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", required=True)
    ap.add_argument("--rollouts", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--hold-max", type=int, default=12)
    ap.add_argument("--action-set", default="compound12")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(ROOT_DIR, args.states)))
    jobs = [(p, args.seed * 100000 + i * 1000 + k, args.horizon, args.hold_max, args.action_set, None)
            for i, p in enumerate(paths) for k in range(args.rollouts)]
    print(f"{len(paths)} states x {args.rollouts} rollouts", flush=True)
    rows = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(rollout, jobs, chunksize=2):
            rows.append(r)
    per = {}
    for r in rows:
        per.setdefault(r["state"], []).append(r)
    for s, rs in sorted(per.items()):
        print(f"{s}: best {min(x['best'] for x in rs):8.1f}  finish {sum(x['finish'] for x in rs)}/{len(rs)}", flush=True)
    top = sorted(rows, key=lambda r: r["best"])[:5]
    print("overall best", [(t["state"], round(t["best"], 1), t["finish"]) for t in top])
    if args.out:
        json.dump({"args": vars(args), "top": top, "n": len(rows),
                   "finished": sum(r["finish"] for r in rows)}, open(os.path.join(ROOT_DIR, args.out), "w"))


if __name__ == "__main__":
    main()
