"""Systematic move sweep at a hard spot of a slide (no randomness).

Takes the action sequence of a scripts/explore_slide.py run, rebuilds exact
savestates at a few branch points shortly before where every rollout died, and
from each one tries every scripted opening

    single:   action a held for d decisions          (all compound12 actions)
    two-step: direction a1 for d1, then a2 for d2    (turns followed by jumps/runs)

followed by --tail greedy policy decisions. Each attempt reports the coins
gained (RAM counter), whether and when it died, and how close it came to the
finish; the best ones are listed. The emulator is deterministic, so a
working opening is a reproducible action sequence.

Usage:
    python scripts/sweep_moves.py --search runs/explore/ds3-20261104-084950/search.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np  # noqa: E402

from scripts import explore_slide as ex  # noqa: E402


def _attempt(job):
    state, history, opening, tail, point = job
    env, (reset, act) = ex._W["env"], ex._W["policy"]
    env.state_path = state
    obs, info = env.reset()
    reset(obs, history)
    c0 = env.ram.get_coins()
    goal = ex._W["goal"]
    plan = [a for a, d in opening for _ in range(d)]
    best_c, dmin, died, t = c0, np.inf, False, 0
    for t in range(len(plan) + tail):
        g = act(obs)
        obs, _, term, _, info = env.step(plan[t] if t < len(plan) else g)
        best_c = max(best_c, env.ram.get_coins())
        dmin = min(dmin, float(np.linalg.norm(info["pos_3d"] - goal)))
        if term:
            died = bool(info.get("died"))
            break
    return {"point": point, "opening": opening, "coins_gained": int(best_c - c0), "died": died,
            "decisions": t + 1, "min_dist": round(dmin, 1),
            "final_pos": [round(float(v), 1) for v in info["pos_3d"]]}


def _init(track, model, goal):
    ex._init(track, model, 6, "compound12")
    ex._W["goal"] = np.asarray(goal)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", required=True, help="search.json of an explore_slide run")
    ap.add_argument("--points", default="300,320,335,346",
                    help="Decisions of the searched run to branch from")
    ap.add_argument("--tail", type=int, default=150, help="Greedy decisions after each opening")
    ap.add_argument("--model", default="models/curriculum_flow25_r3_best.zip")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from scripts.eval_finish_line import make_policy
    from src.envs import savestate_path
    from src.ram_state import TRACK_GOALS
    spec = json.load(open(args.search, encoding="utf-8"))
    track, actions = spec["track"], spec["actions"]
    goal = TRACK_GOALS[track]["target_pos"]
    model = os.path.join(ROOT_DIR, args.model)
    run = os.path.join(os.path.dirname(os.path.abspath(args.search)), f"sweep-{time.strftime('%H%M%S')}")
    os.makedirs(run)
    ex._ACTION_SET[0] = "compound12"
    env = ex._env(track, savestate_path(track))
    policy = make_policy("ppo", model, None, "cpu")

    jobs_points = []
    for k in sorted(int(x) for x in args.points.split(",")):
        acts, pos, coins, died, _, hist = ex._play(env, policy, savestate_path(track), None, 0, 0.0, None,
                                                   forced=actions[:k])
        assert not died, f"replaying the searched run died before decision {k}"
        path = os.path.join(run, f"point-{k:04d}.dst")
        env.emu.savestate.save_file(path)
        jobs_points.append((k, path, hist, int(coins[-1])))
        print(f"branch point {k}: coins {int(coins[-1])}, pos {pos[-1].round(0)}", flush=True)
    env.close()

    n = 12  # compound12
    singles = [((a, d),) for a in range(n) for d in (2, 4, 8, 12, 16, 24)]
    turns = (1, 2, 3, 4, 6, 7)                       # left, right, up, down, up+left, up+right
    follows = (5, 8, 9, 10, 11, 1, 2)                # B, up+B, up+left+B, up+right+B, up+Y, left, right
    doubles = [((a1, d1), (a2, d2)) for a1, d1, a2, d2 in
               itertools.product(turns, (4, 8, 16), follows, (2, 4, 8)) if a1 != a2]
    openings = singles + doubles
    jobs = [(path, hist, op, args.tail, k) for k, path, hist, _ in jobs_points for op in openings]
    print(f"{len(openings)} openings x {len(jobs_points)} points = {len(jobs)} attempts", flush=True)
    results, t0 = [], time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(track, model, goal.tolist())) as pool:
        for i, r in enumerate(pool.imap_unordered(_attempt, jobs, chunksize=4), 1):
            results.append(r)
            if r["coins_gained"] >= 3 or (not r["died"]):
                print("CANDIDATE " + json.dumps(r), flush=True)
            if i % 200 == 0:
                print(f"{i}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)
    results.sort(key=lambda r: (-r["coins_gained"], r["died"], -r["decisions"]))
    out = args.out or os.path.join(run, "sweep.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"search": args.search, "points": [(k, c) for k, _, _, c in jobs_points],
                   "openings": len(openings), "tail": args.tail, "results": results}, fh, indent=1)
    print("TOP")
    for r in results[:15]:
        print(json.dumps(r))
    print(f"alive after tail: {sum(not r['died'] for r in results)} / {len(results)}; "
          f"max coins gained: {results[0]['coins_gained']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
