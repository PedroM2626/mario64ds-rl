"""Greedy refinement of dive_search: keep the best prefix, shoot again from its closest point.

Usage:
    python scripts/dive_refine.py --state data/curriculum/ds3_route/route-0285.dst --out results/dive_refine.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.dive_search import rollout  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", required=True)
    ap.add_argument("--rollouts", type=int, default=480)
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--hold-max", type=int, default=12)
    ap.add_argument("--stages", type=int, default=30)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--prefix-json", default=None, help="json list of actions to start from")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    path = os.path.join(ROOT_DIR, args.state)
    prefix = json.load(open(args.prefix_json)) if args.prefix_json else []
    best, stale, history = 1e18, 0, []
    with Pool(args.workers) as pool:
        for stage in range(args.stages):
            jobs = [(path, stage * 100000 + k, args.horizon, args.hold_max, "compound12", prefix)
                    for k in range(args.rollouts)]
            rows = pool.map(rollout, jobs, chunksize=2)
            top = min(rows, key=lambda r: r["best"])
            fin = [r for r in rows if r["finish"]]
            if fin:
                top = min(fin, key=lambda r: len(r["actions"]))
            gain = best - top["best"]
            print(f"stage {stage}: prefix {len(prefix)} -> best {top['best']:.1f} (gain {gain:.1f}), "
                  f"finish {len(fin)}/{len(rows)}", flush=True)
            history.append({"stage": stage, "best": top["best"], "finish": len(fin)})
            if top["best"] < best - 1.0:
                best, stale = top["best"], 0
                prefix = top["actions"][:top["best_at"]] if not top["finish"] else top["actions"]
                json.dump({"state": args.state, "actions": prefix, "best": best, "finish": bool(fin),
                           "history": history}, open(os.path.join(ROOT_DIR, args.out), "w"))
            else:
                stale += 1
            if fin or stale >= args.patience:
                break
    print("done", best, "finish" if fin else "")


if __name__ == "__main__":
    main()
