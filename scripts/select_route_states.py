"""Keep the savestates of a dense_human_states.py directory whose position projects onto the route.

An episode that starts from a savestate begins at its own place on the route (Mario64DSEnv.reset uses
Route.locate); a start that is off the corridor has no place and would pay the whole route behind it, so
those are left out. Writes copies to --out and prints (decision, route arc length) of the kept ones.

Usage:
    python scripts/select_route_states.py data/curriculum/ttc_end_dense --route config/routes/ttc_human_v1.json \
        --out data/curriculum/ttc_end_valid
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--route", default="config/routes/ttc_human_v1.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from src.route import Route
    route = Route.load(os.path.join(ROOT_DIR, args.route))
    src = os.path.join(ROOT_DIR, args.src)
    out = os.path.join(ROOT_DIR, args.out)
    os.makedirs(out, exist_ok=True)
    kept = []
    for st in json.load(open(os.path.join(src, "index.json"), encoding="utf-8"))["states"]:
        s = route.locate(st["pos"])
        print(st["decision"], st["pos"], "route_s", None if s is None else round(s, 1), "KEEP" if s is not None else "off corridor")
        if s is not None:
            shutil.copy(os.path.join(src, st["file"]), os.path.join(out, st["file"]))
            kept.append({"decision": st["decision"], "file": st["file"], "route_s": round(s, 1)})
    json.dump({"source": args.src, "states": kept}, open(os.path.join(out, "index.json"), "w", encoding="utf-8"), indent=1)
    print(f"{len(kept)} kept -> {args.out}")


if __name__ == "__main__":
    main()
