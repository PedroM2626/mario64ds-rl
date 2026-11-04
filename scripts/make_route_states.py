"""Savestates along a route found by scripts/explore_slide.py (no human needed).

Replays the first --upto actions of a search.json from the track's start in a
deterministic emulator and writes a savestate every --every decisions
(and the start itself), so that training can reset to many points of the route
(src/train_curriculum.py --start-states). Also writes index.json with each
savestate's decision, position and coin count, so milestones can be named
(for example "just before the red arrow" = decisions 300-346 of the ds3 route).

Usage:
    python scripts/make_route_states.py --search runs/explore/ds3-spot3b/search.json \
        --upto 720 --every 15 --out data/curriculum/ds3_route
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", required=True)
    ap.add_argument("--upto", type=int, required=True, help="Number of actions of the route to use")
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--action-set", default="compound12", choices=["legacy6", "compound12"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path
    spec = json.load(open(args.search, encoding="utf-8"))
    track, actions = spec["track"], spec["actions"][:args.upto]
    out = os.path.join(ROOT_DIR, args.out)
    os.makedirs(out, exist_ok=True)
    env = Mario64DSEnv(rom_path=rom_path(), state_path=savestate_path(track), max_steps=100000,
                       flow_weight=0.0, dist_reward_enabled=False, track_id=track,
                       action_set=args.action_set)
    env.reset()
    index = []

    def save(t):
        name = f"route-{t:04d}.dst"
        env.emu.savestate.save_file(os.path.join(out, name))
        index.append({"file": name, "decision": t, "coins": int(env.ram.get_coins()),
                      "pos": [round(float(v), 1) for v in env.ram.get_pos_3d()]})

    save(0)
    for t, a in enumerate(actions, 1):
        _, _, term, _, info = env.step(int(a))
        if term:
            raise SystemExit(f"the route dies at decision {t}: {info}")
        if t % args.every == 0:
            save(t)
    env.close()
    with open(os.path.join(out, "index.json"), "w", encoding="utf-8") as fh:
        json.dump({"search": args.search, "upto": args.upto, "every": args.every, "states": index}, fh, indent=1)
    print(f"{len(index)} savestates in {args.out}; last decision {index[-1]['decision']}, coins {index[-1]['coins']}")


if __name__ == "__main__":
    main()
