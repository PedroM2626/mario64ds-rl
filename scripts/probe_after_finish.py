"""What happens after the finish-region test fires on ds3?

Runs a BTR checkpoint greedily until an episode ends in the finish region (retrying: the game is not
deterministic), then keeps stepping the emulator for --extra more decisions with the same policy and
logs position, coin counter and distance to the target every few decisions. Used to check whether the
region ends the episode before the real end of the slide.

Usage:
    python scripts/probe_after_finish.py --model models/btr_ref10/step_2500005.pt --attempts 10
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
    ap.add_argument("--track", default="ds3")
    ap.add_argument("--attempts", type=int, default=10)
    ap.add_argument("--extra", type=int, default=150)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from scripts.train_btr import ref_env_kwargs
    from src.btr import BTRAgent
    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path

    agent = BTRAgent(10, device="cpu")
    agent.load(os.path.join(ROOT_DIR, args.model))
    env = Mario64DSEnv(rom_path=rom_path(), state_path=savestate_path(args.track), max_steps=5000,
                       track_id=args.track, **ref_env_kwargs(args.track))
    result = None
    for attempt in range(args.attempts):
        obs, _ = env.reset()
        dq = deque([np.zeros((84, 84), np.uint8)] * 3 + [obs[:, :, 0]], maxlen=4)
        c0, hit, rows = env.ram.get_coins(), None, []
        for t in range(1500 + args.extra):
            a = int(agent.act(np.stack(list(dq))[None], noisy=False)[0])
            obs, _, term, _, info = env.step(a)
            dq.append(obs[:, :, 0])
            if term and hit is None:
                if not info.get("reached_goal"):
                    break                      # died: next attempt
                hit = t + 1
            if hit is not None:
                pos = env.ram.get_pos_3d()
                rows.append({"decision": t + 1, "after_finish": t + 1 - hit, "pos": [round(float(v), 1) for v in pos],
                             "coins": int(env.ram.get_coins() - c0),
                             "dist": round(float(info.get("dist_to_goal", float("nan"))), 1)})
                if t + 1 - hit >= args.extra:
                    break
        if hit is not None:
            result = {"attempt": attempt + 1, "finish_decision": hit, "trace": rows}
            break
    env.close()
    if result is None:
        print("no finish in", args.attempts, "attempts")
        return
    print("finish at decision", result["finish_decision"], "on attempt", result["attempt"])
    for r in result["trace"][::10]:
        print(r)
    if args.out:
        with open(os.path.join(ROOT_DIR, args.out), "w", encoding="utf-8") as fh:
            json.dump({"model": args.model, **result}, fh, indent=1)


if __name__ == "__main__":
    main()
