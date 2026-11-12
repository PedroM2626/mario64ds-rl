"""Savestates every --every decisions along a human run (the recorder keeps one per 30).

Replays the run's actions in the deterministic emulator, as scripts/prepare_human_route.py does, and
saves a savestate every --every decisions inside [--lo, --hi], with index.json (decision, position).
Used to give the PPO / QR-DQN curriculum many reset points where a policy leaves the course.

Usage:
    python scripts/dense_human_states.py runs/human-demonstrations/<run>/completed/attempt-000001 \
        --every 8 --lo 200 --hi 320 --out data/curriculum/ds3_human_dense
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("attempt")
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=10 ** 6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from src.demos import decision_inputs, load_demo
    from src.env import Mario64DSEnv
    from src.envs import rom_path
    demo = load_demo(os.path.join(ROOT_DIR, args.attempt))
    m = demo["manifest"]
    out = os.path.join(ROOT_DIR, args.out)
    os.makedirs(out, exist_ok=True)
    env = Mario64DSEnv(rom_path=rom_path(), state_path=m["source_state"], max_steps=len(demo["actions"]) + 1,
                       frameskip=m["frameskip"], flow_weight=0.0, action_set=m["action_set"],
                       track_id=m["track_id"], black_screen_death=False)
    env.reset()
    err = float(np.max(np.abs(env.ram.get_pos_3d() - demo["positions"][0])))
    index = []
    # the recorded per-frame inputs when the attempt has them (the TTC run), else the decision actions
    for t, a in enumerate(decision_inputs(demo)):
        if args.lo <= t <= args.hi and (t - args.lo) % args.every == 0:
            name = f"dense-{t:04d}.dst"
            env.emu.savestate.save_file(os.path.join(out, name))
            index.append({"decision": t, "file": name, "pos": [round(float(v), 1) for v in env.ram.get_pos_3d()]})
        env.step(a)
        if t + 1 < len(demo["positions"]):
            err = max(err, float(np.max(np.abs(env.ram.get_pos_3d() - demo["positions"][t + 1]))))
    env.close()
    if err > 1e-3:
        raise SystemExit(f"replay diverged from the recording (max position error {err:.3f}); no index written")
    with open(os.path.join(out, "index.json"), "w", encoding="utf-8") as fh:
        json.dump({"attempt": args.attempt, "states": index}, fh, indent=1)
    print(f"{len(index)} savestates -> {args.out}")


if __name__ == "__main__":
    main()
