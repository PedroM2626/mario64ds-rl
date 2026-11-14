"""Evaluation only: force a jump at the end of the clock hand, then hand control back to the policy.

phase a: the deterministic ref13 policy plays ds8 from the full start; savestates are written at the decisions
         --from .. --to (every --every) around the end of the hand.
phase b: from each savestate, the first K decisions are forced to a fixed action (ref10 jump = 9 as control, or
         up+jump 10, up-left+jump 11, up-right+jump 12), then the deterministic policy plays on for --follow decisions.
         Per trial: best route arc length, whether Mario ends above height 5,800, and whether a star was reached.

Usage:
    python scripts/ttc_jump_override_test.py a --model models/btr_ttc_s9_ref13/step_750000_ref13.pt --out scratch/jt
    python scripts/ttc_jump_override_test.py b --model ... --out scratch/jt --part 0 --parts 4
"""
import argparse
import glob
import json
import os
import sys
from collections import deque

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import numpy as np  # noqa: E402


def make(model, state_path, max_steps):
    from scripts.train_btr import ref_env_kwargs
    from src.btr import BTRAgent
    from src.env import Mario64DSEnv
    from src.envs import rom_path
    agent = BTRAgent(13, device="cpu")
    agent.load(os.path.join(ROOT_DIR, model))
    env = Mario64DSEnv(rom_path=rom_path(), state_path=state_path, max_steps=max_steps, track_id="ds8",
                       **ref_env_kwargs("ds8", action_set="ref13"))
    return agent, env


def pol(agent, dq):
    return int(agent.act(np.stack(list(dq))[None], noisy=False, fixed_taus=True)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["a", "b"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--from", dest="lo", type=int, default=1640)
    ap.add_argument("--to", dest="hi", type=int, default=1700)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--follow", type=int, default=400)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    args = ap.parse_args()
    out = os.path.join(ROOT_DIR, args.out)
    os.makedirs(out, exist_ok=True)
    from src.envs import savestate_path
    if args.phase == "a":
        agent, env = make(args.model, savestate_path("ds8"), 5000)
        obs, _ = env.reset()
        dq = deque([np.zeros((84, 84), np.uint8)] * 3 + [obs[:, :, 0]], maxlen=4)
        for t in range(args.hi + 1):
            if t >= args.lo and (t - args.lo) % args.every == 0:
                env.emu.savestate.save_file(os.path.join(out, f"d{t:04d}.dst"))
            obs, _, term, _, info = env.step(pol(agent, dq))
            dq.append(obs[:, :, 0])
            if term:
                break
        print("phase a done", len(glob.glob(os.path.join(out, "d*.dst"))), "savestates")
        return
    files = sorted(glob.glob(os.path.join(out, "d*.dst")))[args.part::args.parts]
    rows = []
    agent, env = make(args.model, files[0], args.follow + 20)
    for f in files:
        env.state_path = f
        for forced in (9, 10, 11, 12, None):
            for K in ((0,) if forced is None else (2, 4, 8, 12)):
                obs, _ = env.reset()
                dq = deque([np.zeros((84, 84), np.uint8)] * 3 + [obs[:, :, 0]], maxlen=4)
                best, star, y = 0.0, False, 0.0
                for t in range(K + args.follow):
                    a = forced if t < K else pol(agent, dq)
                    obs, _, term, _, info = env.step(a)
                    dq.append(obs[:, :, 0])
                    best = max(best, float(info.get("route_progress") or 0.0))
                    y = float(env.ram.get_pos_3d()[1])
                    if term:
                        star = bool(info.get("success"))
                        break
                rows.append({"state": os.path.basename(f), "forced": forced, "K": K, "best_s": round(best, 1),
                             "y_end": round(y, 1), "star": star})
        json.dump(rows, open(os.path.join(out, f"result_{args.part}.json"), "w"))
    print("phase b done", len(rows), "trials")


if __name__ == "__main__":
    main()
