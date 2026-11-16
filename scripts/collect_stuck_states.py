"""Savestates where the single network gets stuck at the end of the route (waypoint >= --min-wp without a star).

Plays the single network from the normal start (random quantile levels) and, when it has reached --min-wp waypoints and gained
no waypoint for --stall decisions, saves a savestate and the last three frames (the real frame stack), and goes on for --extra
more decisions to save a second one. Output: <out>/stuck-*.dst, stuck-*.npz (frames of the stack) and index_<seed>.json.

Usage (idle priority, see scripts/lowprio.py):
    python scripts/lowprio.py scripts/collect_stuck_states.py --student models/btr_ttc_merged3/final.pt --episodes 18 --seed 41 \
        --out data/distill_search/stuck
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
    ap.add_argument("--student", required=True)
    ap.add_argument("--episodes", type=int, default=18)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--min-wp", type=int, default=360)
    ap.add_argument("--stall", type=int, default=100)
    ap.add_argument("--extra", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=2700)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from scripts.eval_ttc_composite import load_agent
    from scripts.train_btr import ref_env_kwargs
    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path
    from src.route import Route

    torch.manual_seed(args.seed)
    route = Route.load(os.path.join(ROOT_DIR, "config/routes/ttc_human_v1.json"))
    student, n_act = load_agent(args.student)
    env = Mario64DSEnv(rom_path=rom_path(), state_path=savestate_path("ds8"), max_steps=args.max_steps, track_id="ds8",
                       **ref_env_kwargs("ds8", action_set="ref13"))
    out = os.path.join(ROOT_DIR, args.out)
    os.makedirs(out, exist_ok=True)
    index = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        dq = deque([np.zeros((84, 84), np.uint8)] * 3 + [obs[:, :, 0]], maxlen=4)
        best_wp, last_gain, stuck_at, saved, end = 0, 0, None, 0, "horizon"
        for t in range(args.max_steps):
            a = int(student.act(np.stack(list(dq))[None], noisy=False, fixed_taus=False)[0])
            obs, _, term, _, info = env.step(a)
            dq.append(obs[:, :, 0])
            wp = route.passed(0, float(info.get("route_progress") or 0.0))
            if wp > best_wp:
                best_wp, last_gain = wp, t
            if term:
                end = "star" if info.get("success") else "died"
                break
            if stuck_at is None and best_wp >= args.min_wp and t - last_gain >= args.stall:
                stuck_at = t
            if stuck_at is not None and t - stuck_at in (0, args.extra):
                name = f"stuck-{args.seed:02d}-{ep:02d}-{saved}"
                env.emu.savestate.save_file(os.path.join(out, name + ".dst"))
                np.savez(os.path.join(out, name + ".npz"), frames=np.stack(list(dq)))
                index.append({"file": name + ".dst", "episode": ep, "decision": t + 1, "best_wp": best_wp,
                              "pos": [round(float(v), 1) for v in env.ram.get_pos_3d()]})
                saved += 1
                if saved == 2:
                    end = "stuck"
                    break
        print(f"episode {ep}: {end} after {t + 1} decisions, best waypoint {best_wp}, saved {saved}", flush=True)
    env.close()
    with open(os.path.join(out, f"index_{args.seed:02d}.json"), "w", encoding="utf-8") as fh:
        json.dump({"student": args.student, "states": index}, fh, indent=1)


if __name__ == "__main__":
    main()
