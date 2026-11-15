"""Savestates of the main TTC policy at the hand-over point, for training the transfer specialist.

Rolls the main policy out from the normal ds8 start (greedy, random quantile levels, exactly as scripts/probe_ttc.py)
and saves a savestate the first time its route progress reaches --handoff-wp waypoints, plus one more every
--offsets decisions later while the main policy keeps playing, so the specialist sees the hand at different phases.

Usage:
    python scripts/make_ttc_handoff_states.py --model models/btr_ttc_s9/step_750000.pt --handoff-wp 340 \
        --episodes 10 --seed 1 --out data/curriculum/ttc_handoff
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
    ap.add_argument("--handoff-wp", type=int, default=340)
    ap.add_argument("--offsets", default="0,15,30", help="decisions after the first crossing at which more states are saved")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="per-decision probability (before the hand-over) of a random action of the main policy's set held 1-3 decisions; "
                         "the main policy alone reaches the hand almost always through the same states")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from scripts.train_btr import ref_env_kwargs
    from src.btr import BTRAgent, checkpoint_actions
    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path
    from src.route import Route

    import random
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    route = Route.load(os.path.join(ROOT_DIR, "config/routes/ttc_human_v1.json"))
    n_act = checkpoint_actions(os.path.join(ROOT_DIR, args.model))
    action_set = {10: "ref10", 13: "ref13"}[n_act]
    agent = BTRAgent(n_act, device="cpu")
    agent.load(os.path.join(ROOT_DIR, args.model))
    env = Mario64DSEnv(rom_path=rom_path(), state_path=savestate_path("ds8"), max_steps=args.max_steps,
                       track_id="ds8", **ref_env_kwargs("ds8", action_set=action_set))
    out = os.path.join(ROOT_DIR, args.out)
    os.makedirs(out, exist_ok=True)
    offsets = sorted(int(o) for o in args.offsets.split(","))
    index = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        dq = deque([np.zeros((84, 84), np.uint8)] * 3 + [obs[:, :, 0]], maxlen=4)
        cross, saved, best_wp, end = None, 0, 0, "horizon"
        rand_left, rand_a = 0, 0
        for t in range(args.max_steps):
            a = int(agent.act(np.stack(list(dq))[None], noisy=False, fixed_taus=False)[0])
            if cross is None and args.noise > 0:
                if rand_left == 0 and rng.random() < args.noise:
                    rand_a, rand_left = rng.randrange(min(n_act, 10)), rng.randint(1, 3)
                if rand_left > 0:
                    a, rand_left = rand_a, rand_left - 1
            obs, _, term, _, info = env.step(a)
            dq.append(obs[:, :, 0])
            best_wp = route.passed(0, float(info.get("route_progress") or 0.0))
            if cross is None and best_wp >= args.handoff_wp:
                cross = t
            if cross is not None and saved < len(offsets) and t - cross >= offsets[saved]:
                name = f"h{args.seed:02d}-{ep:02d}-{offsets[saved]:02d}.dst"
                env.emu.savestate.save_file(os.path.join(out, name))
                pos = [round(float(v), 1) for v in env.ram.get_pos_3d()]
                index.append({"file": name, "episode": ep, "decision": t + 1, "offset": offsets[saved], "wp": best_wp, "pos": pos})
                saved += 1
            if term:
                end = "star" if info.get("success") else "died"
                break
            if cross is not None and saved >= len(offsets):
                break
        print(ep, "crossed at", None if cross is None else cross + 1, "saved", saved, "best wp", best_wp, end, flush=True)
    env.close()
    with open(os.path.join(out, f"index_{args.seed:02d}.json"), "w", encoding="utf-8") as fh:
        json.dump({"model": args.model, "handoff_wp": args.handoff_wp, "states": index}, fh, indent=1)


if __name__ == "__main__":
    main()
