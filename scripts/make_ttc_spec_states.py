"""Savestates from the transfer specialist's own rollouts, for training it on the states where it goes wrong.

Plays the specialist from a hand-over savestate (stochastic: random quantile levels) and saves a savestate every --every
decisions while Mario is above height 5,800. Episodes that end in the star keep only their first --keep-star states
(the rest were handled well); episodes that die or run out of time keep all of them, so the start set concentrates on the
places where the specialist idles, falls or dies.

Usage:
    python scripts/make_ttc_spec_states.py --spec models/btr_ttc_spec3/final.pt --state scratch/handoff_one/c1.dst \
        --episodes 8 --seed 1 --out data/curriculum/ttc_spec_onpolicy
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
    ap.add_argument("--spec", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--start", type=int, default=40, help="first decision at which states are saved")
    ap.add_argument("--keep-star", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from scripts.eval_ttc_composite import fresh_stack, load_agent
    from scripts.train_btr import ref_env_kwargs
    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path

    torch.manual_seed(args.seed)
    spec, n_spec = load_agent(args.spec)
    env = Mario64DSEnv(rom_path=rom_path(), state_path=savestate_path("ds8"), max_steps=args.max_steps, track_id="ds8",
                       **ref_env_kwargs("ds8", action_set={10: "ref10", 13: "ref13"}[n_spec]))
    out = os.path.join(ROOT_DIR, args.out)
    os.makedirs(out, exist_ok=True)
    state = os.path.join(ROOT_DIR, args.state)
    index = []
    for ep in range(args.episodes):
        env.start_states, env.full_start_prob = [state], 0.0
        obs, _ = env.reset()
        dq = fresh_stack(obs)
        saved, end = [], "horizon"
        for t in range(args.max_steps):
            a = int(spec.act(np.stack(list(dq))[None], noisy=False, fixed_taus=False)[0])
            obs, _, term, _, info = env.step(a)
            dq.append(obs[:, :, 0])
            pos = env.ram.get_pos_3d()
            if t + 1 >= args.start and (t + 1 - args.start) % args.every == 0 and float(pos[1]) > 5800:
                name = f"dense-sp{args.seed:02d}-{ep:02d}-{t + 1:04d}.dst"
                env.emu.savestate.save_file(os.path.join(out, name))
                saved.append({"file": name, "decision": t + 1, "pos": [round(float(v), 1) for v in pos]})
            if term:
                end = "star" if info.get("success") else "died"
                break
        keep = saved[:args.keep_star] if end == "star" else saved
        for s in saved[len(keep):]:
            os.remove(os.path.join(out, s["file"]))
        index += [dict(s, episode=ep, end=end) for s in keep]
        print(f"episode {ep}: {end} after {t + 1} decisions, kept {len(keep)} of {len(saved)} states", flush=True)
    env.close()
    with open(os.path.join(out, f"index_{args.seed:02d}.json"), "w", encoding="utf-8") as fh:
        json.dump({"spec": args.spec, "states": index}, fh, indent=1)


if __name__ == "__main__":
    main()
