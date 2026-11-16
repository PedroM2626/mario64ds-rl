"""Search for the last jump of Tick Tock Clock from the states where the single network gets stuck (policy-guided rollouts).

From each savestate of --states (scripts/collect_stuck_states.py) the transfer specialist tries --attempts times; the attempts
differ in the exploration added on top of its own (random quantile level) choices: with probability eps per decision a random
action of the 13 is held for 1-4 decisions. An attempt counts only when the episode ends in the star (the course unloads with
Mario near the star). Every successful attempt is saved as an expert demonstration for the states it passed through: the
frames (with the three frames of the real stack before it) and the executed actions, used as extra labels for distillation.

Usage (idle priority): python scripts/lowprio.py scripts/search_final_jump.py --spec models/btr_ttc_spec5/final.pt \
    --states data/distill_search/stuck --out data/distill_search/found --shard 0 --shards 2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import deque

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np  # noqa: E402

EPS = [0.0, 0.0, 0.03, 0.03, 0.06, 0.06, 0.1, 0.1, 0.15, 0.15, 0.2, 0.2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--attempts", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=450)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from scripts.eval_ttc_composite import load_agent
    from scripts.train_btr import ref_env_kwargs
    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    spec, n_act = load_agent(args.spec)
    env = Mario64DSEnv(rom_path=rom_path(), state_path=savestate_path("ds8"), max_steps=args.max_steps, track_id="ds8",
                       **ref_env_kwargs("ds8", action_set="ref13"))
    out = os.path.join(ROOT_DIR, args.out)
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ROOT_DIR, args.states, "*.dst")))[args.shard::args.shards]
    rows = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        stack0 = np.load(os.path.join(os.path.dirname(f), name + ".npz"))["frames"]          # (4, 84, 84): the real stack when it was saved
        n_ok = 0
        for k in range(args.attempts):
            eps = EPS[k % len(EPS)]
            env.start_states, env.full_start_prob = [f], 0.0
            env.reset()
            dq = deque([x for x in stack0], maxlen=4)
            frames, actions = [], []
            held, held_a, end = 0, 0, "horizon"
            for t in range(args.max_steps):
                stack = np.stack(list(dq))[None]
                if held == 0 and rng.random() < eps:
                    held, held_a = rng.randint(1, 4), rng.randrange(n_act)
                if held > 0:
                    a, held = held_a, held - 1
                else:
                    a = int(spec.act(stack, noisy=False, fixed_taus=False)[0])
                frames.append(dq[-1].copy())
                actions.append(a)
                obs, _, term, _, info = env.step(a)
                dq.append(obs[:, :, 0])
                if term:
                    end = "star" if info.get("success") else "died"
                    break
            ok = end == "star"
            n_ok += ok
            rows.append({"state": name, "attempt": k, "eps": eps, "end": end, "decisions": t + 1})
            if ok:
                np.savez(os.path.join(out, f"found-{name}-{k:02d}.npz"), ctx=stack0[:3], frames=np.stack(frames),
                         actions=np.array(actions, np.int8))
        print(f"{name}: {n_ok}/{args.attempts} attempts reached the star", flush=True)
    env.close()
    with open(os.path.join(out, f"search_{args.shard:02d}.json"), "w", encoding="utf-8") as fh:
        json.dump({"spec": args.spec, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
