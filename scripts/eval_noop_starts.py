"""Robustness of a (deterministic) policy on ds8: start the episode after k no-op decisions, k given on the command line.

With fixed quantile levels (--deterministic) the policy is a function of the observation and the emulator is deterministic, so
the unperturbed episode is a single trajectory. Starting after k decisions of the no-op action (action 0) puts Mario in
slightly different states and shows whether the policy still gets to the star.

Usage:
    python scripts/lowprio.py scripts/eval_noop_starts.py --model models/btr_ttc_merged/final.pt --deterministic \
        --ks 0 1 2 3 4 5 --max-steps 3000 --out results/ttc_merged_noop_a.json
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
    ap.add_argument("--ks", type=int, nargs="+", required=True)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from scripts.train_btr import ref_env_kwargs
    from src.btr import BTRAgent, checkpoint_actions
    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path
    from src.route import Route

    route = Route.load(os.path.join(ROOT_DIR, "config/routes/ttc_human_v1.json"))
    n_act = checkpoint_actions(os.path.join(ROOT_DIR, args.model))
    agent = BTRAgent(n_act, device="cpu")
    agent.load(os.path.join(ROOT_DIR, args.model))
    env = Mario64DSEnv(rom_path=rom_path(), state_path=savestate_path("ds8"), max_steps=args.max_steps + max(args.ks) + 5,
                       track_id="ds8", **ref_env_kwargs("ds8", action_set={10: "ref10", 13: "ref13"}[n_act]))
    rows = []
    for k in args.ks:
        obs, _ = env.reset()
        dq = deque([np.zeros((84, 84), np.uint8)] * 3 + [obs[:, :, 0]], maxlen=4)
        for _ in range(k):
            obs, *_ = env.step(0)
            dq.append(obs[:, :, 0])
        end, info, t = "horizon", {}, 0
        for t in range(args.max_steps):
            a = int(agent.act(np.stack(list(dq))[None], noisy=False, fixed_taus=args.deterministic)[0])
            obs, _, term, _, info = env.step(a)
            dq.append(obs[:, :, 0])
            if term:
                end = "star" if info.get("success") else "died"
                break
        rows.append({"k": k, "end": end, "decisions": t + 1, "best_wp": route.passed(0, float(info.get("route_progress") or 0.0))})
        print(rows[-1], flush=True)
    env.close()
    with open(os.path.join(ROOT_DIR, args.out), "w", encoding="utf-8") as fh:
        json.dump({"model": args.model, "deterministic": args.deterministic, "rows": rows}, fh)


if __name__ == "__main__":
    main()
