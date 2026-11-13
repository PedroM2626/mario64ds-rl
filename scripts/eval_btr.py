"""Greedy evaluation of a BTR checkpoint (scripts/train_btr.py) on each slide's real start.

One episode per track with the training episode rules (no reward needed): reports the coins collected
(RAM counter), how it ended (finish region / died / horizon), the decisions used and, for ds3, the
closest distance to the finish. The agent acts greedily with the noisy layers off.

Usage:
    python scripts/eval_btr.py --model models/btr_ref10/step_1000000.pt --tracks ds1,ds2,ds3
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
    ap.add_argument("--tracks", default="ds1,ds2,ds3")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--start-random", type=int, default=0,
                    help="perturbed starts: each episode first plays a random number (0..N) of random actions, one "
                         "decision each; a perturbation that ends the episode is redrawn (up to 5 times)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action-set", default="ref10", choices=["ref10", "ref13", "legacy6", "compound12"])
    ap.add_argument("--algo", default="btr", choices=["btr", "ppo"], help="btr: scripts/train_btr.py checkpoint; ppo: scripts/train_ppo_ref.py")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
    if len(tracks) > 1:   # DeSmuME cannot host a second emulator in the same process: one process per track
        import subprocess
        import tempfile
        rows = []
        for t in tracks:
            # The child writes its rows to a file: the emulator's own text on stdout (written without a
            # newline) can end up in the same line as a result and used to hide one episode per slide.
            fd, tmp = tempfile.mkstemp(suffix=".json")
            os.close(fd)
            subprocess.run([sys.executable, os.path.abspath(__file__), "--model", args.model, "--tracks", t,
                            "--max-steps", str(args.max_steps), "--episodes", str(args.episodes),
                            "--device", args.device, "--start-random", str(args.start_random), "--action-set", args.action_set,
                            "--algo", args.algo, "--out", tmp],
                           capture_output=True, text=True, cwd=ROOT_DIR)
            with open(tmp, encoding="utf-8") as fh:
                child = json.load(fh)["rows"]
            os.remove(tmp)
            for r in child:
                rows.append(r)
                print(json.dumps(r), flush=True)
            if len(child) != args.episodes:
                print(f"WARNING: {t}: {len(child)} of {args.episodes} episodes", flush=True)
        if args.out:
            with open(os.path.join(ROOT_DIR, args.out), "w", encoding="utf-8") as fh:
                json.dump({"model": args.model, "start_random": args.start_random, "action_set": args.action_set,
                       "algo": args.algo, "rows": rows}, fh, indent=1)
        return

    from scripts.train_btr import ref_env_kwargs
    from src.btr import BTRAgent
    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path

    from src.env import ACTION_SETS
    n_actions = len(ACTION_SETS[args.action_set][0])
    if args.algo == "ppo":
        from stable_baselines3 import PPO
        ppo = PPO.load(os.path.join(ROOT_DIR, args.model), device="cpu")

        class _Agent:
            def act(self, stacks, noisy=False):
                return ppo.predict(stacks, deterministic=True)[0]
        agent = _Agent()
    else:
        agent = BTRAgent(n_actions, device=args.device)
        agent.load(os.path.join(ROOT_DIR, args.model))
    rows = []
    for track in tracks:
        env = Mario64DSEnv(rom_path=rom_path(), state_path=savestate_path(track), max_steps=args.max_steps,
                           track_id=track, **ref_env_kwargs(track, action_set=args.action_set))
        for ep in range(args.episodes):
            rng = np.random.RandomState(args.seed * 100003 + ep)
            k = 0
            for _try in range(5):
                obs, _ = env.reset()
                dq = deque([np.zeros((84, 84), np.uint8)] * 3 + [obs[:, :, 0]], maxlen=4)
                k, ended = (int(rng.randint(0, args.start_random + 1)) if args.start_random else 0), False
                for _ in range(k):
                    obs, _, term, trunc, _ = env.step(int(rng.randint(n_actions)))
                    dq.append(obs[:, :, 0])
                    if term or trunc:
                        ended = True
                        break
                if not ended:
                    break
            c0, dmin, info, end, t = env.ram.get_coins(), 1e18, {}, "horizon", 0
            for t in range(args.max_steps):
                a = int(agent.act(np.stack(list(dq))[None], noisy=False)[0])
                obs, _, term, trunc, info = env.step(a)
                dq.append(obs[:, :, 0])
                dmin = min(dmin, float(info.get("dist_to_goal", dmin)))
                if term:
                    end = "finish" if info.get("reached_goal") else "died"
                    break
            rows.append({"track": track, "episode": ep, "end": end, "decisions": t + 1,
                         "coins_gained": int(env.ram.get_coins() - c0), "min_dist": round(dmin, 1),
                         "random_actions": k})
            print(json.dumps(rows[-1]), flush=True)
        env.close()
    if args.out:
        with open(os.path.join(ROOT_DIR, args.out), "w", encoding="utf-8") as fh:
            json.dump({"model": args.model, "start_random": args.start_random, "action_set": args.action_set,
                       "algo": args.algo, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
