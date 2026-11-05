"""DAgger from one human run of ds3: a policy that finishes the slide from the real start.

Behaviour cloning of a single trajectory drifts off it (scripts/clone_demo.py: 0 of 20
perturbed runs reached the goal). DAgger fixes that by training on the states the learner
itself visits, labelled by an expert. The expert here is the human run: for a visited
position the label is the action taken at the nearest point of the run, searched in a window
that only moves forward along it. Per iteration: roll the learner out (the expert's action
with probability --beta, noise held for 1-3 decisions with probability --eps), label every
state, aggregate, retrain from the current weights, then evaluate greedily from the real
start and from perturbed starts (3-8 random actions).

Usage:
    python scripts/dagger_human.py --demo runs/human-demonstrations/<run>/completed/attempt-000001 \
        --out models/ds3_dagger.zip
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
import torch  # noqa: E402


def fresh_stack(obs):
    return deque([np.zeros((84, 84), np.uint8)] * 3 + [obs[:, :, 0]], maxlen=4)


class Expert:
    """Label = the human's action at the nearest point of the run (window moves forward only)."""

    def __init__(self, positions, actions):
        self.P, self.A = np.asarray(positions), np.asarray(actions)
        self.i = 0

    def reset(self):
        self.i = 0

    def label(self, pos):
        lo, hi = max(0, self.i - 3), min(len(self.A), self.i + 40)
        d = np.linalg.norm(self.P[lo:hi] - pos, axis=1)
        j = lo + int(np.argmin(d))
        self.i = max(self.i, j)
        return int(self.A[min(self.i, len(self.A) - 1)]), float(d.min())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--model", default="models/curriculum_flow25_r3_c12.zip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--result", default="results/ds3_dagger.json")
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=700)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--beta-decay", type=float, default=0.7)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--off-route", type=float, default=2500.0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval-seeds", type=int, default=6)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from src.demos import load_demo
    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path

    demo = load_demo(os.path.join(ROOT_DIR, args.demo))
    expert = Expert(demo["positions"], demo["actions"])
    env = Mario64DSEnv(rom_path=rom_path(), state_path=savestate_path("ds3"), max_steps=args.horizon + 600,
                       flow_weight=0.0, dist_reward_enabled=False, track_id="ds3", action_set="compound12",
                       black_screen_death=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(os.path.join(ROOT_DIR, args.model), device=device)
    pol = model.policy
    opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)

    def act(dq):
        return int(model.predict(np.stack(list(dq))[None], deterministic=True)[0][0])

    def run(beta, eps, seed, perturb=0, collect=False, horizon=None):
        r = np.random.default_rng(seed)
        obs, _ = env.reset()
        expert.reset()
        dq = fresh_stack(obs)
        X, Y, hold, noise = [], [], 0, 0
        info, t, dmin, off = {}, 0, 1e18, 0.0
        for t in range(horizon or args.horizon):
            pos = env.ram.get_pos_3d()
            lab, off = expert.label(pos)
            if collect:
                X.append(np.stack(list(dq)))
                Y.append(lab)
            if t < perturb:
                a = int(r.integers(12))
            elif hold > 0:
                a, hold = noise, hold - 1
            elif r.random() < eps:
                noise, hold = int(r.integers(12)), int(r.integers(0, 3))
                a = noise
            elif r.random() < beta:
                a = lab
            else:
                a = act(dq)
            obs, _, term, _, info = env.step(a)
            dq.append(obs[:, :, 0])
            dmin = min(dmin, float(info.get("dist_to_goal", dmin)))
            if term or off > args.off_route:
                break
        end = "finish" if info.get("reached_goal") else "died" if info.get("died") else \
            "off-route" if off > args.off_route else "horizon"
        return X, Y, {"end": end, "decisions": t + 1, "min_dist": round(dmin, 1)}

    # iteration 0: the human run itself
    obs, _ = env.reset()
    dq = fresh_stack(obs)
    X, Y = [], []
    for a in demo["actions"]:
        X.append(np.stack(list(dq)))
        Y.append(int(a))
        obs, _, term, _, _ = env.step(int(a))
        dq.append(obs[:, :, 0])
        if term:
            break
    history, beta, best = [], args.beta, -1
    for it in range(args.iterations + 1):
        if it > 0:
            ends = []
            for k in range(args.rollouts):
                x, y, res = run(beta, args.eps, 1000 * it + k, collect=True)
                X += x
                Y += y
                ends.append(res["end"])
            print(f"iter {it}: {args.rollouts} rollouts ({ends.count('finish')} finished), "
                  f"dataset {len(Y)}", flush=True)
            beta *= args.beta_decay
        Xa, Ya = np.stack(X), np.asarray(Y)
        for epoch in range(args.epochs):
            pol.set_training_mode(True)
            perm = rng.permutation(len(Ya))
            for i in range(0, len(Ya), args.batch):
                idx = perm[i:i + args.batch]
                obs_t = pol.obs_to_tensor(Xa[idx])[0]
                loss = -pol.get_distribution(obs_t).log_prob(torch.as_tensor(Ya[idx], device=device)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        pol.set_training_mode(False)
        _, _, start = run(0.0, 0.0, 0, horizon=1200)
        pert = [run(0.0, 0.0, 50 + s, perturb=3 + s % 6, horizon=1200)[2] for s in range(args.eval_seeds)]
        n_ok = sum(p["end"] == "finish" for p in pert)
        print(f"  eval: from start {start} | perturbed finish {n_ok}/{len(pert)} "
              f"({[p['end'][:4] for p in pert]})", flush=True)
        history.append({"iteration": it, "samples": len(Y), "from_start": start, "perturbed": pert,
                        "perturbed_finished": n_ok})
        score = (start["end"] == "finish") * 100 + n_ok
        if score > best:
            best = score
            model.save(os.path.join(ROOT_DIR, args.out))
        with open(os.path.join(ROOT_DIR, args.result), "w") as fh:
            json.dump({"demo": args.demo, "args": vars(args), "history": history}, fh, indent=1)
        if start["end"] == "finish" and n_ok == len(pert):
            break
    env.close()


if __name__ == "__main__":
    main()
