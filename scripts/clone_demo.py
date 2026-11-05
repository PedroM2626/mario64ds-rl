"""Behaviour-clone a demonstration into a PPO policy and test how well it imitates it.

Question this answers: the found ds3 route contains the manoeuvre past the red
arrow, yet neither PPO nor QR-DQN reproduced it. Can the policy network imitate
it when taught directly (cross-entropy on the demonstrated action)? If yes, RL can
refine from there; if not, the problem is not exploration or credit assignment.

Dataset (frame stacks built exactly as the trainers and make_policy do, [0,0,0,frame]
at a reset):
  * the continuous replay of the demo from the track start, and
  * for every route savestate, the next --window demo decisions from a fresh stack
    (training resets to those savestates have no frame history).
Evaluation (greedy, CPU-equivalent policy, --horizon decisions, goal as in training):
  * the four arrow savestates (decisions 300-345), as in the PPO/QR-DQN milestones;
  * every route savestate (does the clone reach the goal from there?);
  * perturbed starts: k random actions (k in 3..8) from each arrow savestate, several
    seeds -- imitation that survives a small deviation from the demonstrated states.
The arrow savestates are inside the training data, so (1) measures imitation, (3)
measures robustness; neither is held-out generalisation to new courses.

Usage:
    python scripts/clone_demo.py --demo data/curriculum/ds3_route_demo/attempt-000001 \
        --states-dir data/curriculum/ds3_route --out models/ds3_bc_c12.zip
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

GOAL = {"pos": [6432.0, -11263.0, -11499.0], "radius": 600.0}


def fresh_stack(obs):
    return deque([np.zeros((84, 84), np.uint8)] * 3 + [obs[:, :, 0]], maxlen=4)


def collect(env, actions, state):
    env.state_path = state
    obs, _ = env.reset()
    dq = fresh_stack(obs)
    X, Y = [], []
    for a in actions:
        X.append(np.stack(list(dq)))
        Y.append(int(a))
        obs, _, term, _, _ = env.step(int(a))
        dq.append(obs[:, :, 0])
        if term:
            break
    return X, Y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--states-dir", required=True)
    ap.add_argument("--model", default="models/curriculum_flow25_r3_c12.zip", help="compound12 PPO to start from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--result", default="results/ds3_bc_demo.json")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from src.demos import load_demo
    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path

    demo = load_demo(os.path.join(ROOT_DIR, args.demo))
    actions = [int(a) for a in demo["actions"]]
    index = json.load(open(os.path.join(ROOT_DIR, args.states_dir, "index.json"), encoding="utf-8"))["states"]
    env = Mario64DSEnv(rom_path=rom_path(), state_path=savestate_path("ds3"), max_steps=2000, flow_weight=0.0,
                       dist_reward_enabled=False, track_id="ds3", action_set="compound12", goal_override=GOAL)

    X, Y = collect(env, actions, savestate_path("ds3"))
    n_cont = len(Y)
    for s in index:
        t = s["decision"]
        if t >= len(actions):
            continue
        x, y = collect(env, actions[t:t + args.window], os.path.join(ROOT_DIR, args.states_dir, s["file"]))
        X += x
        Y += y
    X, Y = np.stack(X), np.asarray(Y)
    print(f"dataset: {len(Y)} samples ({n_cont} continuous + {len(Y) - n_cont} from {len(index)} savestates); "
          f"actions used: {np.bincount(Y, minlength=12).tolist()}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(os.path.join(ROOT_DIR, args.model), device=device)
    pol = model.policy
    opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)
    for epoch in range(1, args.epochs + 1):
        pol.set_training_mode(True)
        perm = rng.permutation(len(Y))
        tot, correct = 0.0, 0
        for i in range(0, len(Y), args.batch):
            idx = perm[i:i + args.batch]
            obs_t = pol.obs_to_tensor(X[idx])[0]
            a_t = torch.as_tensor(Y[idx], device=device)
            dist = pol.get_distribution(obs_t)
            loss = -dist.log_prob(a_t).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        pol.set_training_mode(False)
        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                pred = np.concatenate([model.predict(X[i:i + 256], deterministic=True)[0]
                                       for i in range(0, len(Y), 256)])
            print(f"epoch {epoch:3d} loss {tot / len(Y):.4f} train accuracy {np.mean(pred == Y):.3f}", flush=True)
    acc = float(np.mean(pred == Y))
    model.save(os.path.join(ROOT_DIR, args.out))

    def act(dq):
        return int(model.predict(np.stack(list(dq))[None], deterministic=True)[0][0])

    def run(path, perturb=0, seed=0):
        env.state_path = path
        obs, _ = env.reset()
        dq = fresh_stack(obs)
        r = np.random.default_rng(seed)
        c0, dmin, info = env.ram.get_coins(), np.inf, {}
        goal = np.asarray(GOAL["pos"])
        for t in range(args.horizon):
            a = int(r.integers(12)) if t < perturb else act(dq)
            obs, _, term, _, info = env.step(a)
            dq.append(obs[:, :, 0])
            dmin = min(dmin, float(np.linalg.norm(info["pos_3d"] - goal)))
            if term:
                break
        end = "finish" if info.get("reached_goal") else "died" if info.get("died") else "horizon"
        return {"end": end, "decisions": t + 1, "coins_gained": int(env.ram.get_coins() - c0),
                "min_dist": round(dmin, 1)}

    sd = os.path.join(ROOT_DIR, args.states_dir)
    arrow = {s["decision"]: run(os.path.join(sd, s["file"])) for s in index if s["decision"] in (300, 315, 330, 345)}
    allr = {s["decision"]: run(os.path.join(sd, s["file"])) for s in index}
    pert = []
    for d in (300, 315, 330, 345):
        f = os.path.join(sd, next(s["file"] for s in index if s["decision"] == d))
        for seed in range(args.seeds):
            k = int(np.random.default_rng(100 + seed).integers(3, 9))
            pert.append({"decision": d, "seed": seed, "perturb": k, **run(f, perturb=k, seed=seed)})
    env.close()
    reached = lambda rs: sum(r["end"] == "finish" for r in rs)
    summary = {"demo": args.demo, "samples": int(len(Y)), "epochs": args.epochs, "train_accuracy": round(acc, 3),
               "arrow_states": {str(k): v for k, v in arrow.items()},
               "arrow_reached_goal": reached(arrow.values()),
               "route_states_reached_goal": reached(allr.values()), "route_states": len(allr),
               "perturbed_reached_goal": reached(pert), "perturbed_runs": len(pert),
               "perturbed": pert}
    with open(os.path.join(ROOT_DIR, args.result), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"arrow states reaching the goal: {summary['arrow_reached_goal']}/4 | route states: "
          f"{summary['route_states_reached_goal']}/{summary['route_states']} | perturbed: "
          f"{summary['perturbed_reached_goal']}/{summary['perturbed_runs']} | train accuracy {acc:.3f}")
    for k, v in arrow.items():
        print(f"  arrow {k}: {v}")


if __name__ == "__main__":
    main()
