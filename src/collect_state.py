"""Collect exact-state transitions (RAM) for the state-based dynamics model.

The state-model pipeline (the reference smw-pinn's paradigm): learn
f(s, a) -> s' directly on Mario's physics struct read from RAM, then plan MPC
entirely in state space. This collector records, per episode:

    states  (T+1, 6)  raw fixed-point words at every step boundary
    actions (T,)
    rewards (T,)       env reward (dense flow + death penalty) for reference
    dones   (T,)       kill-plane hit or black-screen death

Behaviors mirror the pixel pipeline: `ppo` (deterministic expert descents),
`ppo_noise` (perturbed expert: racing-speed deaths + recoveries), `random`.

Usage:
    python -m src.collect_state --states ds1,ds2,ds3 \
        --episodes 3,6,4 --out data/state_ds.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np

from src.env import Mario64DSEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM_ACTIONS = 6


def rollout(env, policy, ram0):
    """One episode; returns (states (T+1,6), actions, rewards, falls, dones).

    ``falls`` marks kill-plane hits (a real in-game fall; slide courses RESPAWN
    you mid-track, so a fall is a heavily negative event but not terminal).
    ``dones`` marks true episode terminations (env black-screen death or the
    1350-step horizon).
    """
    obs, info = env.reset()
    s = info["ram"] if info.get("ram") is not None else ram0
    states = [np.asarray(s, dtype=np.float64)]
    actions, rewards, falls, dones = [], [], [], []
    done, steps = False, 0
    dq = [np.zeros((84, 84), np.uint8)] * 3 + [obs[:, :, 0]]
    while not done and steps < env.max_steps:
        stack = np.stack(dq, axis=0)[np.newaxis, :]
        action = policy(steps, stack)
        obs, r, term, trunc, info = env.step(int(action))
        dq.append(obs[:, :, 0]); dq.pop(0)
        s1 = info.get("ram")
        s1 = np.asarray(s1, dtype=np.float64) if s1 is not None else states[-1]
        states.append(s1)
        actions.append(int(action))
        rewards.append(float(r))
        from src.ram_state import Sm64DsState
        falls.append(bool(Sm64DsState.is_dead(s1)))
        dones.append(bool(term))
        done = bool(term or trunc)
        steps += 1
    return (np.array(states), np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32), np.array(falls),
            np.array(dones))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="ds1,ds2,ds3")
    ap.add_argument("--episodes", default="3,6,4",
                    help="episodes per behavior: ppo,ppo_noise,random")
    ap.add_argument("--noise-eps", type=float, default=0.15)
    ap.add_argument("--ppo-model", default="models/curriculum_flow25_r3_best.zip")
    ap.add_argument("--max-steps", type=int, default=1350)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/state_transitions.pkl")
    args = ap.parse_args()

    n_ppo, n_noise, n_rand = [int(x) for x in args.episodes.split(",")]
    rng = np.random.default_rng(args.seed)
    ppo = None
    if n_ppo or n_noise:
        from stable_baselines3 import PPO
        ppo = PPO.load(os.path.join(ROOT, args.ppo_model))

    rom = os.path.join(ROOT, "data/Super Mario 64 DS (USA) (Rev 1).nds")
    tracks = [s.strip() for s in args.states.split(",") if s.strip()]
    env = None
    data = {}
    try:
        for tr in tracks:
            ss = os.path.join(ROOT, f"data/Super Mario 64 DS (USA) (Rev 1).{tr}")
            if env is None:
                env = Mario64DSEnv(rom, ss, max_steps=args.max_steps)
            env.state_path = ss
            eps = []
            for beh, n in (("ppo", n_ppo), ("ppo_noise", n_noise), ("random", n_rand)):
                for ep in range(n):
                    if beh == "ppo":
                        def policy(_s, _o, _p=ppo):
                            a, _ = _p.predict(_o, deterministic=True)
                            return int(a[0])
                    elif beh == "ppo_noise":
                        def policy(_s, _o, _p=ppo, _r=rng, _e=args.noise_eps):
                            if _r.random() < _e:
                                return int(_r.integers(NUM_ACTIONS))
                            a, _ = _p.predict(_o, deterministic=False)
                            return int(a[0])
                    else:
                        def policy(_s, _o, _r=rng):
                            return int(_r.integers(NUM_ACTIONS))
                    ram0 = env.ram.poll()
                    ep_data = rollout(env, policy, ram0)
                    eps.append(dict(behavior=beh, **{
                        "states": ep_data[0], "actions": ep_data[1],
                        "rewards": ep_data[2], "falls": ep_data[3],
                        "dones": ep_data[4]}))
                    T = len(ep_data[1])
                    print(f"[collect-state] {tr}/{beh} ep{ep}: T={T} "
                          f"falls={int(ep_data[3].sum())} "
                          f"terminal={'DEAD' if ep_data[4][-1] else ('ok' if T >= env.max_steps else 'trunc')}",
                          flush=True)
            data[tr] = eps
    finally:
        if env:
            env.close()

    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    n = sum(len(v) for v in data.values())
    steps = sum(len(e["actions"]) for eps in data.values() for e in eps)
    print(f"[collect-state] saved {n} episodes / {steps} transitions -> {out_path}")


if __name__ == "__main__":
    main()
