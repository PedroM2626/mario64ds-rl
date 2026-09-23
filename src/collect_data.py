"""Collect real emulator transitions into a world-model replay buffer.

This is the *data-collection* stage of the model-based pipeline: the emulator is
the only source of ground truth, and we deliberately keep its use small (tens of
thousands of steps) because the world model lets the *policy* train afterwards
without touching the (slow) emulator at all.

Behavior policies used to generate transitions:
* ``random``  — diverse exploration, including failures (teaches the terminal /
  death dynamics and the abyss penalty).
* ``ppo``     — replays an already-trained SB3 PPO agent (e.g. the benchmark
  ``curriculum_flow25_r3_best.zip``) so the buffer also contains *successful*
  full-descent trajectories. This is exactly the reference's principle of
  grounding the learned dynamics in genuine transitions.

Run as a module, e.g.:

    python -m src.collect_data --states ds1,ds2,ds3 --episodes-per-state 8 \
        --behavior random,ppo --ppo-model models/curriculum_flow25_r3_best.zip \
        --out data/wm_buffer.pkl
"""

from __future__ import annotations

import argparse
import os
from collections import deque

import numpy as np

from src.env import Mario64DSEnv
from src.wm_replay import EpisodeBuffer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stack_from_deque(dq: deque) -> np.ndarray:
    """(1,4,84,84) CHW float in [0,255] -> matches SB3 VecFrameStack order."""
    return np.stack(list(dq), axis=0)[np.newaxis, :]


def _rollout(env, policy, frameskip_actions) -> dict:
    obs, _ = env.reset()
    first = obs[:, :, 0]
    dq = deque([np.zeros_like(first)] * 3 + [first], maxlen=4)
    frames, actions, rewards, continues = [], [], [], []
    done = False
    steps = 0
    max_steps = env.max_steps
    while not done and steps < max_steps:
        action = frameskip_actions(steps, _stack_from_deque(dq))
        obs, reward, terminated, truncated, _ = env.step(int(action))
        frame = obs[:, :, 0]
        dq.append(frame)
        frames.append(frame)
        actions.append(int(action))
        rewards.append(float(reward))
        # continue = 1 unless a *terminal* death occurred at this step
        continues.append(0.0 if terminated else 1.0)
        done = bool(terminated or truncated)
        steps += 1
    return dict(frames=np.array(frames), actions=np.array(actions),
                rewards=np.array(rewards), continues=np.array(continues))


def collect(rom_path, states, episodes_per_state, behaviors, buffer: EpisodeBuffer,
            ppo_model=None, seed=0, max_steps=1350, flow_weight=2.5, step_penalty=0.02,
            frameskip=4, verbose=True, ppo_deterministic=False):
    rng = np.random.default_rng(seed)
    num_actions = 6
    ppo = None
    if "ppo" in behaviors:
        if not ppo_model:
            raise ValueError("--ppo-model required when using ppo behavior")
        from stable_baselines3 import PPO
        ppo = PPO.load(ppo_model)

    # DeSmuME cannot be (re)initialized more than once per process (access
    # violation), so we keep a SINGLE emulator instance alive and switch the
    # savestate file it loads on each reset() to move between tracks.
    existing = [s for s in states
                if os.path.exists(os.path.join(ROOT, "data",
                                               f"Super Mario 64 DS (USA) (Rev 1).{s}"))]
    if not existing:
        raise FileNotFoundError("no valid savestates found")
    env = Mario64DSEnv(rom_path, os.path.join(ROOT, "data",
                       f"Super Mario 64 DS (USA) (Rev 1).{existing[0]}"),
                       max_steps=max_steps, flow_weight=flow_weight,
                       step_penalty=step_penalty, frameskip=frameskip)
    try:
        for state in existing:
            env.state_path = os.path.join(ROOT, "data",
                                          f"Super Mario 64 DS (USA) (Rev 1).{state}")
            for beh in behaviors:
                for ep in range(episodes_per_state):
                    if beh == "random":
                        def policy_fn(_s, _o, _rng=rng):
                            return int(_rng.integers(num_actions))
                    elif beh == "ppo":
                        def policy_fn(_s, _o, _ppo=ppo, _det=ppo_deterministic):
                            a, _ = _ppo.predict(_o, deterministic=_det)
                            return int(a[0])
                    else:
                        raise ValueError(beh)
                    roll = _rollout(env, None, policy_fn)
                    completed = (len(roll["continues"]) > 0 and roll["continues"][-1] > 0.5)
                    buffer.add(roll["frames"], roll["actions"], roll["rewards"],
                               roll["continues"], source=beh, completed=completed)
                    if verbose:
                        print(f"[collect] {state}/{beh} ep{ep}: len={len(roll['frames'])} "
                              f"R={roll['rewards'].sum():.1f} "
                              f"{'COMPLETED' if completed else 'died'}", flush=True)
    finally:
        env.close()
    return buffer


def main():
    parser = argparse.ArgumentParser(description="Collect real transitions for the world model")
    parser.add_argument("--rom", default="data/Super Mario 64 DS (USA) (Rev 1).nds")
    parser.add_argument("--states", default="ds1,ds2,ds3")
    parser.add_argument("--episodes-per-state", type=int, default=8)
    parser.add_argument("--behavior", default="random,ppo",
                        help="comma list of {random,ppo}")
    parser.add_argument("--ppo-model", default="models/curriculum_flow25_r3_best.zip")
    parser.add_argument("--ppo-deterministic", action="store_true",
                        help="greedy PPO rollouts (reproduces the completing benchmark behavior)")
    parser.add_argument("--out", default="data/wm_buffer.pkl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1350)
    parser.add_argument("--flow-weight", type=float, default=2.5)
    args = parser.parse_args()

    rom_path = os.path.join(ROOT, args.rom)
    states = [s.strip() for s in args.states.split(",") if s.strip()]
    behaviors = [b.strip() for b in args.behavior.split(",") if b.strip()]
    ppo_path = os.path.join(ROOT, args.ppo_model) if args.ppo_model else None
    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)

    buffer = EpisodeBuffer()
    if os.path.exists(out_path):
        buffer.load(out_path)
        print(f"[collect] resumed buffer with {buffer.total_steps} steps / {len(buffer)} episodes")

    collect(rom_path, states, args.episodes_per_state, behaviors, buffer,
            ppo_model=ppo_path, seed=args.seed, max_steps=args.max_steps,
            flow_weight=args.flow_weight, ppo_deterministic=args.ppo_deterministic)
    buffer.save(out_path)
    print(f"[collect] saved {buffer.total_steps} steps / {len(buffer)} episodes -> {out_path}")


if __name__ == "__main__":
    main()
