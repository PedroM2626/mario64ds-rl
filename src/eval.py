"""Fair and reproducible evaluation across algorithms and savestates.

This script executes N evaluation episodes per savestate with deterministic or
stochastic policies, setting random seeds, and exporting per-episode metrics
to CSV files for controlled cross-algorithmic comparison (PPO vs Rainbow vs QR-DQN).

Examples:
    python -m src.eval --algo ppo --model models/ppo_mario64ds_continued_4_envs_best.zip --n-episodes 5 --deterministic
    python -m src.eval --algo rainbow --model models/rainbow_mario64ds_1h_best.pth --features impala --n-episodes 5
    python -m src.eval --algo qrdqn --model models/qrdqn_mario64ds.zip --n-episodes 5
"""

import argparse
import csv
import os

import numpy as np
import torch


def _savestates(base_dir, only=None):
    all_states = [
        os.path.join(base_dir, "data", "Super Mario 64 DS (USA) (Rev 1).ds1"),
        os.path.join(base_dir, "data", "Super Mario 64 DS (USA) (Rev 1).ds2"),
        os.path.join(base_dir, "data", "Super Mario 64 DS (USA) (Rev 1).ds3"),
    ]
    if only is not None:
        return [all_states[only]]
    return all_states


def _load_sb3_with_legacy_patch(Algo, model_path, device="auto"):
    """Loads an SB3 model with fallback redirection for legacy QR-DQN models.

    Earlier legacy checkpoints may have referenced ``src.impala_cnn.ImpalaCNN``
    as their ``features_extractor_class``. The current SB3-compatible class is
    ``src.sb3_impala.ImpalaFeaturesExtractor``. This patch dynamically redirects
    legacy pickle references if a type mismatch is detected.
    """
    try:
        return Algo.load(model_path, device=device)
    except TypeError as e:
        if "Box" not in str(e) and "%" not in str(e):
            raise
        import src.impala_cnn
        from src.sb3_impala import ImpalaFeaturesExtractor
        orig = src.impala_cnn.ImpalaCNN
        src.impala_cnn.ImpalaCNN = ImpalaFeaturesExtractor
        try:
            return Algo.load(model_path, device=device)
        finally:
            src.impala_cnn.ImpalaCNN = orig


def eval_sb3(algo, model_path, savestates, rom_path, n_episodes, deterministic, seed,
             max_steps=1350):
    # NOTE: Uses SubprocVecEnv (not DummyVecEnv) because DeSmuME C++ core causes
    # access violation exceptions if multiple emulators exist in the same Python process.
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
    from stable_baselines3.common.monitor import Monitor
    from src.env import Mario64DSEnv

    if algo == "ppo":
        from stable_baselines3 import PPO as Algo
    elif algo == "qrdqn":
        from sb3_contrib import QRDQN as Algo
    else:
        raise ValueError(algo)
    model = _load_sb3_with_legacy_patch(Algo, model_path, device="auto")

    rows = []
    for s_idx, sp in enumerate(savestates):
        def make_env():
            env = Mario64DSEnv(rom_path, sp)
            env = Monitor(env)
            return env

        vec = SubprocVecEnv([make_env])
        vec = VecFrameStack(vec, n_stack=4)
        vec = VecTransposeImage(vec)
        for ep in range(n_episodes):
            try:
                vec.seed(seed + s_idx * 100 + ep)
            except Exception:
                pass
            obs = vec.reset()
            done, ep_rew, steps = False, 0.0, 0
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, rews, dones, infos = vec.step(action)
                ep_rew += float(rews[0])
                steps += 1
                done = bool(dones[0])
            survived = steps >= max_steps
            rows.append({"savestate": os.path.basename(sp), "episode": ep,
                         "reward": ep_rew, "steps": steps, "survived": survived})
            print(f"[{algo} {os.path.basename(sp)} ep={ep}] reward={ep_rew:.2f} steps={steps}")
        vec.close()
    return rows


def eval_rainbow(model_path, features, savestates, rom_path, n_episodes, seed, device,
                 max_steps=1350):
    import gymnasium as gym
    from gymnasium.wrappers import FrameStackObservation
    from tianshou.data import Batch
    from tianshou.env import SubprocVectorEnv
    from src.env import Mario64DSEnv
    from src.impala_cnn import ImpalaCNN, TianshouNatureCNN, RainbowNet
    from tianshou.algorithm.modelfree.c51 import C51Policy

    feature_net = (ImpalaCNN(c=4, h=84, w=84, features_dim=256) if features == "impala"
                   else TianshouNatureCNN(c=4, h=84, w=84, features_dim=512))
    model = RainbowNet(feature_net, 6, 51, noisy_std=0.5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    policy = C51Policy(model=model, action_space=gym.spaces.Discrete(6),
                       num_atoms=51, v_min=-100.0, v_max=100.0).to(device)
    policy.eval()

    rows = []
    for s_idx, sp in enumerate(savestates):
        def make_env(s=sp):
            return FrameStackObservation(Mario64DSEnv(rom_path, s), stack_size=4)
        env = SubprocVectorEnv([make_env])
        for ep in range(n_episodes):
            np.random.seed(seed + s_idx * 100 + ep)
            torch.manual_seed(seed + s_idx * 100 + ep)
            obs, info = env.reset()
            done, ep_rew, steps = False, 0.0, 0
            while not done:
                with torch.no_grad():
                    act = policy(Batch(obs=obs, info=info)).act
                obs, rew, term, trunc, info = env.step(act)
                ep_rew += float(rew[0])
                steps += 1
                done = bool(term[0] or trunc[0])
            rows.append({"savestate": os.path.basename(sp), "episode": ep,
                         "reward": ep_rew, "steps": steps, "survived": steps >= max_steps})
            print(f"[rainbow-{features} {os.path.basename(sp)} ep={ep}] reward={ep_rew:.2f} steps={steps}")
        env.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Fair evaluation across algorithms and configurations")
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "qrdqn", "rainbow"])
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--features", type=str, default="impala", choices=["impala", "nature"])
    parser.add_argument("--n-episodes", type=int, default=5, help="Number of episodes per savestate")
    parser.add_argument("--deterministic", action="store_true", help="Execute greedy/deterministic actions")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--savestate-idx", type=int, default=None, choices=[0, 1, 2],
                        help="Evaluate specific savestate index only")
    parser.add_argument("--max-steps", type=int, default=1350,
                        help="Step limit per episode (survival condition: steps >= max-steps)")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (default: eval_<algo>.csv)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rom_path = os.path.join(base_dir, "data", "Super Mario 64 DS (USA) (Rev 1).nds")
    savestates = _savestates(base_dir, args.savestate_idx)

    if args.algo in ("ppo", "qrdqn"):
        rows = eval_sb3(args.algo, args.model, savestates, rom_path,
                        args.n_episodes, args.deterministic, args.seed,
                        max_steps=args.max_steps)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rows = eval_rainbow(args.model, args.features, savestates, rom_path,
                            args.n_episodes, args.seed, device,
                            max_steps=args.max_steps)

    out = args.out or f"eval_{args.algo}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["savestate", "episode", "reward", "steps", "survived"])
        w.writeheader()
        w.writerows(rows)

    rewards = [r["reward"] for r in rows]
    steps = [r["steps"] for r in rows]
    surv = sum(1 for r in rows if r["survived"])
    print(f"\n== {args.algo.upper()} ({len(rows)} episodes) ==")
    print(f"reward: mean={np.mean(rewards):.2f} ± {np.std(rewards):.2f} | "
          f"steps: mean={np.mean(steps):.1f} | survival: {surv}/{len(rows)}")
    print(f"CSV saved to: {out}")


if __name__ == "__main__":
    main()
