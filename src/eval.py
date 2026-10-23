"""Fair and reproducible evaluation across algorithms and savestates.

The evaluator is deliberately explicit about the two things that used to make
the published numbers ambiguous:

* **device** -- the same checkpoint can complete ds3 on CPU and die on CUDA
  (see results/device_divergence.json). The default here is cpu, the instrument
  used by the published benchmark; pass --device auto/cuda to study the other
  arm. The resolved device and the checkpoint SHA256 are written to the JSON.
* **independent samples** -- three deterministic episodes from one savestate are
  three copies of one trajectory. The summary reports n_unique next to n and a
  Wilson interval for survival, instead of presenting the copies as evidence.

Examples:
    python -m src.eval --algo ppo --model models/curriculum_flow25_r3_best.zip \
        --n-episodes 5 --deterministic --seed 0 --device cpu
    python -m src.eval --algo qrdqn --model models/qrdqn_mario64ds.zip \
        --n-episodes 3 --deterministic --flow-weight 1.0
"""

import argparse
import csv
import hashlib
import json
import math
import os

import numpy as np
import torch

from src.envs import savestate_path


def _savestates(base_dir, only=None):
    all_states = [savestate_path(t, base_dir) for t in ("ds1", "ds2", "ds3")]
    if only is not None:
        return [all_states[only]]
    return all_states


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (k successes of n)."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize_rows(rows):
    """Per-track and overall statistics, including the effective unique count."""
    by_track = {}
    for row in rows:
        by_track.setdefault(row["savestate"], []).append(row)

    def summarize(group):
        rewards = [float(r["reward"]) for r in group]
        steps = [int(r["steps"]) for r in group]
        survived = sum(1 for r in group if bool(r["survived"]))
        unique = len({(round(float(r["reward"]), 6), int(r["steps"])) for r in group})
        n = len(group)
        lo, hi = wilson_ci(survived, n)
        return {
            "n": n,
            "n_unique": unique,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "mean_steps": float(np.mean(steps)) if steps else 0.0,
            "survived": survived,
            "survival_rate": (survived / n) if n else 0.0,
            "survival_ci95": [round(lo, 6), round(hi, 6)],
        }

    tracks = {}
    for name, group in sorted(by_track.items()):
        tracks[name] = summarize(group)
    all_rows = list(rows)
    overall = summarize(all_rows) if all_rows else summarize([])
    overall["n_unique"] = sum(t["n_unique"] for t in tracks.values())
    return {"tracks": tracks, "overall": overall}


def _seed_everything(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_sb3_with_legacy_patch(Algo, model_path, device="auto"):
    """Load an SB3 model with fallback redirection for legacy QR-DQN models.

    Earlier legacy checkpoints may have referenced src.impala_cnn.ImpalaCNN as
    their features_extractor_class. The current SB3-compatible class is
    src.sb3_impala.ImpalaFeaturesExtractor. This patch dynamically redirects
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
             max_steps=1350, device="cpu", flow_weight=1.0, step_penalty=0.02):
    if algo == "ppo":
        from stable_baselines3 import PPO as Algo
    elif algo == "qrdqn":
        from sb3_contrib import QRDQN as Algo
    else:
        raise ValueError(algo)
    model = _load_sb3_with_legacy_patch(Algo, model_path, device=device)
    resolved = str(getattr(model, "device", device))

    from src.envs import build_vec

    rows = []
    for s_idx, sp in enumerate(savestates):
        vec = build_vec(rom_path, [sp], rank0=0, seed=seed, max_steps=max_steps,
                        frameskip=4, step_penalty=step_penalty,
                        flow_weight=flow_weight, monitor=True)
        for ep in range(n_episodes):
            episode_seed = seed + s_idx * 100 + ep
            _seed_everything(episode_seed)
            try:
                vec.seed(episode_seed)
            except Exception:
                pass
            try:
                model.set_random_seed(episode_seed)
            except Exception:
                pass
            obs = vec.reset()
            done, ep_rew, steps = False, 0.0, 0
            while not done and steps < max_steps:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, rews, dones, _infos = vec.step(action)
                ep_rew += float(rews[0])
                steps += 1
                done = bool(dones[0])
            survived = steps >= max_steps
            rows.append({
                "savestate": os.path.basename(sp), "episode": ep,
                "reward": ep_rew, "steps": steps, "survived": survived,
                "device": resolved, "flow_weight": flow_weight,
                "deterministic": bool(deterministic), "seed": episode_seed,
            })
            print(f"[{algo} {os.path.basename(sp)} ep={ep} seed={episode_seed}] "
                  f"reward={ep_rew:.2f} steps={steps} device={resolved}")
        vec.close()
    return rows


def eval_rainbow(model_path, features, savestates, rom_path, n_episodes, seed, device,
                 max_steps=1350, flow_weight=1.0, step_penalty=0.02):
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
            return FrameStackObservation(
                Mario64DSEnv(rom_path, s, max_steps=max_steps,
                             step_penalty=step_penalty, flow_weight=flow_weight),
                stack_size=4,
            )
        env = SubprocVectorEnv([make_env])
        for ep in range(n_episodes):
            episode_seed = seed + s_idx * 100 + ep
            _seed_everything(episode_seed)
            try:
                env.seed([episode_seed])
            except Exception:
                pass
            obs, info = env.reset()
            done, ep_rew, steps = False, 0.0, 0
            while not done and steps < max_steps:
                with torch.no_grad():
                    act = policy(Batch(obs=obs, info=info)).act
                obs, rew, term, trunc, info = env.step(act)
                ep_rew += float(rew[0])
                steps += 1
                done = bool(term[0] or trunc[0])
            rows.append({
                "savestate": os.path.basename(sp), "episode": ep,
                "reward": ep_rew, "steps": steps,
                "survived": steps >= max_steps, "device": device,
                "flow_weight": flow_weight, "deterministic": True,
                "seed": episode_seed,
            })
            print(f"[rainbow-{features} {os.path.basename(sp)} ep={ep} "
                  f"seed={episode_seed}] reward={ep_rew:.2f} steps={steps} device={device}")
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
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device: cpu (default; reproduces the published ds3 result), "
                             "auto or cuda. An explicit cuda on a CPU-only host fails early.")
    parser.add_argument("--flow-weight", type=float, default=1.0,
                        help="Optical-flow weight in the evaluated env (published independent "
                             "eval uses 1.0; the training weight is 2.5)")
    parser.add_argument("--step-penalty", type=float, default=0.02,
                        help="Per-step cost in the evaluated env")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (default: eval_<algo>.csv)")
    parser.add_argument("--json", type=str, default=None,
                        help="Output JSON path (default: the CSV path with .json)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rom_path = os.path.join(base_dir, "data", "Super Mario 64 DS (USA) (Rev 1).nds")
    savestates = _savestates(base_dir, args.savestate_idx)

    if args.algo in ("ppo", "qrdqn"):
        from src.device import resolve_device
        device_arg = args.device if args.device in ("auto", "cpu")             else resolve_device(args.device)
        rows = eval_sb3(args.algo, args.model, savestates, rom_path,
                        args.n_episodes, args.deterministic, args.seed,
                        max_steps=args.max_steps, device=device_arg,
                        flow_weight=args.flow_weight, step_penalty=args.step_penalty)
    else:
        from src.device import resolve_device
        device = resolve_device(args.device)
        rows = eval_rainbow(args.model, args.features, savestates, rom_path,
                            args.n_episodes, args.seed, device,
                            max_steps=args.max_steps, flow_weight=args.flow_weight,
                            step_penalty=args.step_penalty)

    out = args.out or f"eval_{args.algo}.csv"
    fieldnames = ["savestate", "episode", "reward", "steps", "survived",
                  "device", "flow_weight", "deterministic", "seed"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary = summarize_rows(rows)
    print(f"\n== {args.algo.upper()} ({len(rows)} episodes; "
          f"{summary['overall']['n_unique']} unique trajectories) ==")
    for track, s in summary["tracks"].items():
        print(f"  {track}: n={s['n']} unique={s['n_unique']} "
              f"reward={s['mean_reward']:.2f} ± {s['std_reward']:.2f} | "
              f"steps={s['mean_steps']:.1f} | survival={s['survived']}/{s['n']} "
              f"(95% CI {s['survival_ci95'][0]:.2f}-{s['survival_ci95'][1]:.2f})")
    print(f"CSV saved to: {out}")

    json_path = args.json or (os.path.splitext(out)[0] + ".json")
    payload = {
        "schema": 1,
        "algo": args.algo,
        "features": args.features,
        "model": os.path.abspath(args.model),
        "model_sha256": _sha256(args.model) if os.path.exists(args.model) else None,
        "requested_device": args.device,
        "device": rows[0].get("device") if rows else args.device,
        "flow_weight": args.flow_weight,
        "step_penalty": args.step_penalty,
        "max_steps": args.max_steps,
        "n_episodes": args.n_episodes,
        "deterministic": bool(args.deterministic),
        "seed": args.seed,
        "summary": summary,
        "rows": rows,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
