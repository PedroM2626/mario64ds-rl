"""Curriculum learning: progressively master all 3 tracks without catastrophic forgetting.

Empirical motivation (see README §Three-Track Generalization):
- Fine-tuning a 2-way 500k model with ds2 in the mix at lr=3e-4 caused
  catastrophic forgetting of ds1 (Experiment A).
- Training 2M total steps at lr=3e-4 resulted in 0/3 deterministic survival (Experiment D):
  additional steps do not solve interference within the shared CNN.

Curriculum strategy:
1. Start from the best checkpoint (2-way 500k: ds1✓ ds3✓).
2. Execute phases with DECAYING LEARNING RATES (1e-4 -> 5e-5 -> 2.5e-5):
   smaller updates protect previously consolidated skills while acquiring the new track.
3. Include all tracks in the training mix across ALL phases.
4. Select checkpoints via BALANCED EVALUATION: EvalCallback evaluates deterministically
   across all 3 tracks, saving the checkpoint with the highest MEAN reward (a single-track
   specialist yields mean ~-33; a model surviving all 3 tracks yields mean ~+63).
5. Early termination when all 3 tracks achieve full survival.

Usage:
    python -m src.train_curriculum --run-id ppo_curriculum
    python -m src.train_curriculum --run-id ppo_curriculum \
        --start-model models/grid_ppo_nature_s0_500k_best.zip \
        --phases "200000:1e-4,200000:5e-5,200000:2.5e-5"
"""

import argparse
import csv
import os

import mlflow
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage

from src.env import Mario64DSEnv
from src.train_ppo import MLflowCallback

PHASES_DEFAULT = "200000:1e-4,200000:5e-5,200000:2.5e-5"


def make_env(rom_path, state_path, rank=0, seed=0, max_steps=1350, frameskip=4,
             step_penalty=0.02, flow_weight=1.0):
    def _init():
        env = Mario64DSEnv(
            rom_path=rom_path, state_path=state_path,
            max_steps=max_steps, frameskip=frameskip,
            step_penalty=step_penalty, flow_weight=flow_weight,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def build_vec(rom_path, states, rank0, seed, max_steps, frameskip, n_stack=4,
              step_penalty=0.02, flow_weight=1.0):
    vec = SubprocVecEnv([
        make_env(rom_path, sp, rank=rank0 + i, seed=seed,
                 max_steps=max_steps, frameskip=frameskip,
                 step_penalty=step_penalty, flow_weight=flow_weight)
        for i, sp in enumerate(states)
    ])
    vec = VecFrameStack(vec, n_stack=n_stack)
    vec = VecTransposeImage(vec)
    return vec


def eval_tracks(model_path, rom_path, states, seed, n_eps=3, max_steps=1350,
                step_penalty=0.02, flow_weight=2.5):
    """Deterministic evaluation per track (sequential SubprocVecEnv for process isolation)."""
    from src.eval import _load_sb3_with_legacy_patch
    model = _load_sb3_with_legacy_patch(PPO, model_path, device="auto")

    rows = []
    for s_idx, sp in enumerate(states):
        vec = build_vec(rom_path, [sp], 2000 + s_idx, seed, max_steps, 4,
                        step_penalty=step_penalty, flow_weight=flow_weight)
        track_label = os.path.basename(sp).split(".")[-1]
        for ep in range(n_eps):
            obs = vec.reset()
            done, ep_rew, steps = False, 0.0, 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rews, dones, infos = vec.step(action)
                ep_rew += float(rews[0])
                steps += 1
                done = bool(dones[0])
            rows.append({"track": track_label, "episode": ep,
                         "reward": ep_rew, "steps": steps,
                         "survived": steps >= max_steps})
        vec.close()
    return rows

# Backward compatibility alias
eval_pistas = eval_tracks


def main():
    parser = argparse.ArgumentParser(description="Curriculum learning across all 3 tracks")
    parser.add_argument("--rom", type=str, default="data/Super Mario 64 DS (USA) (Rev 1).nds")
    parser.add_argument("--run-id", type=str, default="ppo_curriculum")
    parser.add_argument("--start-model", type=str, default="models/grid_ppo_nature_s0_500k_best.zip",
                        help="Initial checkpoint (default: 2-way 500k, ds1✓ ds3✓)")
    parser.add_argument("--states", type=str, default="ds1,ds2,ds3")
    parser.add_argument("--phases", type=str, default=PHASES_DEFAULT,
                        help="Comma-separated 'steps:lr' phases (decaying learning rate schedule)")
    parser.add_argument("--n-envs", type=int, default=3)
    parser.add_argument("--eval-freq", type=int, default=10000,
                        help="Steps between EvalCallback evaluations (must be < phase steps)")
    parser.add_argument("--step-penalty", type=float, default=0.02,
                        help="Per-step penalty (anti-camping)")
    parser.add_argument("--flow-weight", type=float, default=2.5,
                        help="Optical flow weight (2.5: dense forward signal overcoming abyss death noise)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    phases = []
    for token in args.phases.split(","):
        steps_s, lr_s = token.strip().split(":")
        phases.append((int(steps_s), float(lr_s)))

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rom_full = os.path.join(base_dir, args.rom)
    states = [
        os.path.join(base_dir, "data", f"Super Mario 64 DS (USA) (Rev 1).{s.strip()}")
        for s in args.states.split(",") if s.strip()
    ]
    for sp in states:
        if not os.path.exists(sp):
            raise FileNotFoundError(f"Savestate not found: {sp}")
    if not os.path.exists(os.path.join(base_dir, args.start_model)):
        raise FileNotFoundError(f"Initial checkpoint not found: {args.start_model}")

    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(base_dir, 'mlflow.db')}")
    mlflow.set_experiment("Mario64_NDS_RL")

    models_dir = os.path.join(base_dir, "models")
    results_csv = os.path.join(base_dir, "results_curriculum.csv")

    all_rows = []
    current_model = args.start_model
    global_best_path, global_best_mean = None, -np.inf

    with mlflow.start_run(run_name=args.run_id):
        mlflow.log_param("model_type", "PPO_Nature_Curriculum")
        mlflow.log_param("start_model", args.start_model)
        mlflow.log_param("states", args.states)
        mlflow.log_param("phases", args.phases)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("n_envs", args.n_envs)

        for i, (steps, lr) in enumerate(phases, 1):
            phase_id = f"{args.run_id}_p{i}"
            print(f"\n===== PHASE {i}/{len(phases)}: {steps} steps @ lr={lr} "
                  f"(resume: {current_model}) =====", flush=True)

            train_envs = build_vec(rom_full, states, 0, args.seed, 1350, 4, n_stack=4,
                                   step_penalty=args.step_penalty, flow_weight=args.flow_weight)
            # Eval env contains ALL tracks (1 env per track): mean reward
            # from EvalCallback selects the balanced model across all geometries.
            eval_env = build_vec(rom_full, states, 1000 + i, args.seed, 1350, 4, n_stack=4,
                                 step_penalty=args.step_penalty, flow_weight=args.flow_weight)

            model = PPO.load(current_model, env=train_envs, device="auto")
            # Override learning rate schedule with current curriculum phase LR
            model.learning_rate = lr
            model.lr_schedule = lambda _: lr
            print(f"Device: {model.device} | phase lr: {lr}", flush=True)

            best_model_dir = os.path.join(models_dir, phase_id)
            os.makedirs(best_model_dir, exist_ok=True)
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=best_model_dir,
                log_path=os.path.join(base_dir, "tensorboard_logs", phase_id),
                eval_freq=args.eval_freq,
                n_eval_episodes=3,   # 3 episodes x 3 envs = 9 evaluation episodes
                deterministic=True,  # Survival consistency is primary objective
                render=False,
            )

            try:
                model.learn(total_timesteps=steps,
                            callback=[eval_callback, MLflowCallback()],
                            progress_bar=True,
                            reset_num_timesteps=False)
            finally:
                train_envs.close()
                eval_env.close()

            # Deterministic per-track evaluation of phase best checkpoint
            phase_best = os.path.join(best_model_dir, "best_model.zip")
            if not os.path.exists(phase_best):
                print(f"PHASE {i}: best_model.zip not found; skipping evaluation", flush=True)
                continue

            rows = eval_tracks(phase_best, rom_full, states, args.seed, n_eps=3,
                               max_steps=1350,
                               step_penalty=args.step_penalty,
                               flow_weight=args.flow_weight)
            track_means = {}
            for track in sorted(set(r["track"] for r in rows)):
                vv = [r for r in rows if r["track"] == track]
                track_means[track] = (np.mean([r["reward"] for r in vv]),
                                      sum(1 for r in vv if r["survived"]), len(vv))
            balanced_mean = float(np.mean([m[0] for m in track_means.values()]))
            all_survived = all(m[1] == m[2] for m in track_means.values())

            for track, (mean_r, surv, n) in track_means.items():
                print(f"  {track}: {mean_r:.2f} · surv {surv}/{n}", flush=True)
                mlflow.log_metric(f"{track}_reward", mean_r, step=i)
                mlflow.log_metric(f"{track}_surv", surv / n, step=i)
            mlflow.log_metric("balanced_mean", balanced_mean, step=i)
            print(f"  BALANCED MEAN: {balanced_mean:.2f} | all tracks survived: {all_survived}", flush=True)

            for r in rows:
                r["phase"] = i
                r["lr"] = lr
                all_rows.append(r)

            if balanced_mean > global_best_mean:
                global_best_mean = balanced_mean
                global_best_path = phase_best

            # Next phase resumes from the balanced best of this phase
            current_model = phase_best

            if all_survived:
                print(f"\nCONVERGED in phase {i}: all 3 tracks survived!", flush=True)
                break

        # Consolidate CSV
        with open(results_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["phase", "lr", "track", "episode",
                                              "reward", "steps", "survived"])
            w.writeheader()
            w.writerows(all_rows)

        # Save global best
        if global_best_path:
            import shutil
            final_dst = os.path.join(models_dir, f"{args.run_id}_best.zip")
            shutil.copy2(global_best_path, final_dst)
            mlflow.log_artifact(final_dst, artifact_path="models")
            print(f"\nBest balanced model: {final_dst} (mean reward {global_best_mean:.2f})", flush=True)

        print(f"Curriculum completed. CSV saved to: {results_csv}", flush=True)


if __name__ == "__main__":
    main()
