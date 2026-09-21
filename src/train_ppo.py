import os
import argparse
import mlflow
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.env import Mario64DSEnv


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


class MLflowCallback(BaseCallback):
    """Callback to log PPO training metrics to MLflow every 1000 steps."""
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        if self.n_calls % 1000 == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                mean_len = np.mean([ep['l'] for ep in self.model.ep_info_buffer])
                mlflow.log_metric("mean_reward", mean_reward, step=self.num_timesteps)
                mlflow.log_metric("mean_episode_length", mean_len, step=self.num_timesteps)
        return True


def build_policy_kwargs(features: str):
    """Returns (policy, policy_kwargs) for fair architectural comparison.

    - ``nature``: Standard SB3 CnnPolicy (NatureCNN).
    - ``impala``: CnnPolicy with custom ``ImpalaFeaturesExtractor``
      (identical residual blocks to Rainbow in ``src/impala_cnn.py``).
    """
    if features == "nature":
        return "CnnPolicy", {}
    if features == "impala":
        from src.sb3_impala import ImpalaFeaturesExtractor
        return "CnnPolicy", dict(
            features_extractor_class=ImpalaFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
    raise ValueError(f"--features must be 'nature' or 'impala', received: {features}")


def main():
    parser = argparse.ArgumentParser(description="Train Mario 64 DS RL Agent with PPO (NatureCNN or IMPALA)")
    parser.add_argument("--rom", type=str, default="data/Super Mario 64 DS (USA) (Rev 1).nds", help="Path to NDS ROM")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total timesteps to train")
    parser.add_argument("--run-id", type=str, default="ppo_mario64ds_baseline", help="Name/ID for this training run")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--resume", type=str, default=None, help="Path to a previous model (.zip) to resume training")
    parser.add_argument("--features", type=str, default="nature", choices=["nature", "impala"],
                        help="Visual extractor: 'nature' (SB3 baseline) or 'impala' (fair comparison with Rainbow)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--states", type=str, default="ds1,ds3",
                        help="Savestates included in training mix (e.g., 'ds1,ds2,ds3' to generalize across 3 tracks)")
    parser.add_argument("--max-steps", type=int, default=1350, help="Maximum steps per episode (must match env)")
    parser.add_argument("--frameskip", type=int, default=4, help="Emulator frameskip")
    parser.add_argument("--step-penalty", type=float, default=0.02,
                        help="Time penalty per step (anti-camping)")
    parser.add_argument("--flow-weight", type=float, default=2.5,
                        help="Weight for optical flow reward (2.5 validated: dense forward progress "
                             "signal overcomes abyss death noise; 0.5/1.0 causes fresh runs to degrade)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rom_full = os.path.join(base_dir, args.rom)

    mlflow_db = os.path.join(base_dir, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment("Mario64_NDS_RL")

    savestates = [
        os.path.join(base_dir, "data", f"Super Mario 64 DS (USA) (Rev 1).{s.strip()}")
        for s in args.states.split(",") if s.strip()
    ]
    for sp in savestates:
        if not os.path.exists(sp):
            raise FileNotFoundError(f"Savestate not found: {sp}")
    env_fns = []
    for i in range(args.n_envs):
        state = savestates[i % len(savestates)]
        env_fns.append(make_env(rom_full, state, rank=i, seed=args.seed,
                                max_steps=args.max_steps, frameskip=args.frameskip,
                                step_penalty=args.step_penalty,
                                flow_weight=args.flow_weight))

    train_envs = SubprocVecEnv(env_fns)
    train_envs = VecFrameStack(train_envs, n_stack=4)
    # Explicit transpose: (B, 84, 84, 4) HWC -> (B, 4, 84, 84) CHW for CNN input
    train_envs = VecTransposeImage(train_envs)

    # Eval env (single instance, primary savestate = first in mix)
    eval_env = SubprocVecEnv([make_env(rom_full, savestates[0], rank=1000, seed=args.seed,
                                       max_steps=args.max_steps, frameskip=args.frameskip)])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)

    # TensorBoard logging
    log_path = os.path.join(base_dir, "tensorboard_logs", args.run_id)

    # Evaluation callback
    # Dedicated directory per run_id prevents checkpoint collisions during concurrent runs
    models_dir = os.path.join(base_dir, 'models')
    best_model_dir = os.path.join(models_dir, args.run_id)
    os.makedirs(best_model_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_path,
        eval_freq=5000,
        n_eval_episodes=3,
        deterministic=False,
        render=False
    )

    policy, policy_kwargs = build_policy_kwargs(args.features)

    if args.resume:
        print(f"Resuming training from {args.resume}...")
        model = PPO.load(args.resume, env=train_envs, device="auto", custom_objects={"tensorboard_log": log_path})
    else:
        model = PPO(
            policy,
            train_envs,
            verbose=1,
            tensorboard_log=log_path,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Encourages exploration
            seed=args.seed,
            policy_kwargs=policy_kwargs,
            device="auto"
        )

    print(f"Starting PPO ({args.features}) training with {args.n_envs} parallel environments...")
    print(f"Total timesteps: {args.timesteps}")
    print(f"Device: {model.device}")

    with mlflow.start_run(run_name=args.run_id):
        mlflow.log_param("model_type", f"PPO_{args.features.upper()}")
        mlflow.log_param("features", args.features)
        mlflow.log_param("total_timesteps", args.timesteps)
        mlflow.log_param("n_envs", args.n_envs)
        mlflow.log_param("learning_rate", 3e-4)
        mlflow.log_param("n_steps", 256)
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("ent_coef", 0.01)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("max_steps", args.max_steps)
        mlflow.log_param("frameskip", args.frameskip)

        try:
            model.learn(
                total_timesteps=args.timesteps,
                callback=[eval_callback, MLflowCallback()],
                progress_bar=True,
                reset_num_timesteps=not bool(args.resume)
            )
        finally:
            train_envs.close()
            eval_env.close()

        # Save final model
        final_path = os.path.join(models_dir, f"{args.run_id}_final")
        model.save(final_path)
        mlflow.log_artifact(final_path + ".zip", artifact_path="models")

        # EvalCallback saves the best model in <models>/<run_id>/best_model.zip
        best_path = os.path.join(best_model_dir, "best_model.zip")
        if os.path.exists(best_path):
            import shutil
            renamed = os.path.join(models_dir, f"{args.run_id}_best.zip")
            shutil.copy2(best_path, renamed)
            mlflow.log_artifact(renamed, artifact_path="models")

        print(f"Training complete! Models saved to {models_dir}")


if __name__ == "__main__":
    main()
