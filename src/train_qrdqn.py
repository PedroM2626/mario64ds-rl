"""Treina QR-DQN (SB3-Contrib) no Mario 64 DS.

Este script não existia no repo, mas há ``models/qrdqn_mario64ds.zip`` e runs
``qrdqn_mario64ds`` no MLflow — ou seja, o modelo foi treinado com um script
ad-hoc não versionado. Esta versão versiona o procedimento e permite a
comparação justa QR-DQN vs PPO vs Rainbow com ``--features nature|impala``.

Exemplos:
    python -m src.train_qrdqn --run-id qrdqn_mario64ds --n-envs 4 --timesteps 500000
    python -m src.train_qrdqn --run-id qrdqn_impala --features impala --n-envs 4
"""

import os
import argparse

import mlflow
import numpy as np
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.env import Mario64DSEnv


def make_env(rom_path, state_path, rank=0, seed=0, max_steps=1350, frameskip=4):
    def _init():
        env = Mario64DSEnv(
            rom_path=rom_path, state_path=state_path,
            max_steps=max_steps, frameskip=frameskip,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


class MLflowCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        if self.n_calls % 1000 == 0 and len(self.model.ep_info_buffer) > 0:
            mlflow.log_metric("mean_reward",
                              float(np.mean([ep["r"] for ep in self.model.ep_info_buffer])),
                              step=self.num_timesteps)
            mlflow.log_metric("mean_episode_length",
                              float(np.mean([ep["l"] for ep in self.model.ep_info_buffer])),
                              step=self.num_timesteps)
        return True


def build_policy_kwargs(features: str):
    if features == "nature":
        return "CnnPolicy", {}
    if features == "impala":
        from src.sb3_impala import ImpalaFeaturesExtractor
        return "CnnPolicy", dict(
            features_extractor_class=ImpalaFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
    raise ValueError(features)


def main():
    parser = argparse.ArgumentParser(description="Train Mario 64 DS with QR-DQN")
    parser.add_argument("--rom", type=str, default="data/Super Mario 64 DS (USA) (Rev 1).nds")
    parser.add_argument("--run-id", type=str, default="qrdqn_mario64ds")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--features", type=str, default="nature", choices=["nature", "impala"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=1350)
    parser.add_argument("--frameskip", type=int, default=4)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rom_full = os.path.join(base_dir, args.rom)
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(base_dir, 'mlflow.db')}")
    mlflow.set_experiment("Mario64_NDS_RL")

    state_ds1 = os.path.join(base_dir, "data", "Super Mario 64 DS (USA) (Rev 1).ds1")
    state_ds3 = os.path.join(base_dir, "data", "Super Mario 64 DS (USA) (Rev 1).ds3")
    savestates = [state_ds1, state_ds3]

    train_envs = SubprocVecEnv([
        make_env(rom_full, savestates[i % 2], rank=i, seed=args.seed,
                 max_steps=args.max_steps, frameskip=args.frameskip)
        for i in range(args.n_envs)
    ])
    train_envs = VecFrameStack(train_envs, n_stack=4)
    train_envs = VecTransposeImage(train_envs)

    eval_env = SubprocVecEnv([make_env(rom_full, state_ds1, rank=1000, seed=args.seed,
                                       max_steps=args.max_steps, frameskip=args.frameskip)])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)

    log_path = os.path.join(base_dir, "tensorboard_logs", args.run_id)
    models_dir = os.path.join(base_dir, "models")
    best_model_dir = os.path.join(models_dir, args.run_id)
    os.makedirs(best_model_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=best_model_dir, log_path=log_path,
        eval_freq=5000, n_eval_episodes=3, deterministic=False, render=False,
    )

    policy, policy_kwargs = build_policy_kwargs(args.features)
    model = QRDQN(
        policy, train_envs, verbose=1, tensorboard_log=log_path,
        learning_rate=args.lr, batch_size=args.batch_size,
        # Mesmo motivo do Rainbow (OOM com FrameStack): buffer padrão de 1M
        # com obs (4,84,84) uint8 estoura ~26 GiB. 20k mantém ~1.5 GB.
        buffer_size=20000,
        seed=args.seed, policy_kwargs=policy_kwargs, device="auto",
    )
    print(f"Starting QR-DQN ({args.features}) | device={model.device} | envs={args.n_envs}")

    with mlflow.start_run(run_name=args.run_id):
        mlflow.log_param("model_type", f"QRDQN_{args.features.upper()}")
        mlflow.log_param("features", args.features)
        mlflow.log_param("total_timesteps", args.timesteps)
        mlflow.log_param("n_envs", args.n_envs)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("max_steps", args.max_steps)
        mlflow.log_param("frameskip", args.frameskip)
        try:
            model.learn(total_timesteps=args.timesteps,
                        callback=[eval_callback, MLflowCallback()],
                        progress_bar=True)
        finally:
            train_envs.close()
            eval_env.close()

        final_path = os.path.join(models_dir, f"{args.run_id}_final")
        model.save(final_path)
        mlflow.log_artifact(final_path + ".zip", artifact_path="models")

        import shutil
        best_path = os.path.join(best_model_dir, "best_model.zip")
        if os.path.exists(best_path):
            renamed = os.path.join(models_dir, f"{args.run_id}_best.zip")
            shutil.copy2(best_path, renamed)
            mlflow.log_artifact(renamed, artifact_path="models")
        print(f"Training complete! Models saved to {models_dir}")


if __name__ == "__main__":
    main()
