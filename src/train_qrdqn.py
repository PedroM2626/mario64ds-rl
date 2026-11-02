"""Trains QR-DQN (SB3-Contrib) on Mario 64 DS.

Enables controlled comparisons: QR-DQN vs PPO vs Rainbow with either NatureCNN
or IMPALA visual feature extractors.

Examples:
    python -m src.train_qrdqn --run-id qrdqn_mario64ds --n-envs 4 --timesteps 500000
    python -m src.train_qrdqn --run-id qrdqn_impala --features impala --n-envs 4
"""

import os
import argparse

try:
    import mlflow
except ImportError:
    mlflow = None
import numpy as np
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

from src.artifacts import log_model_reference
from src.envs import build_vec, make_env, savestate_paths


class MLflowCallback(BaseCallback):
    """Callback to log QR-DQN training metrics to MLflow every 1000 steps."""
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        if mlflow is not None and mlflow.active_run() and self.n_calls % 1000 == 0 and len(self.model.ep_info_buffer) > 0:
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
    parser.add_argument("--rom", type=str, default="data/Super Mario 64 DS (USA) (Rev 1).nds", help="Path to NDS ROM")
    parser.add_argument("--run-id", type=str, default="qrdqn_mario64ds", help="Name/ID for training run")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--features", type=str, default="nature", choices=["nature", "impala"],
                        help="Visual extractor: 'nature' or 'impala'")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=1350, help="Max steps per episode")
    parser.add_argument("--frameskip", type=int, default=4, help="Emulator frameskip")
    parser.add_argument("--states", type=str, default="ds1,ds3",
                        help="Savestates in the training mix, cycled across --n-envs")
    parser.add_argument("--flow-weight", type=float, default=1.0,
                        help="Optical-flow reward weight in the training env "
                             "(1.0 is the historical grid setting)")
    parser.add_argument("--eval-flow-weight", type=float, default=1.0,
                        help="Optical-flow weight in the evaluation env (default 1.0)")
    parser.add_argument("--step-penalty", type=float, default=0.02,
                        help="Per-step cost in the environment")
    parser.add_argument("--device", type=str, default="auto",
                        help="Torch device: auto, cpu or cuda")
    parser.add_argument("--eval-stochastic", action="store_true",
                        help="Use stochastic actions for checkpoint selection (default: deterministic)")
    parser.add_argument("--dist-reward", action="store_true", default=True,
                        help="Enable 3D distance potential reward shaping")
    parser.add_argument("--no-dist-reward", dest="dist_reward", action="store_false",
                        help="Disable 3D distance potential reward shaping")
    parser.add_argument("--dist-weight", type=float, default=0.05,
                        help="Weight for 3D potential distance shaping (default: 0.05)")
    parser.add_argument("--dist-mode", type=str, default="progress", choices=["progress", "potential"],
                        help="Distance reward mode: 'progress' (high-water mark ratchet, prevents jump exploits) or 'potential' (bidirectional)")
    parser.add_argument("--goal-reward", type=float, default=200.0,
                        help="Terminal bonus for reaching goal coordinates")
    parser.add_argument("--star-reward", type=float, default=500.0,
                        help="Terminal bonus for star collection")
    parser.add_argument("--use-ram-death", action="store_true", default=True,
                        help="Enable instantaneous RAM-based kill-plane detection")
    parser.add_argument("--buffer-size", type=int, default=100000,
                        help="Replay buffer capacity in transitions (default: 100000)")
    parser.add_argument("--exploration-fraction", type=float, default=0.3,
                        help="Fraction of entire training period over which exploration rate is annealed (default: 0.3)")
    parser.add_argument("--exploration-final-eps", type=float, default=0.05,
                        help="Final value of random action probability (default: 0.05)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rom_full = os.path.join(base_dir, args.rom)
    if mlflow is not None:
        mlflow.set_tracking_uri(f"sqlite:///{os.path.join(base_dir, 'mlflow.db')}")
        mlflow.set_experiment("Mario64_NDS_RL")

    savestates = savestate_paths(args.states, base_dir)
    for sp in savestates:
        if not os.path.exists(sp):
            raise FileNotFoundError(f"Savestate not found: {sp}")

    train_envs = SubprocVecEnv([
        make_env(rom_full, savestates[i % len(savestates)], rank=i, seed=args.seed,
                 max_steps=args.max_steps, frameskip=args.frameskip,
                 step_penalty=args.step_penalty, flow_weight=args.flow_weight,
                 dist_reward_enabled=args.dist_reward, dist_weight=args.dist_weight,
                 goal_reward=args.goal_reward, star_reward=args.star_reward,
                 use_ram_death=args.use_ram_death, dist_reward_mode=args.dist_mode)
        for i in range(args.n_envs)
    ])
    train_envs = VecFrameStack(train_envs, n_stack=4)
    train_envs = VecTransposeImage(train_envs)

    eval_states = list(dict.fromkeys(savestates))
    eval_env = build_vec(
        rom_full, eval_states, rank0=1000, seed=args.seed,
        max_steps=args.max_steps, frameskip=args.frameskip,
        step_penalty=args.step_penalty, flow_weight=args.eval_flow_weight,
        dist_reward_enabled=args.dist_reward, dist_weight=args.dist_weight,
        goal_reward=args.goal_reward, star_reward=args.star_reward,
        use_ram_death=args.use_ram_death, dist_reward_mode=args.dist_mode,
    )

    log_path = os.path.join(base_dir, "tensorboard_logs", args.run_id)
    models_dir = os.path.join(base_dir, "models")
    best_model_dir = os.path.join(models_dir, args.run_id)
    os.makedirs(best_model_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=best_model_dir, log_path=log_path,
        eval_freq=5000, n_eval_episodes=3,
        deterministic=not args.eval_stochastic, render=False,
    )

    policy, policy_kwargs = build_policy_kwargs(args.features)
    model = QRDQN(
        policy, train_envs, verbose=1, tensorboard_log=log_path,
        learning_rate=args.lr, batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        seed=args.seed, policy_kwargs=policy_kwargs, device=args.device,
    )
    print(f"Starting QR-DQN ({args.features}) | device={model.device} | envs={args.n_envs} | buffer={args.buffer_size}")

    params = {
        "model_type": f"QRDQN_{args.features.upper()}",
        "features": args.features,
        "total_timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "exploration_fraction": args.exploration_fraction,
        "exploration_final_eps": args.exploration_final_eps,
        "max_steps": args.max_steps,
        "frameskip": args.frameskip,
        "states": args.states,
        "flow_weight": args.flow_weight,
        "eval_flow_weight": args.eval_flow_weight,
        "step_penalty": args.step_penalty,
        "dist_reward": args.dist_reward,
        "dist_weight": args.dist_weight,
        "dist_mode": args.dist_mode,
        "goal_reward": args.goal_reward,
        "star_reward": args.star_reward,
        "use_ram_death": args.use_ram_death,
        "eval_deterministic": not args.eval_stochastic,
        "requested_device": args.device,
        "resolved_device": str(model.device),
    }

    from contextlib import nullcontext
    run_ctx = mlflow.start_run(run_name=args.run_id) if mlflow is not None else nullcontext()
    with run_ctx:
        if mlflow is not None and mlflow.active_run():
            for k, v in params.items():
                mlflow.log_param(k, v)
        try:
            callbacks = [eval_callback]
            if mlflow is not None:
                callbacks.append(MLflowCallback())
            model.learn(total_timesteps=args.timesteps,
                        callback=callbacks,
                        progress_bar=False)
        finally:
            train_envs.close()
            eval_env.close()

        final_path = os.path.join(models_dir, f"{args.run_id}_final")
        model.save(final_path)
        log_model_reference(final_path + ".zip")

        import shutil
        best_path = os.path.join(best_model_dir, "best_model.zip")
        if os.path.exists(best_path):
            renamed = os.path.join(models_dir, f"{args.run_id}_best.zip")
            shutil.copy2(best_path, renamed)
            log_model_reference(renamed)
        print(f"Training complete! Models saved to {models_dir}")


if __name__ == "__main__":
    main()
