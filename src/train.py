import os
import argparse
import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from env import Mario64DSEnv

class MLflowCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                mlflow.log_metric("episode_reward", info["episode"]["r"], step=self.num_timesteps)
                mlflow.log_metric("episode_length", info["episode"]["l"], step=self.num_timesteps)
        return True

def make_env(rom_path, state_path, rank):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        env = Mario64DSEnv(rom_path=rom_path, state_path=state_path)
        env = Monitor(env)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="Train Mario 64 DS RL Agent")
    parser.add_argument("--rom", type=str, default="data/Super Mario 64 DS (USA) (Rev 1).nds", help="Path to NDS ROM")
    parser.add_argument("--state", type=str, default="data/Super Mario 64 DS (USA) (Rev 1).ds1", help="Path to Savestate")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps to train")
    parser.add_argument("--test-run", action="store_true", help="Run a short test to verify environment")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments to run")
    parser.add_argument("--run-id", type=str, default="ppo_mario64ds", help="Name/ID for this training run")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rom_full = os.path.join(base_dir, args.rom)
    state_full = os.path.join(base_dir, args.state)

    if not os.path.exists(rom_full):
        print(f"Warning: ROM not found at {rom_full}.")
    if not os.path.exists(state_full):
        print(f"Warning: Savestate not found at {state_full}.")

    mlflow_db = os.path.join(base_dir, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment("Mario64_NDS_RL")

    with mlflow.start_run(run_name=args.run_id):
        mlflow.log_param("model_type", "PPO")
        mlflow.log_param("total_timesteps", args.timesteps)
        mlflow.log_param("frameskip", 4)
        mlflow.log_param("num_envs", args.num_envs)
        
        env_fns = [make_env(rom_full, state_full, i) for i in range(args.num_envs)]
        env = SubprocVecEnv(env_fns)
        env = VecFrameStack(env, n_stack=4)

        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=os.path.join(base_dir, "tensorboard_logs"))
        
        mlflow.log_param("learning_rate", model.learning_rate)
        mlflow.log_param("batch_size", model.batch_size)

        print("Starting training...")
        timesteps = 1000 if args.test_run else args.timesteps
        
        models_dir = os.path.join(base_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=os.path.join(models_dir, 'checkpoints'),
            name_prefix=args.run_id,
            save_replay_buffer=False,
            save_vecnormalize=True
        )

        try:
            model.learn(total_timesteps=timesteps, callback=[MLflowCallback(), checkpoint_callback])
        except KeyboardInterrupt:
            print("\nTraining interrupted! Saving current progress...")

        model_path = os.path.join(models_dir, args.run_id)
        model.save(model_path)
        print(f"Training complete and model saved to {model_path}.")
        
        mlflow.log_artifact(f"{model_path}.zip", artifact_path="models")
        print("Model registered in MLflow.")

if __name__ == "__main__":
    main()
