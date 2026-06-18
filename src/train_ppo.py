import os
import argparse
import mlflow
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter

from src.env import Mario64DSEnv


def make_env(rom_path, state_path, rank=0):
    def _init():
        env = Mario64DSEnv(rom_path=rom_path, state_path=state_path)
        env = Monitor(env)
        return env
    return _init


class MLflowCallback(BaseCallback):
    """Callback para logar métricas do PPO no MLflow a cada 1000 passos."""
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


def main():
    parser = argparse.ArgumentParser(description="Train Mario 64 DS RL Agent with PPO + NatureCNN (Baseline)")
    parser.add_argument("--rom", type=str, default="data/Super Mario 64 DS (USA) (Rev 1).nds", help="Path to NDS ROM")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total timesteps to train")
    parser.add_argument("--run-id", type=str, default="ppo_mario64ds_baseline", help="Name/ID for this training run")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rom_full = os.path.join(base_dir, args.rom)
    
    mlflow_db = os.path.join(base_dir, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment("Mario64_NDS_RL")

    state_ds1 = os.path.join(base_dir, "data", "Super Mario 64 DS (USA) (Rev 1).ds1")
    state_ds3 = os.path.join(base_dir, "data", "Super Mario 64 DS (USA) (Rev 1).ds3")
    
    # PPO permite múltiplas instâncias paralelas via SubprocVecEnv
    # Isso acelera drasticamente a coleta de experiências
    savestates = [state_ds1, state_ds3]
    env_fns = []
    for i in range(args.n_envs):
        state = savestates[i % len(savestates)]
        env_fns.append(make_env(rom_full, state, rank=i))
    
    train_envs = SubprocVecEnv(env_fns)
    train_envs = VecFrameStack(train_envs, n_stack=4)
    
    # Eval env (1 instância)
    eval_env = SubprocVecEnv([make_env(rom_full, state_ds1)])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # Logger Tensorboard
    log_path = os.path.join(base_dir, "tensorboard_logs", args.run_id)
    
    # Callback de avaliação
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=log_path,
        eval_freq=5000,
        n_eval_episodes=3,
        deterministic=False,
        render=False
    )
    
    # PPO com NatureCNN (policy_kwargs padrão do SB3 para imagens)
    model = PPO(
        "CnnPolicy",  # NatureCNN é o padrão do SB3 para observações de imagem
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
        ent_coef=0.01,  # Incentiva exploração
        device="auto"
    )

    print(f"Starting PPO training with {args.n_envs} parallel environments...")
    print(f"Total timesteps: {args.timesteps}")
    print(f"Device: {model.device}")
    
    with mlflow.start_run(run_name=args.run_id):
        mlflow.log_param("model_type", "PPO_NatureCNN_Baseline")
        mlflow.log_param("total_timesteps", args.timesteps)
        mlflow.log_param("n_envs", args.n_envs)
        mlflow.log_param("learning_rate", 3e-4)
        mlflow.log_param("n_steps", 256)
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("ent_coef", 0.01)
        
        try:
            model.learn(
                total_timesteps=args.timesteps,
                callback=[eval_callback, MLflowCallback()],
                progress_bar=True
            )
        finally:
            train_envs.close()
            eval_env.close()
        
        # Salvar modelo final
        final_path = os.path.join(models_dir, f"{args.run_id}_final")
        model.save(final_path)
        mlflow.log_artifact(final_path + ".zip", artifact_path="models")
        
        # O EvalCallback salva o melhor modelo automaticamente como "best_model.zip"
        best_path = os.path.join(models_dir, "best_model.zip")
        if os.path.exists(best_path):
            # Renomear para incluir o run_id
            import shutil
            renamed = os.path.join(models_dir, f"{args.run_id}_best.zip")
            shutil.copy2(best_path, renamed)
            mlflow.log_artifact(renamed, artifact_path="models")
        
        print(f"Training complete! Models saved to {models_dir}")


if __name__ == "__main__":
    main()
