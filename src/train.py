import os
import argparse
import mlflow
import torch
import numpy as np

from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, PrioritizedVectorReplayBuffer
from tianshou.algorithm.modelfree.rainbow import RainbowDQN
from tianshou.algorithm.modelfree.c51 import C51Policy
from tianshou.algorithm.optim import TorchOptimizerFactory
from tianshou.trainer import OffPolicyTrainer, OffPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

from gymnasium.wrappers import FrameStackObservation
from src.env import Mario64DSEnv
from src.impala_cnn import ImpalaCNN, TianshouNatureCNN, RainbowNet

# Suporte C51: recompensa de morte (-100) e teto de timeout definem o suporte.
V_MIN = -100.0
V_MAX = 100.0


def make_env(rom_path, state_path, seed=0, max_steps=450, frameskip=4):
    def _init():
        env = Mario64DSEnv(
            rom_path=rom_path, state_path=state_path,
            max_steps=max_steps, frameskip=frameskip,
        )
        # FrameStack empilha os ultimos 4 frames num array
        env = FrameStackObservation(env, stack_size=4)
        env.reset(seed=seed)
        return env
    return _init


def build_feature_net(features: str):
    """Comparação justa: 'impala' (default, residual) vs 'nature' (Mnih et al.)."""
    if features == "impala":
        # A FrameStack vai enviar arrays de shape (84, 84, 4), então c=4
        return ImpalaCNN(c=4, h=84, w=84, features_dim=256)
    if features == "nature":
        return TianshouNatureCNN(c=4, h=84, w=84, features_dim=512)
    raise ValueError(f"--features deve ser 'impala' ou 'nature', recebido: {features}")


def main():
    parser = argparse.ArgumentParser(description="Train Mario 64 DS RL Agent with Tianshou Rainbow")
    parser.add_argument("--rom", type=str, default="data/Super Mario 64 DS (USA) (Rev 1).nds", help="Path to NDS ROM")
    parser.add_argument("--state", type=str, default="data/Super Mario 64 DS (USA) (Rev 1).ds1", help="Path to Savestate")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total timesteps to train")
    parser.add_argument("--test-run", action="store_true", help="Run a short test to verify environment")
    parser.add_argument("--run-id", type=str, default="rainbow_mario64ds", help="Name/ID for this training run")
    parser.add_argument("--features", type=str, default="impala", choices=["impala", "nature"],
                        help="Extrator visual: 'impala' (default) ou 'nature' (comparação justa com PPO)")
    parser.add_argument("--seed", type=int, default=0, help="Seed para reproducibilidade")
    parser.add_argument("--max-steps", type=int, default=450, help="Passos máximos por episódio")
    parser.add_argument("--frameskip", type=int, default=4, help="Frameskip do emulador")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rom_full = os.path.join(base_dir, args.rom)

    mlflow_db = os.path.join(base_dir, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment("Mario64_NDS_RL")

    state_ds1 = os.path.join(base_dir, "data", "Super Mario 64 DS (USA) (Rev 1).ds1")
    state_ds3 = os.path.join(base_dir, "data", "Super Mario 64 DS (USA) (Rev 1).ds3")

    # IMPORTANTE: O DeSmuME C-Library causa Access Violation se instanciarmos multiplos emuladores
    # no mesmo processo Python. Por isso o Tianshou PRECISA usar SubprocVectorEnv para separar na RAM.
    env_fns = [
        make_env(rom_full, state_ds1, seed=args.seed, max_steps=args.max_steps, frameskip=args.frameskip),
        make_env(rom_full, state_ds3, seed=args.seed + 1, max_steps=args.max_steps, frameskip=args.frameskip)
    ]
    train_envs = SubprocVectorEnv(env_fns)
    test_envs = SubprocVectorEnv([make_env(rom_full, state_ds1, seed=args.seed + 999,
                                           max_steps=args.max_steps, frameskip=args.frameskip)])

    # Architecture
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feature_net = build_feature_net(args.features)

    action_shape = train_envs.action_space[0].n if isinstance(train_envs.action_space, list) else train_envs.action_space[0].n
    num_atoms = 51

    # Nossa rede RainbowNet com NoisyLinear e Dueling
    model = RainbowNet(feature_net, action_shape, num_atoms, noisy_std=0.5).to(device)

    # Tianshou 2.0 API separa a politica (Network) do Algorithm (Loop logic)
    # .to(device) é OBRIGATÓRIO: o C51Policy guarda o buffer `support` na CPU,
    # e sem isso o treino em CUDA explode com device mismatch (cuda:0 vs cpu).
    policy = C51Policy(
        model=model,
        action_space=train_envs.action_space[0],
        num_atoms=num_atoms,
        v_min=V_MIN,
        v_max=V_MAX
    ).to(device)

    optim_factory = TorchOptimizerFactory(torch.optim.Adam, lr=1e-4)

    algorithm = RainbowDQN(
        policy=policy,
        optim=optim_factory,
        gamma=0.99,
        n_step_return_horizon=3,
        target_update_freq=500
    )

    # Buffer de Prioridade (PER - Prioritized Experience Replay) nativo do Tianshou!
    # O tamanho foi reduzido para 20000 para evitar estouro de memória (OOM) no deepcopy interno do Tianshou.
    buffer = PrioritizedVectorReplayBuffer(total_size=20000, buffer_num=len(env_fns), alpha=0.6, beta=0.4)

    train_collector = Collector(algorithm, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(algorithm, test_envs, exploration_noise=False)

    # Logger Tensorboard integrado ao MLflow
    log_path = os.path.join(base_dir, "tensorboard_logs", args.run_id)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(alg):
        models_dir = os.path.join(base_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        torch.save(alg.policy.model.state_dict(), os.path.join(models_dir, f"{args.run_id}_best.pth"))

    print(f"Starting training with Tianshou Rainbow DQN ({args.features})...")
    timesteps = 1000 if args.test_run else args.timesteps

    # Tianshou 2.0 OffPolicyTrainer
    params = OffPolicyTrainerParams(
        training_collector=train_collector,
        test_collector=test_collector,
        max_epochs=max(1, timesteps // 1000),
        epoch_num_steps=1000,
        batch_size=32,
        collection_step_num_env_steps=10,
        update_step_num_gradient_steps_per_sample=0.1,
        test_step_num_episodes=2,
        save_best_fn=save_best_fn,
        logger=logger
    )

    trainer = OffPolicyTrainer(algorithm=algorithm, params=params)

    with mlflow.start_run(run_name=args.run_id):
        mlflow.log_param("model_type", f"Tianshou_Rainbow_{args.features.capitalize()}")
        mlflow.log_param("features", args.features)
        mlflow.log_param("total_timesteps", timesteps)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("max_steps", args.max_steps)
        mlflow.log_param("frameskip", args.frameskip)
        mlflow.log_param("v_min", V_MIN)
        mlflow.log_param("v_max", V_MAX)
        try:
            result = trainer.run()
            print(f"Finished training! Result: {result}")
        finally:
            train_envs.close()
            test_envs.close()

        model_path = os.path.join(base_dir, 'models', f"{args.run_id}_final.pth")
        torch.save(algorithm.policy.model.state_dict(), model_path)
        mlflow.log_artifact(model_path, artifact_path="models")

if __name__ == "__main__":
    main()
