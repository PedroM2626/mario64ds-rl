import os
import time
import argparse
import torch
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from tianshou.data import Batch
from tianshou.env import DummyVectorEnv

from src.env import Mario64DSEnv
from src.impala_cnn import ImpalaCNN, TianshouNatureCNN, RainbowNet
from tianshou.algorithm.modelfree.c51 import C51Policy


def load_rainbow_policy(model_path, device, features="impala"):
    """Loads Rainbow DQN policy (IMPALA or Nature)."""
    if features == "impala":
        feature_net = ImpalaCNN(c=4, h=84, w=84, features_dim=256)
    elif features == "nature":
        feature_net = TianshouNatureCNN(c=4, h=84, w=84, features_dim=512)
    else:
        raise ValueError(f"features must be 'impala' or 'nature', received: {features}")
    action_shape = 6
    num_atoms = 51
    model = RainbowNet(feature_net, action_shape, num_atoms, noisy_std=0.5).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Rainbow ({features}) weights loaded from {model_path}")
    else:
        print(f"WARNING: Weights not found at {model_path}. Using random weights.")

    policy = C51Policy(
        model=model,
        action_space=gym.spaces.Discrete(action_shape),
        num_atoms=num_atoms,
        v_min=-100.0,
        v_max=100.0
    ).to(device)
    policy.eval()
    return policy


def load_sb3_model(algo, model_path):
    """Loads SB3 model (PPO or QR-DQN). Extractor is restored from checkpoint."""
    if algo == "ppo":
        from stable_baselines3 import PPO as Algo
    elif algo == "qrdqn":
        from sb3_contrib import QRDQN as Algo
    else:
        raise ValueError(algo)
    if not os.path.exists(model_path):
        print(f"WARNING: {algo.upper()} model not found at {model_path}")
        return None
    try:
        model = Algo.load(model_path)
        print(f"{algo.upper()} model loaded from {model_path}")
        return model
    except TypeError as e:
        # Legacy QRDQN model referencing src.impala_cnn.ImpalaCNN
        if "Box" not in str(e):
            raise
        import src.impala_cnn
        from src.sb3_impala import ImpalaFeaturesExtractor
        orig = src.impala_cnn.ImpalaCNN
        src.impala_cnn.ImpalaCNN = ImpalaFeaturesExtractor
        try:
            model = Algo.load(model_path)
            print(f"{algo.upper()} (legacy, IMPALA patch) loaded from {model_path}")
            return model
        finally:
            src.impala_cnn.ImpalaCNN = orig


def play_rainbow(policy, rom_path, savestates, device, deterministic=False):
    """Visualization using Rainbow DQN, recreating an isolated environment per savestate."""
    for idx, state_path in enumerate(savestates):
        print(f"\n--- Rainbow DQN - Savestate {idx + 1}: {os.path.basename(state_path)} ---")

        def make_eval_env(sp=state_path):
            env = Mario64DSEnv(rom_path, sp, render_mode='human')
            env = FrameStackObservation(env, stack_size=4)
            return env

        env = DummyVectorEnv([make_eval_env])
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            batch = Batch(obs=obs, info=info)
            with torch.no_grad():
                result = policy(batch)
                action = result.act
                if deterministic:
                    # C51 is distributional; greedy mode = argmax over atom mean distribution.
                    pass

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated[0] or truncated[0]
            total_reward += reward[0]
            steps += 1
            time.sleep(0.01)

        print(f"Finished: reward={total_reward:.2f}, steps={steps}")
        env.close()


def play_sb3(model, rom_path, savestates, label="PPO", deterministic=False):
    """Visualization using an SB3 model (PPO or QR-DQN), recreating environment per savestate."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
    from stable_baselines3.common.monitor import Monitor

    for idx, state_path in enumerate(savestates):
        print(f"\n--- {label} - Savestate {idx + 1}: {os.path.basename(state_path)} ---")

        def make_env(sp=state_path):
            env = Mario64DSEnv(rom_path, sp, render_mode='human')
            env = Monitor(env)
            return env

        vec_env = DummyVecEnv([make_env])
        vec_env = VecFrameStack(vec_env, n_stack=4)
        vec_env = VecTransposeImage(vec_env)

        obs = vec_env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = vec_env.step(action)
            total_reward += rewards[0]
            done = dones[0]
            steps += 1
            time.sleep(0.01)

        print(f"Finished: reward={total_reward:.2f}, steps={steps}")
        vec_env.close()


def play():
    parser = argparse.ArgumentParser(description="Visualize Mario 64 DS RL Agent")
    parser.add_argument("--algo", type=str, default="rainbow", choices=["rainbow", "ppo", "qrdqn"],
                        help="Algorithm: 'rainbow' (Tianshou), 'ppo' or 'qrdqn' (SB3)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model weights (auto-detected if not provided)")
    parser.add_argument("--features", type=str, default="impala", choices=["impala", "nature"],
                        help="Visual extractor for Rainbow (ignored for PPO/QRDQN, which read from .zip)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Greedy/deterministic actions (for reproducibility; default is stochastic)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rom_path = os.path.join(base_dir, 'data', 'Super Mario 64 DS (USA) (Rev 1).nds')

    savestates = [
        os.path.join(base_dir, 'data', 'Super Mario 64 DS (USA) (Rev 1).ds1'),
        os.path.join(base_dir, 'data', 'Super Mario 64 DS (USA) (Rev 1).ds3')
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir = os.path.join(base_dir, 'models')

    if args.algo == "rainbow":
        model_path = args.model
        if model_path is None:
            pth_files = [f for f in os.listdir(models_dir) if f.endswith('_best.pth')] if os.path.exists(models_dir) else []
            if pth_files:
                pth_files.sort(key=lambda f: os.path.getmtime(os.path.join(models_dir, f)), reverse=True)
                model_path = os.path.join(models_dir, pth_files[0])
                print(f"Auto-detected Rainbow model: {pth_files[0]}")
            else:
                print("No Rainbow model found in models/. Using random weights.")
                model_path = "NOT_FOUND"

        policy = load_rainbow_policy(model_path, device, features=args.features)
        play_rainbow(policy, rom_path, savestates, device, deterministic=args.deterministic)

    else:
        model_path = args.model
        if model_path is None:
            pattern = args.algo.lower()
            zip_files = [f for f in os.listdir(models_dir)
                         if f.endswith('.zip') and pattern in f.lower()] if os.path.exists(models_dir) else []
            if zip_files:
                zip_files.sort(key=lambda f: os.path.getmtime(os.path.join(models_dir, f)), reverse=True)
                model_path = os.path.join(models_dir, zip_files[0])
                print(f"Auto-detected {args.algo.upper()} model: {zip_files[0]}")
            else:
                print(f"ERROR: No {args.algo.upper()} model found in models/. "
                      f"Train one first with: python -m src.train_ppo / src.train_qrdqn")
                return

        model = load_sb3_model(args.algo, model_path)
        if model:
            play_sb3(model, rom_path, savestates,
                     label=f"{args.algo.upper()}", deterministic=args.deterministic)


if __name__ == "__main__":
    play()
