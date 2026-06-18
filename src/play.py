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
from src.impala_cnn import ImpalaCNN, RainbowNet
from tianshou.algorithm.modelfree.c51 import C51Policy


def load_rainbow_policy(model_path, device):
    """Carrega a política Rainbow DQN com IMPALA CNN."""
    feature_net = ImpalaCNN(c=4, h=84, w=84, features_dim=256)
    action_shape = 6
    num_atoms = 51
    model = RainbowNet(feature_net, action_shape, num_atoms, noisy_std=0.5).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Rainbow weights loaded from {model_path}")
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


def load_ppo_model(model_path):
    """Carrega o modelo PPO do Stable-Baselines3."""
    from stable_baselines3 import PPO
    if os.path.exists(model_path):
        model = PPO.load(model_path)
        print(f"PPO model loaded from {model_path}")
        return model
    else:
        print(f"WARNING: PPO model not found at {model_path}")
        return None


def play_rainbow(policy, rom_path, savestates, device):
    """Visualização estocástica com Rainbow DQN."""
    def make_eval_env():
        env = Mario64DSEnv(rom_path, savestates[0], render_mode='human')
        env = FrameStackObservation(env, stack_size=4)
        return env

    env = DummyVectorEnv([make_eval_env])

    for idx, state_path in enumerate(savestates):
        print(f"\n--- Rainbow DQN - Savestate {idx + 1} ---")
        env.workers[0].env.env.state_path = state_path
        
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            batch = Batch(obs=obs, info=info)
            with torch.no_grad():
                result = policy(batch)
                action = result.act
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated[0] or truncated[0]
            total_reward += reward[0]
            steps += 1
            time.sleep(0.01)

        print(f"Finished: reward={total_reward:.2f}, steps={steps}")
    
    env.close()


def play_ppo(model, rom_path, savestates):
    """Visualização estocástica com PPO."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    from stable_baselines3.common.monitor import Monitor
    
    def make_env():
        env = Mario64DSEnv(rom_path, savestates[0], render_mode='human')
        env = Monitor(env)
        return env
    
    vec_env = DummyVecEnv([make_env])
    vec_env = VecFrameStack(vec_env, n_stack=4)

    for idx, state_path in enumerate(savestates):
        print(f"\n--- PPO NatureCNN - Savestate {idx + 1} ---")
        vec_env.envs[0].env.state_path = state_path
        
        obs = vec_env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=False)
            obs, rewards, dones, infos = vec_env.step(action)
            total_reward += rewards[0]
            done = dones[0]
            steps += 1
            time.sleep(0.01)

        print(f"Finished: reward={total_reward:.2f}, steps={steps}")
    
    vec_env.close()


def play():
    parser = argparse.ArgumentParser(description="Visualize Mario 64 DS RL Agent")
    parser.add_argument("--algo", type=str, default="rainbow", choices=["rainbow", "ppo"],
                        help="Algorithm to use: 'rainbow' (Rainbow DQN + IMPALA) or 'ppo' (PPO + NatureCNN)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model weights (auto-detected if not provided)")
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
        # Auto-detect model path
        model_path = args.model
        if model_path is None:
            # Procura o arquivo _best.pth mais recente
            pth_files = [f for f in os.listdir(models_dir) if f.endswith('_best.pth')] if os.path.exists(models_dir) else []
            if pth_files:
                pth_files.sort(key=lambda f: os.path.getmtime(os.path.join(models_dir, f)), reverse=True)
                model_path = os.path.join(models_dir, pth_files[0])
                print(f"Auto-detected Rainbow model: {pth_files[0]}")
            else:
                print("No Rainbow model found in models/. Using random weights.")
                model_path = "NOT_FOUND"
        
        policy = load_rainbow_policy(model_path, device)
        play_rainbow(policy, rom_path, savestates, device)
    
    elif args.algo == "ppo":
        model_path = args.model
        if model_path is None:
            # Procura o arquivo .zip mais recente
            zip_files = [f for f in os.listdir(models_dir) if f.endswith('.zip') and 'ppo' in f.lower()] if os.path.exists(models_dir) else []
            if zip_files:
                zip_files.sort(key=lambda f: os.path.getmtime(os.path.join(models_dir, f)), reverse=True)
                model_path = os.path.join(models_dir, zip_files[0])
                print(f"Auto-detected PPO model: {zip_files[0]}")
            else:
                print("ERROR: No PPO model found in models/. Train one first with: python -m src.train_ppo")
                return
        
        model = load_ppo_model(model_path)
        if model:
            play_ppo(model, rom_path, savestates)


if __name__ == "__main__":
    play()
