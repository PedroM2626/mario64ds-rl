import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from src.env import Mario64DSEnv

def play():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rom_path = os.path.join(base_dir, 'data', 'Super Mario 64 DS (USA) (Rev 1).nds')
    model_path = os.path.join(base_dir, 'models', 'ppo_mario64ds')

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    savestates = [
        os.path.join(base_dir, 'data', 'Super Mario 64 DS (USA) (Rev 1).ds1'),
        os.path.join(base_dir, 'data', 'Super Mario 64 DS (USA) (Rev 1).ds2'),
        os.path.join(base_dir, 'data', 'Super Mario 64 DS (USA) (Rev 1).ds3')
    ]

    print(f"Initializing environment with {savestates[0]}...")
    # Wrap in DummyVecEnv and VecFrameStack to match training architecture
    def make_env():
        # Passing render_mode here inside the actual environment
        env = Mario64DSEnv(rom_path, savestates[0], render_mode='human')
        env = Monitor(env)
        return env
        
    vec_env = DummyVecEnv([make_env])
    vec_env = VecFrameStack(vec_env, n_stack=4)

    for idx, state_path in enumerate(savestates):
        print(f"\n--- Playing Phase {idx + 1} ---")
        
        # We need to hack the underlying environment path so it loads the new savestate on reset.
        # DummyVecEnv.envs is the list of underlying unvectorized environments.
        # envs[0] is Monitor. envs[0].env is Mario64DSEnv
        vec_env.envs[0].env.state_path = state_path 
        
        obs = vec_env.reset()
        done = False
        total_reward = 0

        # We keep track of the underlying "done" condition from the environment.
        # DummyVecEnv automatically resets, so we track when it naturally ends.
        while not done:
            # Stochastic visualization as requested by the user
            action, _states = model.predict(obs, deterministic=False) 
            obs, rewards, dones, infos = vec_env.step(action)
            total_reward += rewards[0]

            # done is a boolean array for VecEnvs
            done = dones[0]

            # Limit FPS slightly so humans can watch it properly
            time.sleep(0.01)

        print(f"Phase {idx + 1} finished with total reward: {total_reward:.2f}")

    vec_env.close()

if __name__ == "__main__":
    play()
