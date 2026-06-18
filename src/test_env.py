import torch
from tianshou.env import SubprocVectorEnv
from src.env import Mario64DSEnv
from src.impala_cnn import ImpalaCNN, RainbowNet
from gymnasium.wrappers import FrameStackObservation
import numpy as np

def make_env():
    def _init():
        env = Mario64DSEnv(rom_path="data/Super Mario 64 DS (USA) (Rev 1).nds", state_path="data/Super Mario 64 DS (USA) (Rev 1).ds1")
        env = FrameStackObservation(env, stack_size=4)
        return env
    return _init

if __name__ == "__main__":
    envs = SubprocVectorEnv([make_env()])
    obs, info = envs.reset()
    print("Reset obs shape:", obs.shape, "Has NaN:", np.isnan(obs).any())
    
    # Init model
    feature_net = ImpalaCNN(c=4, h=84, w=84, features_dim=256)
    model = RainbowNet(feature_net, 6, 51)
    
    for _ in range(10):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        q, _ = model(obs_tensor)
        print("Q Has NaN:", torch.isnan(q).any().item())
        
        # Take random action
        act = np.random.randint(0, 6, size=(1,))
        obs, rew, done, trunc, info = envs.step(act)
        print(f"Rew: {rew}, Done: {done}, Trunc: {trunc}, Obs NaN: {np.isnan(obs).any()}")
