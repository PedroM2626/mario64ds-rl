import torch
import numpy as np
from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, PrioritizedVectorReplayBuffer
from tianshou.algorithm.modelfree.rainbow import RainbowDQN
from tianshou.algorithm.modelfree.c51 import C51Policy
from tianshou.algorithm.optim import TorchOptimizerFactory
from src.env import Mario64DSEnv
from src.impala_cnn import ImpalaCNN, RainbowNet
from gymnasium.wrappers import FrameStackObservation

def make_env():
    def _init():
        env = Mario64DSEnv(rom_path="data/Super Mario 64 DS (USA) (Rev 1).nds", state_path="data/Super Mario 64 DS (USA) (Rev 1).ds1")
        env = FrameStackObservation(env, stack_size=4)
        return env
    return _init

if __name__ == "__main__":
    envs = SubprocVectorEnv([make_env(), make_env()])
    feature_net = ImpalaCNN(c=4, h=84, w=84, features_dim=256)
    model = RainbowNet(feature_net, 6, 51)
    
    policy = C51Policy(
        model=model,
        action_space=envs.action_space[0],
        num_atoms=51,
        v_min=-100.0,
        v_max=100.0
    )
    
    optim_factory = TorchOptimizerFactory(torch.optim.Adam, lr=1e-4)
    algorithm = RainbowDQN(policy=policy, optim=optim_factory, gamma=0.99, n_step_return_horizon=3, target_update_freq=500)
    
    buffer = PrioritizedVectorReplayBuffer(total_size=100, buffer_num=2, alpha=0.6, beta=0.4)
    collector = Collector(algorithm, envs, buffer, exploration_noise=True)
    
    print("Collecting 10 steps...")
    collector.reset()
    collector.collect(n_step=10)
    
    for key in ["obs", "act", "rew", "done", "obs_next", "info", "policy"]:
        try:
            val = getattr(buffer, key)
            if hasattr(val, "has_nans"):
                has_nan = val.has_nans()
                print(f"Buffer {key}: NaN={has_nan}")
            elif isinstance(val, np.ndarray):
                has_nan = np.isnan(val).any()
                print(f"Buffer {key}: NaN={has_nan}")
            if hasattr(val, "hasnull"):
                has_nan = val.hasnull()
                print(f"Buffer {key}: NaN={has_nan}")
                if has_nan:
                    for k2, v2 in val.items():
                        if isinstance(v2, np.ndarray) and (np.isnan(v2).any() or np.isinf(v2).any()):
                            print(f"    -> Inner {k2} has NaN/Inf!")
                        elif hasattr(v2, "hasnull") and v2.hasnull():
                            print(f"    -> Inner {k2} has NaN/Inf!")
                        elif isinstance(v2, torch.Tensor) and (torch.isnan(v2).any().item() or torch.isinf(v2).any().item()):
                            print(f"    -> Inner {k2} has NaN/Inf!")
            else:
                print(f"Buffer {key}: Type {type(val)}")
        except Exception as e:
            print(f"Buffer {key}: Error checking - {e}")

    try:
        print(f"Buffer hasnull(): {buffer.hasnull()}")
    except Exception as e:
        print(f"Error checking buffer.hasnull(): {e}")
