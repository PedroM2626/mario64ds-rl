import os
import pytest
import numpy as np
from env import Mario64DSEnv

@pytest.fixture
def env():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rom_path = os.path.join(base_dir, "data", "Super Mario 64 DS (USA) (Rev 1).nds")
    state_path = os.path.join(base_dir, "data", "Super Mario 64 DS (USA) (Rev 1).ds1")
    
    environment = Mario64DSEnv(rom_path=rom_path, state_path=state_path)
    yield environment
    environment.close()

def test_env_initialization(env):
    assert env.action_space.n == 6
    assert env.observation_space.shape == (84, 84, 1)

def test_env_reset(env):
    obs, info = env.reset()
    assert obs.shape == (84, 84, 1)
    assert obs.dtype == np.uint8
    assert "steps" in info

def test_env_step(env):
    env.reset()
    # 0 = Noop, 1 = Left, 2 = Right, 3 = Up, 4 = Down, 5 = Jump
    obs, reward, done, truncated, info = env.step(1)
    
    assert obs.shape == (84, 84, 1)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert "steps" in info
