"""Unit tests for environment logic that do NOT require emulator or ROM binaries.

Covers reward conventions (anti-suicide constraint), spaces validation,
and environment instantiation using a mocked DeSmuME emulator.
"""

from unittest.mock import MagicMock, patch

import numpy as np


def _make_env(**kwargs):
    with patch("src.env.DeSmuME") as mock_emu_cls:
        mock_emu = MagicMock()
        mock_emu_cls.return_value = mock_emu
        from src.env import Mario64DSEnv
        env = Mario64DSEnv.__new__(Mario64DSEnv)
        # Call __init__ with mocked emulator
        Mario64DSEnv.__init__(env, rom_path="fake.nds", state_path="fake.ds1", **kwargs)
        return env


def test_reward_convention_death_worse_than_timeout():
    env = _make_env()
    assert env.death_penalty == 100.0
    assert env.timeout_penalty == 0.0
    assert env.death_penalty > env.timeout_penalty


def test_default_spaces_and_limits():
    env = _make_env()
    assert env.action_space.n == 6
    assert env.observation_space.shape == (84, 84, 1)
    assert env.max_steps == 1350  # 90s at frameskip 4 (duration needed to complete descent)
    assert env.frameskip == 4


def test_custom_max_steps_frameskip():
    env = _make_env(max_steps=100, frameskip=2)
    assert env.max_steps == 100
    assert env.frameskip == 2
