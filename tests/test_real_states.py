"""Integration test with the real DeSmuME emulator and real savestates.

DeSmuME's C++ core allows only one emulator instance per process lifetime.
Tests re-use a single Mario64DSEnv instance across savestates.
"""

import os
import pytest
from src.env import Mario64DSEnv

ROM = "data/Super Mario 64 DS (USA) (Rev 1).nds"


@pytest.mark.skipif(not os.path.exists(ROM), reason="Real ROM not available")
def test_real_savestates_flow():
    # Instantiate single environment
    env = Mario64DSEnv(rom_path=ROM, state_path="data/Super Mario 64 DS (USA) (Rev 1).ds4")

    try:
        # 1. Test ds4 (finish of ds1)
        obs, info = env.reset()
        assert info["dist_to_goal"] <= 250.0
        obs, reward, done, trunc, info = env.step(0)
        assert done
        assert info["reached_goal"]
        assert reward >= 199.0

        # 2. Test ds5 (finish of ds2)
        env.state_path = "data/Super Mario 64 DS (USA) (Rev 1).ds5"
        env.goal_spec = None
        obs, info = env.reset()
        assert info["dist_to_goal"] <= 250.0
        obs, reward, done, trunc, info = env.step(0)
        assert done
        assert info["reached_goal"]
        assert reward >= 199.0

        # 3. Test ds6 (finish of ds3)
        env.state_path = "data/Super Mario 64 DS (USA) (Rev 1).ds6"
        env.goal_spec = None
        obs, info = env.reset()
        assert info["dist_to_goal"] <= 250.0
        obs, reward, done, trunc, info = env.step(0)
        assert done
        assert info["reached_goal"]
        assert reward >= 199.0

        # 4. Test ds8 (TTC start)
        env.state_path = "data/Super Mario 64 DS (USA) (Rev 1).ds8"
        env.goal_spec = None
        obs, info = env.reset()
        assert info["course_id"] == 14
        assert info["dist_to_goal"] > 1000.0

        # 5. Test ds9 (TTC star collection via backflip)
        env.state_path = "data/Super Mario 64 DS (USA) (Rev 1).ds9"
        env.frameskip = 1
        env.goal_spec = None
        obs, info = env.reset()
        assert info["course_id"] == 14
        assert info["star_count"] == 25

        done = False
        for f in range(60):
            if f < 8:
                env.emu.input.keypad_add_key(1024)  # KEY_R
                env.emu.cycle()
                env.emu.input.keypad_rm_key(1024)
            elif f < 35:
                env.emu.input.keypad_add_key(1024 | 2)  # KEY_R | KEY_B
                env.emu.cycle()
                env.emu.input.keypad_rm_key(1024 | 2)
            else:
                env.emu.cycle()
            obs, reward, done, trunc, info = env.step(0)
            if done:
                break
        assert info["star_collected"]
        assert reward >= 500.0

    finally:
        env.close()
