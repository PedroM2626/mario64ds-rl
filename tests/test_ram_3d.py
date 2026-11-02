"""Unit tests for 3D coordinates, RAM-based death detection, and goal/star rewards."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.ram_state import (
    Sm64DsState, TRACK_GOALS, KILL_PLANE,
    POS_X_ADDR, POS_Y_ADDR, POS_Z_ADDR,
    STAR_COUNT_ADDR, COURSE_ID_ADDR
)
from src.env import Mario64DSEnv


def test_track_goals_metadata():
    for tid in ("ds1", "ds2", "ds3", "ds8", "ds9"):
        assert tid in TRACK_GOALS
        spec = TRACK_GOALS[tid]
        assert "target_pos" in spec
        assert spec["target_pos"].shape == (3,)
        assert "goal_type" in spec
        assert spec["goal_type"] in ("position", "star")
        assert "success_dist" in spec


def test_sm64ds_state_mock_readers():
    mock_emu = MagicMock()
    # Mock memory read for 3D coordinates: (X=1000*4096, Y=2000*4096, Z=3000*4096)
    def fake_read(start, end, size, signed):
        if start == POS_X_ADDR:
            return 1000 * 4096
        elif start == POS_Y_ADDR:
            return 2000 * 4096
        elif start == POS_Z_ADDR:
            return 3000 * 4096
        elif start == STAR_COUNT_ADDR:
            return 25
        elif start == COURSE_ID_ADDR:
            return 11
        return 0

    mock_emu.memory.read.side_effect = fake_read
    st = Sm64DsState(mock_emu)

    pos = st.get_pos_3d()
    np.testing.assert_allclose(pos, [1000.0, 2000.0, 3000.0])
    assert st.get_star_count() == 25
    assert st.get_course_id() == 11


def test_is_dead_conditions():
    # Kill plane hit via state vector
    state = np.zeros(6, dtype=np.float64)
    state[4] = KILL_PLANE
    assert Sm64DsState.is_dead(state=state)
    state[4] = KILL_PLANE + 10000
    assert not Sm64DsState.is_dead(state=state)

    # Kill plane hit via 3D height threshold
    pos = np.array([0.0, -7000.0, 0.0])
    assert Sm64DsState.is_dead(pos_3d=pos, kill_y=-6500.0)
    pos_safe = np.array([0.0, -6000.0, 0.0])
    assert not Sm64DsState.is_dead(pos_3d=pos_safe, kill_y=-6500.0)


def test_rgb_array_render_mode():
    with patch("src.env.DeSmuME") as mock_emu_cls:
        mock_emu = MagicMock()
        mock_emu_cls.return_value = mock_emu
        # Raw display buffer: 384 x 256 x 4 (BGRX)
        raw_buffer = np.zeros((384, 256, 4), dtype=np.uint8)
        # Put blue=200 in channel 0, red=100 in channel 2
        raw_buffer[:192, :, 0] = 200
        raw_buffer[:192, :, 2] = 100
        mock_emu.display_buffer_as_rgbx.return_value = raw_buffer.flatten().tobytes()

        env = Mario64DSEnv.__new__(Mario64DSEnv)
        Mario64DSEnv.__init__(env, rom_path="fake.nds", state_path="fake.ds1", render_mode="rgb_array")

        rendered = env.render()
        assert rendered.shape == (192, 256, 3)
        # Should be converted to RGB: channel 0 is red (100), channel 2 is blue (200)
        assert rendered[0, 0, 0] == 100
        assert rendered[0, 0, 2] == 200


def test_potential_distance_reward_and_goal_reach():
    with patch("src.env.DeSmuME") as mock_emu_cls:
        mock_emu = MagicMock()
        mock_emu_cls.return_value = mock_emu

        coords = [
            np.array([-4444.5, 6409.8, -5976.6]),  # start
            np.array([-5000.0, 0.0, -6000.0]),     # moved closer
            np.array([-6342.0, -5700.0, -6625.0]),  # reached goal (<250 units)
        ]
        step_idx = [0]

        def fake_read(start, end, size, signed):
            idx = min(step_idx[0], len(coords) - 1)
            c = coords[idx]
            if start == POS_X_ADDR:
                return int(c[0] * 4096)
            elif start == POS_Y_ADDR:
                return int(c[1] * 4096)
            elif start == POS_Z_ADDR:
                return int(c[2] * 4096)
            elif start == COURSE_ID_ADDR:
                return 11
            return 0

        mock_emu.memory.read.side_effect = fake_read
        mock_emu.display_buffer_as_rgbx.return_value = np.full((384, 256, 4), 128, dtype=np.uint8).flatten().tobytes()

        env = Mario64DSEnv.__new__(Mario64DSEnv)
        Mario64DSEnv.__init__(env, rom_path="fake.nds", state_path="fake.ds1",
                              dist_reward_enabled=True, dist_weight=0.1,
                              goal_reward=200.0, step_penalty=0.0)

        obs, info = env.reset()
        initial_dist = info["dist_to_goal"]
        assert initial_dist > 5000.0

        # Step 1: Moved closer
        step_idx[0] = 1
        obs, reward, done, truncated, info = env.step(3)  # UP
        assert not done
        assert info["dist_reward"] > 0
        assert reward > 0

        # Step 2: Reached goal
        step_idx[0] = 2
        obs, reward, done, truncated, info = env.step(3)
        assert done
        assert info["reached_goal"]
        assert info["success"]
        assert reward >= 200.0


def test_star_collection_reward():
    with patch("src.env.DeSmuME") as mock_emu_cls:
        mock_emu = MagicMock()
        mock_emu_cls.return_value = mock_emu

        # Position right under star: (2342.5, 6885.7, 2351.1)
        # Star is at (2350.0, 7145.0, 2350.0)
        stars = [25, 26] # star counter increments
        step_idx = [0]

        def fake_read(start, end, size, signed):
            if start == POS_X_ADDR:
                return int(2342.5 * 4096)
            elif start == POS_Y_ADDR:
                return int(6885.7 * 4096)
            elif start == POS_Z_ADDR:
                return int(2351.1 * 4096)
            elif start == STAR_COUNT_ADDR:
                return stars[min(step_idx[0], 1)]
            elif start == COURSE_ID_ADDR:
                return 14
            return 0

        mock_emu.memory.read.side_effect = fake_read
        mock_emu.display_buffer_as_rgbx.return_value = np.full((384, 256, 4), 128, dtype=np.uint8).flatten().tobytes()

        env = Mario64DSEnv.__new__(Mario64DSEnv)
        Mario64DSEnv.__init__(env, rom_path="fake.nds", state_path="data/Super Mario 64 DS (USA) (Rev 1).ds9",
                              star_reward=500.0, step_penalty=0.0)

        obs, info = env.reset()
        assert info["star_count"] == 25
        assert not info["star_collected"]

        # Jump & collect star
        step_idx[0] = 1
        obs, reward, done, truncated, info = env.step(5)  # Jump
        assert done
        assert info["star_collected"]
        assert info["success"]
        assert reward >= 500.0


def test_anti_jumping_exploit_progress_mode():
    """Verify that high-water mark progress mode eliminates the repeated jump exploit."""
    with patch("src.env.DeSmuME") as mock_emu_cls:
        mock_emu = MagicMock()
        mock_emu_cls.return_value = mock_emu

        # Mario jumps up, lands down, jumps up again to same height
        # Floor: Y=-4710, Peak: Y=-4468. Star is at Y=7145
        y_sequence = [-4710.0, -4468.5, -4710.0, -4468.5, -4710.0]
        step_idx = [0]

        def fake_read(start, end, size, signed):
            if start == POS_X_ADDR:
                return int(1470.0 * 4096)
            elif start == POS_Y_ADDR:
                return int(y_sequence[step_idx[0]] * 4096)
            elif start == POS_Z_ADDR:
                return int(128.0 * 4096)
            elif start == COURSE_ID_ADDR:
                return 14
            return 0

        mock_emu.memory.read.side_effect = fake_read
        mock_emu.display_buffer_as_rgbx.return_value = np.full((384, 256, 4), 128, dtype=np.uint8).flatten().tobytes()

        env = Mario64DSEnv.__new__(Mario64DSEnv)
        Mario64DSEnv.__init__(env, rom_path="fake.nds", state_path="data/Super Mario 64 DS (USA) (Rev 1).ds8",
                              dist_reward_enabled=True, dist_weight=0.05, step_penalty=0.02,
                              dist_reward_mode="progress")

        obs, info = env.reset()

        # Step 1: Jump 1 (first time reaching Y=-4468.5) -> High-water mark broken -> Positive reward
        step_idx[0] = 1
        obs, r1, done, truncated, info = env.step(5)
        assert info["dist_reward"] > 0
        assert r1 > 0

        # Step 2: Fall back to floor (Y=-4710.0) -> No reward (not a record)
        step_idx[0] = 2
        obs, r2, done, truncated, info = env.step(0)
        assert info["dist_reward"] == 0.0
        assert r2 == -0.02  # Only step penalty

        # Step 3: Jump 2 (reaches Y=-4468.5 again) -> NOT a new high-water mark! -> Zero dist reward
        step_idx[0] = 3
        obs, r3, done, truncated, info = env.step(5)
        assert info["dist_reward"] == 0.0
        assert r3 == -0.02  # Exploit eliminated: only step penalty incurred!

        # Step 4: Fall back to floor again -> Zero dist reward
        step_idx[0] = 4
        obs, r4, done, truncated, info = env.step(0)
        assert info["dist_reward"] == 0.0
        assert r4 == -0.02
