"""RAM state extraction for Super Mario 64 DS slide courses (USA Rev 1).

Discovered by differential search (see src/ram_search*.py, src/ram_trace.py,
src/ram_verify.py): Mario's physics struct lives at 0x0217F74C-0x0217F764 as
20.12 fixed-point s32 words (address stable across boots for this ROM):

  0x0217F74C  heading-like angle / small state (drifts smoothly; rad ~1e6 units)
  0x0217F750  speed magnitude -- saturates at 409600 (100.0 units/s: the slide
              terminal-velocity cap; ramps with the descent)
  0x0217F754  constant (-16384) during slides; part of the struct
  0x0217F75C  LATERAL position (delta sign follows LEFT/RIGHT; frozen when
              pinned against a wall; +-~100 track units across the track)
  0x0217F760  vertical-speed-like word -- clamps at -307200 (-75.0) after
              ~7 decisions of free fall; -32768 (-8.0) when grounded. It clamps
              on ANY long drop: the legitimate drop into the end room of ds1,
              drops on the slides themselves and real falls into the void
              alike, so it is not a death signal (see fell_out_of_course)
  0x0217F764  vertical velocity (spikes on JUMP; decays with gravity)

This is the exact-state observation the reference (smw-pinn) uses and the
pixel-only pipeline fundamentally could not recover: with it, actions move the
STATE ITSELF, so a learned dynamics model gives the planner precise cliff
foresight (falls are predicted from the height/lateral trajectory, not
inferred from featureless dark pixels).

Poll cost: 6 word-reads per step (no full-RAM dumps).
"""
from __future__ import annotations

import numpy as np

# struct words, in a stable order
WORD_ADDRS = (0x0217F74C, 0x0217F750, 0x0217F754, 0x0217F75C, 0x0217F760, 0x0217F764)
NAMES = ("heading", "speed", "const", "lateral", "height", "vvel")
SCALE = 4096.0                      # 20.12 fixed point -> track units

# True 3D World Position (s32, 20.12 fixed-point)
POS_X_ADDR = 0x0217F714
POS_Y_ADDR = 0x0217F718
POS_Z_ADDR = 0x0217F71C
POS_ADDRS = (POS_X_ADDR, POS_Y_ADDR, POS_Z_ADDR)

# Save-data & Level Context Registers
STAR_COUNT_ADDR = 0x0209FB64         # Global Star Counter in save RAM
# Level/area register. Not the in-game course number: it reads 11 on the Cool,
# Cool Mountain slide (ds1), 5 on the Princess's Secret Slide (ds2), 3 on the
# Tall, Tall Mountain slide (ds3) and 14 in Tick Tock Clock (ds8/ds9). Use the
# values only as fingerprints of these savestates.
COURSE_ID_ADDR = 0x02285238

KILL_PLANE = -300000               # fast-fall clamp of 0x0217F760 (not a death signal)
RESPAWN_HEIGHT = -32768            # -8.0: post-respawn height

FIXED_POINT_TOL = 5000             # boot-time verification tolerance for 'const'

# Track finish specs and 3D goal coordinates
TRACK_GOALS = {
    # Cool, Cool Mountain slide (start: ds1 -> finish: ds4)
    "ds1": {
        "course_id": 11,
        "name": "Cool, Cool Mountain slide",
        "goal_type": "position",
        "target_pos": np.array([-6342.00, -5700.00, -6625.27], dtype=np.float64),
        "success_dist": 250.0,
        # End room, measured by walking Mario around from ds4: floor 0..+28.8
        # above target_pos, up to 1,046.6 units away horizontally.
        "finish_region": {"dy": (-50.0, 100.0), "radius": 1100.0},
        "kill_y": -6500.0,
    },
    # The Princess's Secret Slide (start: ds2 -> finish: ds5)
    "ds2": {
        "course_id": 5,
        "name": "The Princess's Secret Slide",
        "goal_type": "position",
        "target_pos": np.array([-6240.47, -4479.99, 4506.09], dtype=np.float64),
        "success_dist": 250.0,
        # End area from ds5: floor at +0 above target_pos (95% of samples; max
        # +100), up to 2,167.7 away. The band stops at +30 because the last
        # stretch of carpet before the checkered line is at about +147.
        "finish_region": {"dy": (-50.0, 30.0), "radius": 2200.0},
        "kill_y": -6000.0,
    },
    # Tall, Tall Mountain slide (start: ds3 -> finish: ds6)
    "ds3": {
        "course_id": 3,
        "name": "Tall, Tall Mountain slide",
        "goal_type": "position",
        "target_pos": np.array([-514.75, -13645.99, -11697.84], dtype=np.float64),
        "success_dist": 250.0,
        # End area from ds6: -546..+242 around target_pos (not flat), up to
        # 1,537.4 away horizontally.
        "finish_region": {"dy": (-600.0, 300.0), "radius": 1550.0},
        "kill_y": -14500.0,
    },
    # Level 8: Tick Tock Clock (start: ds8 -> star: ds9)
    "ds8": {
        "course_id": 14,
        "name": "Tick Tock Clock",
        "goal_type": "star",
        "target_pos": np.array([2350.00, 7145.00, 2350.00], dtype=np.float64),
        "success_dist": 80.0,
        "kill_y": -5500.0,
        "clock_setting": "Slow",  # Period 180 frames (3.0s at 60Hz)
    },
    # Savestate 9: TTC Star collection jumping point
    "ds9": {
        "course_id": 14,
        "name": "Tick Tock Clock (Star Jump)",
        "goal_type": "star",
        "target_pos": np.array([2350.00, 7145.00, 2350.00], dtype=np.float64),
        "success_dist": 80.0,
        "kill_y": -5500.0,
        "clock_setting": "Slow",
    },
}

# Reverse lookup by course_id for fallback track resolution
COURSE_TO_DEFAULT_TRACK = {
    11: "ds1",
    5: "ds2",
    3: "ds3",
    14: "ds8",
}


class Sm64DsState:
    """Per-step RAM state reader + semantic helpers."""

    def __init__(self, emu):
        self.emu = emu

    def _rs32(self, addr):
        v = self.emu.memory.read(addr, addr, 4, False)
        return v - (1 << 32) if v >= (1 << 31) else v

    def verify(self) -> bool:
        """Boot-time sanity check that the struct is where we expect it."""
        try:
            c = self._rs32(WORD_ADDRS[2])
            return abs(c - (-16384)) <= FIXED_POINT_TOL
        except Exception:
            return False

    def poll(self) -> np.ndarray:
        """Returns the raw 6-word state vector (s32, fixed-point units)."""
        return np.array([self._rs32(a) for a in WORD_ADDRS], dtype=np.float64)

    def get_pos_3d(self) -> np.ndarray:
        """Returns Mario's 3D position [X, Y, Z] in track units (20.12 fixed-point / 4096)."""
        return np.array([self._rs32(a) for a in POS_ADDRS], dtype=np.float64) / SCALE

    def get_star_count(self) -> int:
        """Reads global star counter from save RAM."""
        try:
            return int(self.emu.memory.read(STAR_COUNT_ADDR, STAR_COUNT_ADDR, 4, False))
        except Exception:
            return 0

    def get_course_id(self) -> int:
        """Reads current level/course ID from active level register."""
        try:
            return int(self.emu.memory.read(COURSE_ID_ADDR, COURSE_ID_ADDR, 4, False))
        except Exception:
            return 0

    @staticmethod
    def units(v) -> np.ndarray:
        """Fixed-point words -> track units (20.12)."""
        return np.asarray(v, dtype=np.float64) / SCALE

    @staticmethod
    def fell_out_of_course(pos_3d, kill_y) -> bool:
        """Below the course's lowest floor: a real fall into the void.

        Measured from ds8 (hold Up off the start platform): Y crosses
        kill_y=-5500 at frame 168, the same frame the 0x0217F760 word clamps,
        and the screen goes black only at frame 290 (Y=-10057).
        """
        return pos_3d is not None and kill_y is not None and pos_3d[1] < kill_y

    @staticmethod
    def in_finish_region(pos_3d, target_pos, region) -> bool:
        """Inside a measured end area (height band + horizontal radius)."""
        if pos_3d is None or region is None:
            return False
        dy = pos_3d[1] - target_pos[1]
        h = float(np.hypot(pos_3d[0] - target_pos[0], pos_3d[2] - target_pos[2]))
        return region["dy"][0] <= dy <= region["dy"][1] and h <= region["radius"]

    @staticmethod
    def is_dead(state=None, kill_plane=KILL_PLANE, pos_3d=None, kill_y=None) -> bool:
        """Legacy rule: fast-fall clamp OR below kill_y.

        The clamp fires on legitimate long drops too (see 0x0217F760 above);
        Mario64DSEnv uses it only with death_rule="struct"."""
        if state is not None and len(state) >= 5:
            if state[4] <= kill_plane + 1:
                return True
        if pos_3d is not None and kill_y is not None:
            if pos_3d[1] < kill_y:
                return True
        return False
