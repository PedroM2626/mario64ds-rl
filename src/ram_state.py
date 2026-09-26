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
  0x0217F760  HEIGHT -- during a death fall it clamps at -307200 (-75.0: the
              kill plane); resets to -32768 (-8.0) on respawn
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

KILL_PLANE = -300000               # height clamp observed during death falls
RESPAWN_HEIGHT = -32768            # -8.0: post-respawn height

FIXED_POINT_TOL = 5000             # boot-time verification tolerance for 'const'


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

    @staticmethod
    def units(v) -> np.ndarray:
        """Fixed-point words -> track units (20.12)."""
        return np.asarray(v, dtype=np.float64) / SCALE

    @staticmethod
    def is_dead(state) -> bool:
        """Kill-plane hit (a real fall), readable BEFORE the screen fades."""
        return bool(state[4] <= KILL_PLANE + 1)
