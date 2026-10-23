"""Shared environment construction and save-state paths.

The SB3 trainers, the evaluator and the recording scripts all need the same
three things: resolve a track id to a savestate, build the Mario64DSEnv inside
a subprocess-safe vector env, and stack/transpose the frames exactly the way
the CNN expects. Keeping those in one module avoids the drift that produced the
historical video/eval mismatch documented in the README.
"""

from __future__ import annotations

import os

from src.env import Mario64DSEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKS = ("ds1", "ds2", "ds3")
ROM_STEM = "Super Mario 64 DS (USA) (Rev 1)"
ROM_FILENAME = f"{ROM_STEM}.nds"


def rom_path(path: str | None = None, base_dir: str | None = None) -> str:
    """Absolute path to the ROM (default: data/<USA Rev 1>.nds)."""
    base = base_dir or ROOT
    if not path:
        return os.path.join(base, "data", ROM_FILENAME)
    return path if os.path.isabs(path) else os.path.join(base, path)


def savestate_path(track: str, base_dir: str | None = None) -> str:
    """Resolve ds1 (or a full path) to the matching savestate file."""
    base = base_dir or ROOT
    raw = str(track).strip()
    if os.path.isabs(raw):
        return raw
    if os.sep in raw or (os.altsep and os.altsep in raw):
        # A relative path is relative to the repository root, not the cwd.
        return os.path.join(base, raw)
    name = os.path.basename(raw)
    if name.endswith((".ds1", ".ds2", ".ds3")):
        return os.path.join(base, "data", name)
    if not name.startswith("ds"):
        name = f"ds{name}"
    return os.path.join(base, "data", f"{ROM_STEM}.{name}")


def savestate_paths(states, base_dir: str | None = None) -> list[str]:
    """Resolve a comma-separated string or an iterable of track ids/paths."""
    if isinstance(states, str):
        states = [s.strip() for s in states.split(",") if s.strip()]
    return [savestate_path(s, base_dir) for s in states]


def make_env(rom_path_value: str, state_path: str, rank: int = 0, seed: int = 0,
             max_steps: int = 1350, frameskip: int = 4, step_penalty: float = 0.02,
             flow_weight: float = 1.0, monitor: bool = True, verbose: bool = False):
    """Return an env factory for SubprocVecEnv/DummyVecEnv.

    rank offsets the per-worker seed. The reset happens inside the child
    process, exactly as in the previous per-script implementations.
    """
    def _init():
        env = Mario64DSEnv(
            rom_path=rom_path_value, state_path=state_path,
            max_steps=max_steps, frameskip=frameskip,
            step_penalty=step_penalty, flow_weight=flow_weight, verbose=verbose,
        )
        if monitor:
            from stable_baselines3.common.monitor import Monitor
            env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def build_vec(rom_path_value: str, states, rank0: int = 0, seed: int = 0,
              max_steps: int = 1350, frameskip: int = 4, step_penalty: float = 0.02,
              flow_weight: float = 1.0, n_stack: int = 4, monitor: bool = True):
    """SubprocVecEnv + VecFrameStack + VecTransposeImage, one env per state."""
    from stable_baselines3.common.vec_env import (
        SubprocVecEnv, VecFrameStack, VecTransposeImage,
    )
    paths = states if not isinstance(states, str) else savestate_paths(states)
    vec = SubprocVecEnv([
        make_env(rom_path_value, sp, rank=rank0 + i, seed=seed,
                 max_steps=max_steps, frameskip=frameskip,
                 step_penalty=step_penalty, flow_weight=flow_weight,
                 monitor=monitor)
        for i, sp in enumerate(paths)
    ])
    vec = VecFrameStack(vec, n_stack=n_stack)
    vec = VecTransposeImage(vec)
    return vec
