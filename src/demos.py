"""Human demonstrations recorded by scripts/record_human.py.

A recording stores the env action index applied at every decision, so it is
directly executable by Mario64DSEnv with the same action_set and frameskip.
Rewards and observations are NOT taken from the file: replay_transitions()
re-runs the actions from the recording's source savestate under the caller's
reward configuration and frame stack, so prefilled transitions are exactly the
ones the trainer itself would have produced for those inputs.

This is replay-buffer prefill ("DQfD-lite"): no large-margin supervised loss and
no permanent demo partition, so demo transitions age out of the circular buffer
like any other transition.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 16):
            h.update(chunk)
    return h.hexdigest()


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_demo(attempt_dir: str, verify_hashes: bool = True) -> dict:
    """manifest + arrays of one completed/partial attempt folder."""
    with open(os.path.join(attempt_dir, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not os.path.isabs(manifest["source_state"]):  # recorded relative to the repo root
        manifest["source_state"] = os.path.join(ROOT, manifest["source_state"])
    if verify_hashes:
        for name, digest in manifest["files_sha256"].items():
            if sha256_file(os.path.join(attempt_dir, name)) != digest:
                raise ValueError(f"{attempt_dir}/{name} changed since it was recorded")
        if sha256_file(manifest["source_state"]) != manifest["source_state_sha256"]:
            raise ValueError(f"source savestate {manifest['source_state']} differs from the recording")
    with np.load(os.path.join(attempt_dir, "demo.npz")) as data:
        arrays = {k: data[k] for k in data.files}
    return {"manifest": manifest, **arrays}


def replay_transitions(rom: str, demo: dict, n_stack: int = 4, **env_kwargs) -> list[dict]:
    """Re-execute a demo and return SB3-shaped single-env transitions.

    The episode horizon is lifted to the demo length so a long human run is not
    cut by the training max_steps (truncation would auto-reset mid-demo).
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
    from src.envs import make_env

    m = demo["manifest"]
    actions = demo["actions"].astype(np.int64)
    env_kwargs.update(action_set=m["action_set"], frameskip=m["frameskip"],
                      max_steps=len(actions) + 1, track_id=m.get("track_id"))
    env_kwargs.pop("start_states", None)  # always the recording's own start
    # A child process: two DeSmuME instances in one process take an access
    # violation (scripts/record_dashboard.py), and the trainer may hold others.
    vec = SubprocVecEnv([make_env(rom, m["source_state"], monitor=False, **env_kwargs)])
    vec = VecTransposeImage(VecFrameStack(vec, n_stack=n_stack))
    out = []
    try:
        obs = vec.reset()
        for t, a in enumerate(actions):
            next_obs, reward, done, infos = vec.step(np.array([a]))
            info = infos[0]
            real_next = info["terminal_observation"][None] if done[0] else next_obs
            out.append({"obs": obs.copy(), "next_obs": real_next.copy(), "action": np.array([[a]]),
                        "reward": reward.copy(), "done": done.copy(), "info": info})
            if done[0]:
                if t != len(actions) - 1:
                    raise RuntimeError(f"demo ended early at decision {t} of {len(actions)} "
                                       f"(died={info.get('died')}); replay is not reproducing it")
                break
            obs = next_obs
    finally:
        vec.close()
    return out


def prefill_replay_buffer(model, transitions: list[dict]) -> int:
    """Add transitions to an SB3 off-policy replay buffer; returns count added.

    The buffer is shaped (buffer_size, n_envs, ...), so transitions are grouped
    n_envs at a time; up to n_envs-1 trailing transitions are dropped.
    """
    buf = model.replay_buffer
    n = buf.n_envs
    added = 0
    for i in range(0, len(transitions) - n + 1, n):
        group = transitions[i:i + n]
        buf.add(np.concatenate([g["obs"] for g in group]),
                np.concatenate([g["next_obs"] for g in group]),
                np.concatenate([g["action"] for g in group]),
                np.concatenate([g["reward"] for g in group]),
                np.concatenate([g["done"] for g in group]),
                [g["info"] for g in group])
        added += n
    return added
