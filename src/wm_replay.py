"""Replay buffer of real emulator transitions for training the world model.

Episodes are stored as parallel arrays of single grayscale frames plus the
action / reward / continue signals emitted by ``src/env.py``. Batches are
returned as stacked-frame sequences ``(B, T, 4, 84, 84)`` (channel-first,
oldest->newest, zero-padded at episode start) so the layout matches the
``VecFrameStack`` convention used elsewhere in the project.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional

import numpy as np

STACK = 4


def _pad_frames(frames: np.ndarray) -> np.ndarray:
    """(T,H,W) uint8 -> (T+3,H,W) with 3 leading zero frames."""
    z = np.zeros((STACK - 1,) + frames.shape[1:], dtype=frames.dtype)
    return np.concatenate([z, frames], axis=0)


class EpisodeBuffer:
    def __init__(self):
        self.episodes: List[Dict[str, np.ndarray]] = []
        self.total_steps = 0

    def add(self, frames: np.ndarray, actions: np.ndarray,
            rewards: np.ndarray, continues: np.ndarray,
            source: Optional[str] = None, completed: Optional[bool] = None,
            labels: Optional[np.ndarray] = None):
        assert frames.ndim == 3, "frames must be (T,H,W)"
        if completed is None:
            # an episode that ended without a terminal death (continues[-1]==1)
            # reached the horizon and is treated as a successful descent
            completed = bool(len(continues) == 0 or continues[-1] > 0.5)
        ep = dict(
            frames=frames.astype(np.uint8),
            actions=actions.astype(np.int64),
            rewards=rewards.astype(np.float32),
            continues=continues.astype(np.float32),
            source=source,
            completed=bool(completed),
        )
        if labels is not None:
            ep["labels"] = labels.astype(np.int64)
        self.episodes.append(ep)
        self.total_steps += len(frames)

    def __len__(self):
        return len(self.episodes)

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(dict(episodes=self.episodes, total_steps=self.total_steps), f)

    def load(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.episodes = d["episodes"]
        self.total_steps = d["total_steps"]

    def expert_episodes(self, require_completed: bool = True):
        """Episodes labeled as successful expert (e.g. PPO) demonstrations."""
        out = []
        for e in self.episodes:
            if e.get("source") != "ppo":
                continue
            if require_completed and not e.get("completed", False):
                continue
            out.append(e)
        return out

    def sample_expert(self, batch: int, seq_len: int, episodes=None,
                      rng: Optional[np.random.Generator] = None, from_start: bool = True):
        """Supervised windows for behaviour-cloning / policy distillation.

        Returns (obs_seq (B,T,4,H,W) in [0,1], action_in (B,T) that produced each
        observation, label (B,T) = the expert action to take AT that observation).

        ``from_start=True`` always begins the window at the episode's first frame
        so the RSSM belief is cold-started from the zero initial state exactly as
        the deployment controller does -- this removes the train/deploy belief
        mismatch that otherwise makes a high-accuracy student still fail on-policy.
        """
        import torch
        rng = rng or np.random.default_rng()
        eps = episodes if episodes is not None else self.expert_episodes()
        usable = [e for e in eps if len(e["frames"]) >= seq_len + 2] or eps
        obs = np.zeros((batch, seq_len, STACK, 84, 84), dtype=np.float32)
        act_in = np.zeros((batch, seq_len), dtype=np.int64)
        label = np.zeros((batch, seq_len), dtype=np.int64)
        for b in range(batch):
            e = usable[rng.integers(len(usable))]
            T = len(e["frames"])
            padded = _pad_frames(e["frames"])
            if from_start:
                start = 0
            else:
                start = int(rng.integers(0, max(1, T - seq_len - 1)))
            for t in range(seq_len):
                i = start + t
                win = padded[i:i + STACK]
                if len(win) < STACK:
                    win = np.concatenate([win, np.zeros((STACK - len(win),) + win.shape[1:], win.dtype)])
                obs[b, t] = win.transpose(0, 1, 2).astype(np.float32) / 255.0
                act_in[b, t] = e["actions"][min(i, T - 1)]
                if "labels" in e:  # DAgger: expert label for the visited state
                    label[b, t] = e["labels"][min(i, T - 1)]
                else:               # offline demo: expert's next move
                    label[b, t] = e["actions"][min(i + 1, T - 1)]
        return (torch.from_numpy(obs), torch.from_numpy(act_in), torch.from_numpy(label))

    def sample(self, batch: int, seq_len: int, rng: Optional[np.random.Generator] = None):
        import torch
        rng = rng or np.random.default_rng()
        # Only episodes long enough to fill a sequence (else resample a long one).
        usable = [e for e in self.episodes if len(e["frames"]) >= seq_len + 1]
        if not usable:
            usable = self.episodes
        obs = np.zeros((batch, seq_len, STACK, 84, 84), dtype=np.float32)
        acts = np.zeros((batch, seq_len), dtype=np.int64)
        rews = np.zeros((batch, seq_len), dtype=np.float32)
        conts = np.zeros((batch, seq_len), dtype=np.float32)
        for b in range(batch):
            e = usable[rng.integers(len(usable))]
            T = len(e["frames"])
            padded = _pad_frames(e["frames"])
            start = int(rng.integers(0, max(1, T - seq_len)))
            for t in range(seq_len):
                i = start + t
                win = padded[i:i + STACK]
                if len(win) < STACK:  # safety pad
                    win = np.concatenate([win, np.zeros((STACK - len(win),) + win.shape[1:], win.dtype)])
                obs[b, t] = win.transpose(0, 1, 2).astype(np.float32) / 255.0
                idx = min(i, T - 1)
                acts[b, t] = e["actions"][idx]
                rews[b, t] = e["rewards"][idx]
                conts[b, t] = e["continues"][idx]
        return (torch.from_numpy(obs), torch.from_numpy(acts),
                torch.from_numpy(rews), torch.from_numpy(conts))
