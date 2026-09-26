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
            labels: Optional[np.ndarray] = None, probe_at: Optional[int] = None,
            probe_id: Optional[int] = None, branch_cont: Optional[int] = None):
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
        if probe_at is not None:
            ep["probe_at"] = int(probe_at)
        if probe_id is not None:
            ep["probe_id"] = int(probe_id)
        if branch_cont is not None:
            ep["branch_cont"] = int(branch_cont)
        self.episodes.append(ep)
        self.total_steps += len(frames)

    def __len__(self):
        return len(self.episodes)

    def ensure_returns(self, gamma: float = 0.995):
        """Compute and cache the discounted return G_t for every step of every episode.

        G_t = r_t + gamma * G_{t+1}, with G at the end of a *died* episode being
        the -100 death spike (it is inside the rewards) and at the end of a
        *survived* (completed) episode just the final dense reward. These targets
        are what the world-model value head is fit on: they are the only progress
        signal that provably cannot be earned by falling.
        """
        for e in self.episodes:
            if "returns" in e and e.get("_ret_gamma") == gamma:
                continue
            r = e["rewards"].astype(np.float64)
            T = len(r)
            g = np.zeros(T, dtype=np.float32)
            acc = 0.0
            for t in reversed(range(T)):
                acc = float(r[t]) + gamma * acc
                g[t] = acc
            e["returns"] = g
            e["_ret_gamma"] = gamma

    def returns_stats(self, gamma: float = 0.995):
        """(mean, std) of all per-step returns, for value-target normalization."""
        self.ensure_returns(gamma)
        allg = np.concatenate([e["returns"] for e in self.episodes]) if self.episodes \
            else np.zeros(1)
        return float(allg.mean()), float(allg.std() + 1e-6)

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(dict(episodes=self.episodes, total_steps=self.total_steps), f)

    def load(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.episodes = d["episodes"]
        self.total_steps = d["total_steps"]

    def probe_episodes(self):
        """Counterfactual branch episodes (source='probe*'): same prefix/belief,
        different probe actions, real outcomes."""
        return [e for e in self.episodes
                if str(e.get("source", "")).startswith("probe") and "probe_at" in e]

    def _branch_groups(self):
        """Probe states -> their branch episodes (grouped by continuation
        semantics), with a cached contrast score."""
        import hashlib
        from collections import defaultdict
        eps = self.probe_episodes()
        groups = defaultdict(list)
        for e in eps:
            pa = e["probe_at"]
            cont = int(e.get("branch_cont", 0))
            key = hashlib.md5(
                np.ascontiguousarray(e["frames"][max(0, pa - 1)]).tobytes()
                + bytes([pa, cont]) + str(e.get("probe_id", 0)).encode()
            ).hexdigest()
            groups[key].append(e)
        out = []
        for g in groups.values():
            if len(g) < 2:
                continue
            if "branch_contrast" not in g[0]:
                returns = [float(np.sum(e["rewards"][e["probe_at"]:])) for e in g]
                deaths = any((len(e["continues"]) and e["continues"][-1] < 0.5)
                             for e in g)
                contrast = (max(returns) - min(returns)) + (100.0 if deaths else 0.0)
                for e in g:
                    e["branch_contrast"] = contrast
            out.append(g)
        return out

    def sample_branches(self, batch: int, rng: Optional[np.random.Generator] = None,
                        max_len: int = 44, contrast_oversample: float = 0.5):
        """Batches of complete probe-state groups (all 6 branches together).

        Sampling whole groups makes the counterfactual contrast the dominant
        gradient: MSE-fitting branch episodes independently lets the Q head
        average away rare lethal actions (measured: 2-17 contrast states among
        ~1300 flat ones -> Q stayed flat). ``contrast_oversample`` is the
        probability of drawing a group from the top-contrast quartile instead of
        uniformly.

        Returns (obs (B,T,4,84,84) in [0,1], acts (B,T), rews (B,T),
        conts (B,T), probe_at (B,), valid_len (B,), cont_token (B,)) or None if
        there are no probe episodes. Windows always start at the episode's first
        frame so the posterior at ``probe_at - 1`` is warmed by the shared
        prefix exactly as at deployment.
        """
        import torch
        groups = self._branch_groups()
        if not groups:
            return None
        rng = rng or np.random.default_rng()
        contrasts = [g[0]["branch_contrast"] for g in groups]
        hi_cut = float(np.percentile(contrasts, 75))
        # death groups (contrast > 100 via the death bonus) are the rare,
        # decision-critical ones: they are always in the oversample pool even
        # when flat spread groups dominate the quartile
        hi = [g for g in groups
              if g[0]["branch_contrast"] >= hi_cut or g[0]["branch_contrast"] > 100.0] \
            or groups
        chosen = []
        while len(chosen) < max(batch, 6):
            src = hi if (rng.random() < contrast_oversample and hi) else groups
            chosen.extend(src[int(rng.integers(len(src)))])
        B, T = len(chosen), max_len
        obs = np.zeros((B, T, STACK, 84, 84), dtype=np.float32)
        acts = np.zeros((B, T), dtype=np.int64)
        rews = np.zeros((B, T), dtype=np.float32)
        conts = np.zeros((B, T), dtype=np.float32)
        probe_at = np.zeros(B, dtype=np.int64)
        valid = np.zeros(B, dtype=np.int64)
        tokens = np.zeros(B, dtype=np.int64)
        for b, e in enumerate(chosen):
            L = min(T, len(e["frames"]))
            padded = _pad_frames(e["frames"][:L])
            for t in range(T):
                win = padded[t:t + STACK]
                if len(win) < STACK:
                    win = np.concatenate([win, np.zeros((STACK - len(win),) + win.shape[1:], win.dtype)])
                obs[b, t] = win.transpose(0, 1, 2).astype(np.float32) / 255.0
                if t < L:
                    acts[b, t] = e["actions"][t]
                    rews[b, t] = e["rewards"][t]
                    conts[b, t] = e["continues"][t]
            probe_at[b] = min(e["probe_at"], T - 1)
            valid[b] = L
            tokens[b] = int(e.get("branch_cont", 0))
        return (torch.from_numpy(obs), torch.from_numpy(acts), torch.from_numpy(rews),
                torch.from_numpy(conts), torch.from_numpy(probe_at),
                torch.from_numpy(valid), torch.from_numpy(tokens))

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

    def death_windows(self, seq_len: int):
        """(episode_index, start_index) windows that contain a terminal death.

        Deaths are ~0.1% of steps, so uniform window sampling almost never shows
        the continue head a death -- which is why it learned "never die" and let
        both MPC and imagination rollouts walk off cliffs unpenalized.

        Probe-branch episodes are skipped: their returns are truncated by design
        (they are branch data for the counterfactual Q loss, not full episodes).
        """
        out = []
        for ei, e in enumerate(self.episodes):
            if "probe_at" in e:
                continue
            T = len(e["continues"])
            if T < seq_len + 1:
                continue
            for dpos in np.nonzero(e["continues"] < 0.5)[0]:
                lo = max(0, int(dpos) - seq_len + 1)
                # NOTE: the death is the FINAL frame of an episode, so the latest
                # valid window start is T - seq_len (inclusive). Using T-seq_len-1
                # here silently excluded every death from training.
                hi = min(int(dpos), T - seq_len)
                if hi >= lo:
                    out.append((ei, lo, hi))
        return out

    def sample(self, batch: int, seq_len: int, rng: Optional[np.random.Generator] = None,
               death_oversample: float = 0.0, with_returns: bool = False):
        import torch
        rng = rng or np.random.default_rng()
        if with_returns and any("returns" not in e for e in self.episodes):
            raise RuntimeError("call EpisodeBuffer.ensure_returns(gamma) before "
                               "sampling with returns (train_world_model does this)")
        # Only episodes long enough to fill a sequence (else resample a long one).
        # Probe-branch episodes are excluded from the standard model/value
        # windows: their returns are truncated by design (branch data), and
        # their prefixes are duplicated 6x; they enter training only through the
        # counterfactual branch-Q loss (sample_branches).
        usable_idx = [i for i, e in enumerate(self.episodes)
                      if len(e["frames"]) >= seq_len + 1 and "probe_at" not in e]
        if not usable_idx:
            usable_idx = list(range(len(self.episodes)))
        dw = self.death_windows(seq_len) if death_oversample > 0 else []
        obs = np.zeros((batch, seq_len, STACK, 84, 84), dtype=np.float32)
        acts = np.zeros((batch, seq_len), dtype=np.int64)
        rews = np.zeros((batch, seq_len), dtype=np.float32)
        conts = np.zeros((batch, seq_len), dtype=np.float32)
        rets = np.zeros((batch, seq_len), dtype=np.float32) if with_returns else None
        for b in range(batch):
            if dw and rng.random() < death_oversample:
                ei, lo, hi = dw[rng.integers(len(dw))]
                e = self.episodes[ei]
                start = int(rng.integers(lo, hi + 1))
            else:
                e = self.episodes[usable_idx[rng.integers(len(usable_idx))]]
                # inclusive upper bound so a window can reach the final (terminal) frame
                start = int(rng.integers(0, max(1, len(e["frames"]) - seq_len + 1)))
            T = len(e["frames"])
            padded = _pad_frames(e["frames"])
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
                if with_returns:
                    rets[b, t] = e["returns"][idx]
        out = (torch.from_numpy(obs), torch.from_numpy(acts),
               torch.from_numpy(rews), torch.from_numpy(conts))
        if with_returns:
            out = out + (torch.from_numpy(rets),)
        return out
