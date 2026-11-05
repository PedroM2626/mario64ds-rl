"""QR-DQN extras for rare-manoeuvre problems: held-action exploration and a demo that stays.

* HoldQRDQN: temporally extended epsilon-greedy. SB3's epsilon-greedy replaces the
  greedy action by a random one independently at every step, so a manoeuvre that
  needs one action held for 16 decisions (ds3: up+right for 16, then B) is
  practically never tried. Here an exploration episode picks a random action and
  holds it for 1..hold_max decisions (cf. Dabney et al., 2021, "Temporally-Extended
  epsilon-Greedy Exploration"). The exploration schedule keeps its meaning: with mean
  hold H the start probability is eps / (H - eps * (H - 1)) (a hold's own decisions are
  not start opportunities), which keeps the expected share of random decisions at eps.
  hold_max=1 is the standard behaviour.
* DemoRefreshCallback: the replay buffer is circular, so demonstration transitions
  prefilled at the start are overwritten after buffer_size steps. This re-adds them
  every `every` steps, keeping a fixed share of the buffer demonstrative for the whole
  run (DQfD keeps them permanently; here no margin loss is used).
"""
from __future__ import annotations

import numpy as np
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import BaseCallback


class HoldQRDQN(QRDQN):
    def __init__(self, *args, hold_max: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.hold_max = int(hold_max)
        self._hold_left = None
        self._held = None

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if deterministic or self.hold_max <= 1 or not self.policy.is_vectorized_observation(observation):
            return super().predict(observation, state, episode_start, deterministic)
        n = observation.shape[0]
        if self._hold_left is None or len(self._hold_left) != n:
            self._hold_left = np.zeros(n, dtype=int)
            self._held = np.zeros(n, dtype=int)
        greedy, state = self.policy.predict(observation, state, episode_start, True)
        h = (1 + self.hold_max) / 2.0
        eps = self.exploration_rate
        start_p = min(1.0, eps / (h - eps * (h - 1)))
        action = np.array(greedy, copy=True)
        for i in range(n):
            if self._hold_left[i] == 0 and np.random.rand() < start_p:
                self._held[i] = self.action_space.sample()
                self._hold_left[i] = np.random.randint(1, self.hold_max + 1)
            if self._hold_left[i] > 0:
                action[i] = self._held[i]
                self._hold_left[i] -= 1
        return action, state


class DemoRefreshCallback(BaseCallback):
    """Re-adds the demonstration transitions every `every` environment steps."""

    def __init__(self, transitions, every: int, prefill_fn, verbose: int = 0):
        super().__init__(verbose)
        self.transitions, self.every, self.prefill_fn = transitions, int(every), prefill_fn
        self._next = self.every

    def _on_step(self) -> bool:
        if self.every > 0 and self.num_timesteps >= self._next:
            self.prefill_fn(self.model, self.transitions)
            self._next += self.every
        return True
