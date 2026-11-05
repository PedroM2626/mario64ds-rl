"""PPO helpers for fine-tuning a behaviour-cloned policy without wrecking it.

ValueWarmup: a cloned policy comes with a value head that has never seen the task's
reward, so the first advantages are noise and the policy updates they drive undo the
clone (measured on ds3: the red-arrow passage was lost within 100k steps). For the
first `steps` environment steps only the value head (`value_net` parameters, including
the one inside mlp_extractor) is trained; the policy and the shared CNN stay frozen
and keep acting as the data collector.
"""
from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class ValueWarmup(BaseCallback):
    def __init__(self, steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.steps = int(steps)
        self._start = None
        self._frozen = False

    def _on_training_start(self) -> None:
        self._start = self.num_timesteps
        if self.steps > 0:
            for name, p in self.model.policy.named_parameters():
                p.requires_grad = "value_net" in name
            self._frozen = True

    def _on_step(self) -> bool:
        if self._frozen and self.num_timesteps - self._start >= self.steps:
            for p in self.model.policy.parameters():
                p.requires_grad = True
            self._frozen = False
        return True
