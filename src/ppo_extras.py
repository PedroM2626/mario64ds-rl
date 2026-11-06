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


class BCAnchor(BaseCallback):
    """Keep imitating a demonstration while PPO improves the policy.

    Fine-tuning a behaviour-cloned policy with PPO alone forgot the demonstrated passages (ds3, three
    runs). After every rollout this callback takes `steps` cross-entropy steps on (frame stack, human
    action) pairs from an npz written by scripts/build_bc_set.py, so what the human run shows stays
    fixed while the reward moves the rest.
    """

    def __init__(self, path: str, steps: int = 20, batch: int = 64, lr: float = 1e-5, verbose: int = 0):
        super().__init__(verbose)
        self.path, self.steps, self.batch, self.lr = path, int(steps), int(batch), float(lr)
        self.opt = None
        self.last_loss = float("nan")

    def _on_training_start(self) -> None:
        import numpy as np
        import torch
        z = np.load(self.path)
        self.X, self.Y = z["X"], z["Y"]
        self._rng = np.random.default_rng(0)
        self.opt = torch.optim.Adam(self.model.policy.parameters(), lr=self.lr)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        import torch
        pol = self.model.policy
        losses = []
        for _ in range(self.steps):
            idx = self._rng.integers(0, len(self.Y), self.batch)
            obs_t = pol.obs_to_tensor(self.X[idx])[0]
            a_t = torch.as_tensor(self.Y[idx], device=pol.device)
            loss = -pol.get_distribution(obs_t).log_prob(a_t).mean()
            if not loss.requires_grad:      # the value-head warm-up froze the policy
                return
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(float(loss))
        self.last_loss = sum(losses) / max(1, len(losses))
        self.logger.record("bc_anchor/loss", self.last_loss)
