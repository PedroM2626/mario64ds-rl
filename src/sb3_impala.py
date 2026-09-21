"""IMPALA visual feature extractor compatible with Stable-Baselines3.

The ``ImpalaCNN`` in ``src/impala_cnn.py`` was tailored for Tianshou
(accepts HWC/5D formats, returns ``(features, state)``). SB3 requires a
``BaseFeaturesExtractor`` subclass that ingests CHW tensors ``(B, C, H, W)``
and returns directly ``(B, features_dim)``.

Reuses the ``ImpalaBlock`` definitions from Tianshou to preserve architectural
parity in controlled benchmark evaluations (e.g., PPO+IMPALA vs Rainbow+IMPALA).
"""

import gymnasium as gym
import torch as th
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.impala_cnn import ImpalaBlock


class ImpalaFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            ImpalaBlock(n_input_channels, 16),
            ImpalaBlock(16, 32),
            ImpalaBlock(32, 32),
            nn.ReLU(),
        )

        with th.no_grad():
            sample = th.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1:].numel()

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if observations.dim() == 5 and observations.shape[-1] == 1:
            observations = observations.squeeze(-1)
        return self.linear(self.cnn(observations))
