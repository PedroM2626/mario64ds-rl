"""Extrator IMPALA compatível com Stable-Baselines3.

O ``ImpalaCNN`` em ``src/impala_cnn.py`` foi escrito para o Tianshou
(aceita HWC/5D, retorna ``(features, state)``). O SB3 espera um
``BaseFeaturesExtractor`` que recebe tensor CHW ``(B, C, H, W)`` e retorna
apenas ``(B, features_dim)``. Sem isso, carregar um QR-DQN/PPO com IMPALA
quebra com ``TypeError: Box % int`` (o ``observation_space`` era passado
como ``in_channels``).

Reutiliza os blocos ``ImpalaBlock`` do Tianshou para manter paridade
arquitetural na comparação justa PPO+IMPALA vs Rainbow+IMPALA.
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
        return self.linear(self.cnn(observations))
