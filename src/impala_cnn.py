import torch
import torch.nn as nn
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(x)
        out = self.conv1(out)
        out = self.relu2(out)
        out = self.conv2(out)
        return out + x

class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class ImpalaCNN(nn.Module):
    """
    Extrator de features baseado na arquitetura IMPALA, com blocos residuais.
    Agora é um módulo puro do PyTorch para rodar no Tianshou Rainbow.
    """
    def __init__(self, c=1, h=84, w=84, features_dim: int = 256):
        super(ImpalaCNN, self).__init__()
        
        self.cnn = nn.Sequential(
            ImpalaBlock(c, 16),
            ImpalaBlock(16, 32),
            ImpalaBlock(32, 32),
            nn.ReLU()
        )
        
        with torch.no_grad():
            sample_input = torch.zeros(1, c, h, w)
            output_shape = self.cnn(sample_input).shape[1:]
            
        n_flatten = int(np.prod(output_shape))
        
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
        self.output_dim = features_dim

    def forward(self, observations: torch.Tensor, state=None, info={}):
        if not isinstance(observations, torch.Tensor):
            observations = torch.as_tensor(observations, dtype=torch.float32, device=next(self.parameters()).device)
            
        # Tianshou converts dictionary/arrays to tensors, but normally Gym envs are H,W,C for images.
        # We need to ensure CHW format for PyTorch Conv2D.
        if observations.dim() == 5 and observations.shape[-1] == 1:
            # Output do FrameStackObservation: (Batch, Stack, H, W, 1) -> Squeeze -> (Batch, Stack, H, W)
            observations = observations.squeeze(-1)
        elif observations.dim() == 4 and observations.shape[-1] in [1, 3, 4]:
            # (Batch, H, W, C) -> (Batch, C, H, W)
            observations = observations.permute(0, 3, 1, 2)
        elif observations.dim() == 3 and observations.shape[-1] in [1, 3, 4]:
            # Unbatched HW C
            observations = observations.permute(2, 0, 1).unsqueeze(0)
            
        # Scale to [0, 1] se ainda vier como uint8 (0-255)
        if observations.max() > 1.0:
            observations = observations.float() / 255.0
        return self.linear(self.cnn(observations)), state


class TianshouNatureCNN(nn.Module):
    """NatureCNN (Mnih et al., 2015) adaptada para o Tianshou.

    Mesma arquitetura do ``CnnPolicy`` padrão do SB3, para comparação justa
    Rainbow+NatureCNN vs PPO+NatureCNN. Aceita os mesmos formatos HWC/5D
    que ``ImpalaCNN`` e expõe ``output_dim``.
    """

    def __init__(self, c=4, h=84, w=84, features_dim: int = 512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, c, h, w)).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        self.output_dim = features_dim

    def forward(self, observations: torch.Tensor, state=None, info={}):
        if not isinstance(observations, torch.Tensor):
            observations = torch.as_tensor(
                observations, dtype=torch.float32, device=next(self.parameters()).device
            )
        if observations.dim() == 5 and observations.shape[-1] == 1:
            observations = observations.squeeze(-1)
        elif observations.dim() == 4 and observations.shape[-1] in [1, 3, 4]:
            observations = observations.permute(0, 3, 1, 2)
        elif observations.dim() == 3 and observations.shape[-1] in [1, 3, 4]:
            observations = observations.permute(2, 0, 1).unsqueeze(0)
        if observations.max() > 1.0:
            observations = observations.float() / 255.0
        return self.linear(self.cnn(observations)), state

from tianshou.utils.net.discrete import NoisyLinear

class RainbowNet(nn.Module):
    """
    Combina o ImpalaCNN com as cabeças Dueling Networks e NoisyLinear.
    Retorna logits (batch_size, action_shape * num_atoms) para o RainbowPolicy.
    """
    def __init__(self, feature_net, action_shape, num_atoms=51, noisy_std=0.1):
        super().__init__()
        self.feature_net = feature_net
        self.action_shape = action_shape
        self.num_atoms = num_atoms
        
        # Cabeça Value (V) - Estima o valor do estado
        self.Q_V = nn.Sequential(
            NoisyLinear(feature_net.output_dim, 256, noisy_std),
            nn.ReLU(inplace=True),
            NoisyLinear(256, num_atoms, noisy_std)
        )
        
        # Cabeça Advantage (A) - Estima a vantagem de cada ação
        self.Q_A = nn.Sequential(
            NoisyLinear(feature_net.output_dim, 256, noisy_std),
            nn.ReLU(inplace=True),
            NoisyLinear(256, action_shape * num_atoms, noisy_std)
        )

    def forward(self, obs, state=None, info={}):
        features, state = self.feature_net(obs, state, info)
        
        v = self.Q_V(features).view(-1, 1, self.num_atoms)
        a = self.Q_A(features).view(-1, self.action_shape, self.num_atoms)
        
        # Dueling formula: Q(s,a) = V(s) + A(s,a) - mean(A)
        q = v + a - a.mean(dim=1, keepdim=True)
        
        # A política C51 do Tianshou espera distribuições de probabilidade, não apenas logits!
        # Portanto, aplicamos softmax na dimensão dos átomos (dim=-1)
        dist = torch.softmax(q, dim=-1)
        
        return dist, state
