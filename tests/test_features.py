"""Testes de shape dos extratores visuais (comparação justa).

Garante que IMPALA (Tianshou e SB3) e NatureCNN (Tianshou) aceitam o
formato do projeto e retornam o features_dim esperado.
"""

import gymnasium as gym
import torch


def test_tianshou_impala_forward():
    from src.impala_cnn import ImpalaCNN
    net = ImpalaCNN(c=4, h=84, w=84, features_dim=256)
    # Formato Tianshou pós-FrameStack: (B, Stack, H, W, 1)
    obs = torch.randint(0, 256, (2, 4, 84, 84, 1), dtype=torch.uint8)
    feats, _ = net(obs)
    assert feats.shape == (2, 256)
    assert not torch.isnan(feats).any()


def test_tianshou_nature_forward():
    from src.impala_cnn import TianshouNatureCNN
    net = TianshouNatureCNN(c=4, h=84, w=84, features_dim=512)
    obs = torch.randint(0, 256, (2, 4, 84, 84, 1), dtype=torch.uint8)
    feats, _ = net(obs)
    assert feats.shape == (2, 512)
    assert not torch.isnan(feats).any()


def test_sb3_impala_extractor_forward():
    from src.sb3_impala import ImpalaFeaturesExtractor
    space = gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype="uint8")
    net = ImpalaFeaturesExtractor(space, features_dim=256)
    obs = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8).float() / 255.0
    feats = net(obs)
    assert feats.shape == (2, 256)
    assert not torch.isnan(feats).any()


def test_rainbow_net_dueling_output():
    from src.impala_cnn import ImpalaCNN, RainbowNet
    feature_net = ImpalaCNN(c=4, h=84, w=84, features_dim=256)
    model = RainbowNet(feature_net, 6, 51)
    obs = torch.randint(0, 256, (2, 4, 84, 84, 1), dtype=torch.uint8)
    dist, _ = model(obs)
    # (B, A, átomos), distribuição válida por ação
    assert dist.shape == (2, 6, 51)
    assert torch.allclose(dist.sum(-1), torch.ones(2, 6), atol=1e-4)
