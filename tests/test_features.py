"""Shape and forward pass tests for visual feature extractors.

Ensures that IMPALA (Tianshou and SB3 implementations) and NatureCNN (Tianshou)
ingest expected tensor shapes and output valid feature dimensions without numerical instability.
"""

import gymnasium as gym
import torch


def test_tianshou_impala_forward():
    from src.impala_cnn import ImpalaCNN
    net = ImpalaCNN(c=4, h=84, w=84, features_dim=256)
    # Tianshou shape post-FrameStack: (B, Stack, H, W, 1)
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
    # (B, A, atoms), valid probability distribution per action
    assert dist.shape == (2, 6, 51)
    assert torch.allclose(dist.sum(-1), torch.ones(2, 6), atol=1e-4)
