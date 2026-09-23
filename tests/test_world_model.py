"""Unit tests for the world model pipeline (NO emulator / ROM required).

Covers: latent shapes, model loss finiteness, gradient isolation during
imagination (model frozen, actor updated), replay stacking convention, and the
belief-state controller producing valid discrete actions.
"""

import numpy as np
import torch

from src.world_model import WorldModel, Latent, model_loss, gauss_kl, _gauss_sample
from src.world_model_agent import ActorCritic, actor_critic_update, actor_bc_update, imagine
from src.wm_replay import EpisodeBuffer, STACK


def _tiny_model():
    return WorldModel(deter_dim=16, stoch_dim=8, enc_dim=16, num_actions=6,
                      obs_channels=4, hidden=16)


def _rand_batch(B=4, T=6):
    obs = torch.rand(B, T, 4, 84, 84)
    acts = torch.randint(0, 6, (B, T))
    rews = torch.randn(B, T)
    conts = torch.ones(B, T)
    return obs, acts, rews, conts


def test_observe_sequence_shapes():
    model = _tiny_model()
    obs, acts, _, _ = _rand_batch()
    out = model.observe_sequence(obs, acts)
    B, T = obs.shape[:2]
    assert out["post_mean"].shape == (B, T, model.stoch_dim)
    assert out["post_deter"].shape == (B, T, model.deter_dim)
    assert out["prior_std"].shape == (B, T, model.stoch_dim)
    assert torch.isfinite(out["post_mean"]).all()


def test_model_loss_finite_and_backward():
    model = _tiny_model()
    obs, acts, rews, conts = _rand_batch()
    loss = model_loss(model, obs, acts, rews, conts)
    assert torch.isfinite(loss["loss"])
    loss["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)


def test_gauss_kl_nonnegative():
    m = torch.randn(3, 5)
    s = torch.rand(3, 5) + 0.5
    kl = gauss_kl(m, s, m, s, free_nats=0.0)
    assert (kl >= -1e-5).all()


def test_imagination_updates_actor_only():
    """The world model must stay frozen while the actor-critic trains on imagination."""
    model = _tiny_model()
    for p in model.parameters():
        p.requires_grad_(False)
    ac = ActorCritic(model.deter_dim, model.stoch_dim, 6, hidden=16)
    opt = torch.optim.Adam(ac.parameters(), lr=1e-3)
    obs, acts, _, _ = _rand_batch(B=4, T=6)
    with torch.no_grad():
        out = model.observe_sequence(obs, acts)
    start = Latent(out["post_deter"][:, -1], out["post_mean"][:, -1],
                   out["post_std"][:, -1], _gauss_sample(out["post_mean"][:, -1],
                                                         out["post_std"][:, -1]))
    before = [p.clone() for p in ac.parameters()]
    metrics = actor_critic_update(model, ac, opt, start, horizon=5)
    assert set(["policy_loss", "value_loss", "entropy"]).issubset(metrics)
    changed = any(not torch.equal(a, b) for a, b in zip(before, ac.parameters()))
    assert changed, "actor/critic parameters should update"
    # model params unchanged (frozen)
    assert all(p.grad is None for p in model.parameters())


def test_imagine_returns_finite():
    model = _tiny_model().eval()
    ac = ActorCritic(model.deter_dim, model.stoch_dim, 6, hidden=16)
    start = model.initial_state(3, torch.device("cpu"))
    d = imagine(model, ac, start, horizon=8, gamma=0.99, lam=0.95)
    assert d["imgs"].shape == (3, 8, model.deter_dim + model.stoch_dim)
    assert torch.isfinite(d["returns"]).all()


def test_replay_stacking_and_range():
    from src.wm_replay import _pad_frames
    buf = EpisodeBuffer()
    T = 20
    frames = (np.random.rand(T, 84, 84) * 255).astype(np.uint8)
    buf.add(frames, np.zeros(T, np.int64), np.zeros(T, np.float32), np.ones(T, np.float32))
    obs, acts, rews, conts = buf.sample(batch=2, seq_len=6, rng=np.random.default_rng(0))
    assert obs.shape == (2, 6, STACK, 84, 84)
    assert obs.min() >= 0.0 and obs.max() <= 1.0
    # newest channel (index -1) of every stacked frame is a real (nonzero) frame
    assert obs[:, :, -1].flatten().abs().sum() > 0.0
    # padding convention: 3 leading zero frames then the episode frames
    padded = _pad_frames(frames)
    assert padded.shape[0] == T + (STACK - 1)
    assert padded[:STACK - 1].sum() == 0
    assert np.array_equal(padded[STACK - 1:], frames)


def test_controller_produces_valid_action():
    from src.world_model_controller import WorldModelController
    model = _tiny_model()
    ac = ActorCritic(model.deter_dim, model.stoch_dim, 6, hidden=16)
    ctrl = WorldModelController(model, ac, device="cpu", deterministic=True)
    a0 = ctrl.reset((np.random.rand(84, 84) * 255).astype(np.uint8))
    a1 = ctrl.act((np.random.rand(84, 84) * 255).astype(np.uint8))
    assert 0 <= a0 < 6 and 0 <= a1 < 6


def test_actor_bc_reduces_loss():
    """Distillation should drive the actor to reproduce the expert label."""
    model = _tiny_model().eval()
    ac = ActorCritic(model.deter_dim, model.stoch_dim, 6, hidden=16)
    opt = torch.optim.Adam(ac.parameters(), lr=3e-3)
    obs, acts, _, _ = _rand_batch(B=4, T=6)
    label = torch.randint(0, 6, (4, 6))
    first = None
    for _ in range(40):
        m = actor_bc_update(model, ac, opt, obs, acts, label)
        if first is None:
            first = m["bc_loss"]
    assert m["bc_loss"] < first, "BC loss should decrease when fitting labels"


def test_expert_inference_and_sampling():
    buf = EpisodeBuffer()
    T = 30
    frames = (np.random.rand(T, 84, 84) * 255).astype(np.uint8)
    # a completed (survived) episode -> continues all 1
    buf.add(frames, np.arange(T) % 6, np.zeros(T, np.float32), np.ones(T, np.float32))
    # a died episode -> last continue 0
    cont2 = np.ones(T, np.float32); cont2[-1] = 0.0
    buf.add(frames, np.arange(T) % 6, np.zeros(T, np.float32), cont2)
    experts = buf.expert_episodes()  # untagged -> none (source is None)
    assert experts == []
    import src.train_world_model as twm
    inferred = twm._infer_expert_episodes(buf)
    assert len(inferred) == 1, "only the survived episode is a descent demo"
    obs, ein, elab = buf.sample_expert(2, 6, inferred, np.random.default_rng(0))
    assert obs.shape == (2, 6, STACK, 84, 84)
    assert ein.shape == (2, 6) and elab.shape == (2, 6)
    assert elab.max() < 6 and ein.max() < 6
