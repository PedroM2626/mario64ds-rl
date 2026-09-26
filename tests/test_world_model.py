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


def test_terminal_death_reaches_training_windows():
    """Regression: terminal deaths sit on the FINAL frame, so a window sampler that
    caps its start at T-seq_len-1 silently excludes every death and the continue
    head learns "never die" (this blinded MPC/imagination to cliffs)."""
    buf = EpisodeBuffer()
    T, L = 60, 20
    for _ in range(3):
        frames = (np.random.rand(T, 84, 84) * 255).astype(np.uint8)
        cont = np.ones(T, np.float32)
        cont[-1] = 0.0                      # death on the final frame only
        buf.add(frames, np.zeros(T, np.int64), np.zeros(T, np.float32),
                cont, source="random")
    # the death-window index must find windows that actually cover the final frame
    dw = buf.death_windows(L)
    assert len(dw) > 0, "death_windows() found no death despite terminal continues"
    for _, lo, hi in dw:
        assert hi >= lo
        assert hi + L - 1 >= T - 1, "death window must reach the terminal frame"

    # uniform sampling must be able to surface a death at all
    rng = np.random.default_rng(0)
    seen = 0
    for _ in range(40):
        _, _, _, conts = buf.sample(8, L, rng)
        seen += int((conts < 0.5).sum())
    assert seen > 0, "no terminal death ever appeared in uniform samples"

    # death oversampling must surface deaths far more often
    _, _, _, overs = buf.sample(8, L, np.random.default_rng(1), death_oversample=1.0)
    assert (overs < 0.5).sum() >= 1, "death_oversample=1.0 produced no death"


def test_mask_terminal_reward_ignores_death_spike():
    """The -100 terminal spike must not enter the reward MSE, otherwise the fitted
    dense flow prediction is dragged negative and the planner loses its gradient."""
    model = _tiny_model()
    obs, acts, rews, conts = _rand_batch(B=4, T=6)
    rews = rews.clone(); conts = conts.clone()
    rews[:, -1] = -100.0
    conts[:, -1] = 0.0                      # terminal step carries the spike
    masked = model_loss(model, obs, acts, rews, conts, mask_terminal_reward=True)
    unmasked = model_loss(model, obs, acts, rews, conts)
    assert float(masked["rew"]) < float(unmasked["rew"]), \
        "masking terminal steps must remove the -100 outlier from the reward loss"
    assert torch.isfinite(masked["loss"])


def test_death_weight_changes_continue_loss_scale():
    model = _tiny_model()
    obs, acts, rews, conts = _rand_batch(B=4, T=6)
    conts = conts.clone()
    conts[:, -1] = 0.0                      # one death per row
    base = model_loss(model, obs, acts, rews, conts, death_weight=1.0)
    heavy = model_loss(model, obs, acts, rews, conts, death_weight=50.0)
    assert torch.isfinite(heavy["loss"])
    assert "cont_recall" in heavy
    assert float(heavy["cont"]) != float(base["cont"])


def test_controller_produces_valid_action():
    from src.world_model_controller import WorldModelController
    model = _tiny_model()
    ac = ActorCritic(model.deter_dim, model.stoch_dim, 6, hidden=16)
    ctrl = WorldModelController(model, ac, device="cpu", deterministic=True)
    a0 = ctrl.reset((np.random.rand(84, 84) * 255).astype(np.uint8))
    a1 = ctrl.act((np.random.rand(84, 84) * 255).astype(np.uint8))
    assert 0 <= a0 < 6 and 0 <= a1 < 6


def test_mpc_rollout_value_finite_and_terminal_cost():
    """MPC value rollouts must be finite and a predicted death must lower value."""
    from src.world_model import Latent
    model = _tiny_model().eval()
    start = model.initial_state(1, torch.device("cpu"))
    T, N = 6, 5
    seq = torch.randint(0, 6, (T, N))
    v = model.rollout_value(start, seq, gamma=0.99, pessimism=0.0, terminal_cost=0.0)
    assert v.shape == (N,) and torch.isfinite(v).all()
    v_term = model.rollout_value(start, seq, gamma=0.99, pessimism=0.0,
                                 terminal_cost=1000.0, death_thresh=0.999)
    # with almost every state flagged fatal, the terminated value must be lower
    assert (v_term <= v + 1e-5).all()


def test_episode_returns_exact():
    """Discounted returns must match the closed-form recursion, death spike included."""
    buf = EpisodeBuffer()
    T = 5
    r = np.array([1.0, 2.0, 0.5, 3.0, -100.0], dtype=np.float32)
    cont = np.ones(T, np.float32); cont[-1] = 0.0
    frames = (np.random.rand(T, 84, 84) * 255).astype(np.uint8)
    buf.add(frames, np.zeros(T, np.int64), r, cont, source="random")
    gamma = 0.5
    buf.ensure_returns(gamma)
    g = buf.episodes[0]["returns"]
    expected = np.zeros(T)
    acc = 0.0
    for t in reversed(range(T)):
        acc = r[t] + gamma * acc
        expected[t] = acc
    assert np.allclose(g, expected, atol=1e-4), f"{g} vs {expected}"
    # caching: second call with same gamma is a no-op, different gamma recomputes
    buf.ensure_returns(gamma)
    assert np.allclose(buf.episodes[0]["returns"], expected, atol=1e-4)
    buf.ensure_returns(0.9)
    assert not np.allclose(buf.episodes[0]["returns"], expected, atol=1e-4)


def test_sample_with_returns_alignment():
    """The returns channel must align with the reward channel of the same window."""
    buf = EpisodeBuffer()
    T, L = 40, 10
    frames = (np.random.rand(T, 84, 84) * 255).astype(np.uint8)
    r = np.ones(T, dtype=np.float32)  # constant reward -> strictly increasing return
    buf.add(frames, np.zeros(T, np.int64), r, np.ones(T, np.float32), source="random")
    buf.ensure_returns(0.99)
    s = buf.sample(32, L, np.random.default_rng(0), with_returns=True)
    obs, acts, rews, conts, rets = s
    assert rets.shape == rews.shape
    # G_t = 1 + gamma * G_{t+1} over a constant-reward episode shrinks as t
    # advances (less future reward remains), so a correctly aligned returns
    # window must be strictly decreasing
    assert (rets[:, 1:] < rets[:, :-1]).all()
    assert torch.allclose(rews, torch.ones_like(rews))
    # and sampling with returns requires ensure_returns first
    buf2 = EpisodeBuffer()
    buf2.add(frames, np.zeros(T, np.int64), r, np.ones(T, np.float32))
    try:
        buf2.sample(2, L, np.random.default_rng(0), with_returns=True)
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_model_loss_trains_value_head():
    """model_loss with returns must add a finite value loss that backprops."""
    model = _tiny_model()
    obs, acts, rews, conts = _rand_batch(B=4, T=6)
    returns = torch.randn(4, 6)
    out = model_loss(model, obs, acts, rews, conts, returns_seq=returns,
                     value_weight=1.0)
    assert "value" in out and torch.isfinite(out["value"])
    assert "value_corr" in out
    out["loss"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.value_net.parameters())
    # value_weight=0 must keep the legacy behaviour (no value loss key)
    out0 = model_loss(model, obs, acts, rews, conts, returns_seq=returns,
                      value_weight=0.0)
    assert "value" not in out0


def test_model_loss_value_prior_anchoring_runs():
    """The prior-rollout value anchor (k in [1, prior_rollout_max]) must train and
    stay finite even when the window is too short for a rollout (k path skipped)."""
    model = _tiny_model()
    obs, acts, rews, conts = _rand_batch(B=4, T=20)
    returns = torch.randn(4, 20)
    out = model_loss(model, obs, acts, rews, conts, returns_seq=returns,
                     value_weight=1.0, prior_rollout_max=12)
    assert torch.isfinite(out["value"])
    out["loss"].backward()
    # rollout never longer than the window: k <= T-1
    out2 = model_loss(model, obs, acts, rews, conts, returns_seq=returns,
                      value_weight=1.0, prior_rollout_max=50)
    assert torch.isfinite(out2["value"])
    # too-short windows must not crash (K==0 path)
    obs3, acts3, rews3, conts3 = _rand_batch(B=2, T=2)
    r3 = torch.randn(2, 2)
    out3 = model_loss(model, obs3, acts3, rews3, conts3, returns_seq=r3,
                      value_weight=1.0)
    assert torch.isfinite(out3["value"])


def test_model_loss_trains_q_head():
    """model_loss with returns must also fit Q(s, taken action) and report it."""
    model = _tiny_model()
    obs, acts, rews, conts = _rand_batch(B=4, T=10)
    returns = torch.randn(4, 10)
    out = model_loss(model, obs, acts, rews, conts, returns_seq=returns,
                     value_weight=1.0)
    assert "q" in out and torch.isfinite(out["q"])
    assert "q_corr" in out
    out["loss"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.q_net.parameters())


def test_sample_branches_and_branch_q_loss():
    """Probe-branch group sampling + the counterfactual branch-Q loss must run."""
    buf = EpisodeBuffer()
    P, K = 6, 5
    pre_frames = (np.random.rand(P, 84, 84) * 255).astype(np.uint8)
    pre_actions = np.zeros(P, np.int64)
    pre_rewards = np.zeros(P, np.float32)
    pre_continues = np.ones(P, np.float32)
    for a in range(6):
        br_frames = (np.random.rand(K, 84, 84) * 255).astype(np.uint8)
        frames = np.concatenate([pre_frames, br_frames])
        acts = np.concatenate([pre_actions, np.full(K, a, np.int64)])
        rews = np.concatenate([pre_rewards, (a - 3.0) * np.ones(K, np.float32)])
        cont = np.concatenate([pre_continues, np.ones(K, np.float32)])
        cont[-1] = 0.0 if a == 0 else 1.0          # branch a=0 dies at the end
        buf.add(frames, acts, rews, cont, source="probe", probe_at=P, probe_id=1,
                branch_cont=0)
    assert len(buf.probe_episodes()) == 6
    s = buf.sample_branches(6, np.random.default_rng(0), max_len=P + K)
    assert s is not None
    obs, acts, rews, conts, bpa, bvalid, btok = s
    assert obs.shape == (6, P + K, STACK, 84, 84)
    assert bpa.shape == (6,) and bvalid.shape == (6,) and btok.shape == (6,)
    assert int(bpa.min()) == P
    assert int(btok.min()) == 0

    model = _tiny_model()
    model.ret_mean.fill_(0.0)
    model.ret_std.fill_(1.0)
    out = model_loss(model, obs[:, :6], acts[:, :6], rews[:, :6], conts[:, :6],
                     value_weight=1.0, returns_seq=rews[:, :6],
                     branch=(obs, acts, rews, conts, bpa, bvalid, btok),
                     branch_q_weight=1.0, branch_gamma=0.9)
    assert "branch_q" in out and torch.isfinite(out["branch_q"])
    out["loss"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.q_net.parameters())


def test_rollout_value_q_grounding_identity():
    """With no predicted deaths, q_ground must REPLACE the t=0 model reward with
    the Q term: value(qw) = value(0) - r_hat(s_1) + qw * Q(s_0, a_0)."""
    from src.world_model import Latent
    model = _tiny_model().eval()
    model.ret_mean.fill_(0.0)
    model.ret_std.fill_(1.0)
    with torch.no_grad():  # continue head always alive -> no dying, exact identity
        for p in model.continue_net.parameters():
            p.fill_(10.0)
    start = model.initial_state(1, torch.device("cpu"))
    T, N = 5, 7
    seq = torch.randint(0, 6, (T, N))
    gamma = 0.9
    v0 = model.rollout_value(start, seq, gamma=gamma, terminal_cost=0.0,
                             death_thresh=0.5, q_weight=0.0)
    vq = model.rollout_value(start, seq, gamma=gamma, terminal_cost=0.0,
                              death_thresh=0.5, q_weight=1.0, q_cont=0)
    # manual: first step reward and Q term (noop-token read, per --q-cont 0)
    lat = Latent(*[t.expand(N, *t.shape[1:]) for t in start])
    lat = Latent(lat.deter, lat.mean, lat.std, lat.mean)
    lat1 = model.imagine_step(lat, seq[0], deterministic=True)
    r0 = model.reward_net(model.img(lat1)).squeeze(-1)
    q0 = model.q(lat, seq[0], cont_token=torch.zeros(N, dtype=torch.long))
    expected = v0 - r0 + q0
    assert torch.allclose(vq, expected, atol=1e-4), f"{vq} vs {expected}"


def test_rollout_value_bootstrap_matches_head():
    """With no predicted deaths, value_boot must add exactly gamma^T * V(s_T)."""
    from src.world_model import Latent
    model = _tiny_model().eval()
    model.ret_mean.fill_(0.0)
    model.ret_std.fill_(1.0)
    # bias the continue head to always predict "alive" so no death intervenes
    with torch.no_grad():
        for p in model.continue_net.parameters():
            p.fill_(10.0)
    start = model.initial_state(1, torch.device("cpu"))  # planners tile batch-1 starts
    T, N = 5, 7
    seq = torch.randint(0, 6, (T, N))
    gamma = 0.9
    v0 = model.rollout_value(start, seq, gamma=gamma, terminal_cost=0.0,
                             death_thresh=0.5, value_boot=False)
    vb = model.rollout_value(start, seq, gamma=gamma, terminal_cost=0.0,
                             death_thresh=0.5, value_boot=True)
    # manual final latent + pessimistic (min) ensemble read
    lat = Latent(*[t.expand(N, *t.shape[1:]) for t in start])
    lat = Latent(lat.deter, lat.mean, lat.std, lat.mean)
    for t in range(T):
        lat = model.imagine_step(lat, seq[t], deterministic=True)
    img_end = model.img(lat)
    v_end = torch.stack([(h(img_end).squeeze(-1) * model.ret_std + model.ret_mean)
                        for h in model.value_heads()], 0).min(0).values
    expected = v0 + (gamma ** T) * v_end
    assert torch.allclose(vb, expected, atol=1e-4), f"{vb} vs {expected}"


def test_mpc_controller_value_boot_smoke():
    """MPC must plan valid actions with the value bootstrap enabled."""
    from src.mpc_world_model import MPCController
    model = _tiny_model().eval()
    ctrl = MPCController(model, device="cpu", horizon=4, candidates=8, elites=4,
                         iters=2, value_boot=True)
    a0 = ctrl.reset((np.random.rand(84, 84) * 255).astype(np.uint8))
    a1 = ctrl.act((np.random.rand(84, 84) * 255).astype(np.uint8))
    assert 0 <= a0 < 6 and 0 <= a1 < 6


def test_memoryless_dataset_and_head():
    """Offline: memoryless BC dataset shape + head forward (no emulator)."""
    import torch as th
    from src.wm_replay import EpisodeBuffer
    from src.distill_memoryless import build_dataset, MemorylessPolicy
    buf = EpisodeBuffer()
    T = 25
    frames = (np.random.rand(T, 84, 84) * 255).astype(np.uint8)
    buf.add(frames, np.arange(T) % 6, np.zeros(T, np.float32),
            np.ones(T, np.float32), source="ppo", completed=True)
    X, Y = build_dataset(buf, np.random.default_rng(0))
    assert X.shape == (T, 4, 84, 84) and X.min() >= 0 and X.max() <= 1
    assert Y.shape == (T,) and Y.max() < 6
    head = MemorylessPolicy(16, 6, 16)
    out = head(th.rand(3, 16))
    assert out.shape == (3, 6)
    assert 0 <= int(head.act(th.rand(1, 16)).item()) < 6


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
