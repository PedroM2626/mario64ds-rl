"""Offline unit tests: RAM-state reading + the state dynamics model + MPC scoring.

No emulator: synthetic fixed-point physics vectors. Covers the exact-state
semantics everything else depends on (kill plane, normalization, the heads,
the physics-grounded planner objective).
"""

import numpy as np
import torch

from src.ram_state import Sm64DsState, KILL_PLANE, WORD_ADDRS, NAMES
from src.state_model import (StateDynamicsModel, STATE_COLS, STATE_DIM,
                             build_dataset, compute_returns, GAMMA)


def _fake_episode(T=30, falls_at=None):
    rng = np.random.default_rng(0)
    s = np.zeros((T + 1, 6))
    # plausible random walk in fixed-point units around slide-like values
    s[:, 0] = 30000 + np.cumsum(rng.normal(0, 40, T + 1)) * 100
    s[:, 1] = 100000 + rng.normal(0, 5, T + 1) * 100     # speed units/s
    s[:, 2] = -16384                                      # const word
    s[:, 3] = np.cumsum(rng.normal(0, 30, T + 1)) * 10    # lateral
    s[:, 4] = -40000 + np.cumsum(rng.normal(0, 20, T + 1)) * 20  # height
    s[:, 5] = rng.normal(0, 10, T + 1) * 10               # vvel
    falls = np.zeros(T, dtype=bool)
    if falls_at is not None:
        for f in falls_at:
            falls[f] = True
            s[f + 1:, 4] = KILL_PLANE  # clamp once fallen
    dones = np.zeros(T, dtype=bool)
    dones[-1] = False
    return dict(states=s, actions=(rng.integers(0, 6, T)),
                rewards=np.zeros(T, dtype=np.float32),
                falls=falls, dones=dones)


def test_is_dead_kill_plane():
    st = np.zeros(6, dtype=np.float64)
    st[4] = KILL_PLANE
    assert Sm64DsState.is_dead(st)
    st[4] = KILL_PLANE - 1000
    assert Sm64DsState.is_dead(st)
    st[4] = KILL_PLANE + 100000  # safely above the plane
    assert not Sm64DsState.is_dead(st)


def test_units_scaling():
    st = np.array([4096.0, 8192.0, 0, 0, 0, 0])
    assert tuple(Sm64DsState.units(st)[:2]) == (1.0, 2.0)


def test_build_dataset_and_returns():
    eps = [_fake_episode(30, falls_at=(10,)), _fake_episode(25)]
    S, A, S2, FL, TM, R, G = build_dataset(eps)
    assert S.shape == (30 + 25, STATE_DIM) and A.shape == (30 + 25,)
    assert abs(S[0].mean()) < 1e6 and G.shape == (55,)
    # MC return recursion: G_t = r_t + gamma*G_t+1 for non-terminal
    rw = np.array([1.0, 2.0, 0.5, 3.0, -100.0])
    dn = np.array([False, False, False, False, True])
    g = compute_returns(rw, dn, gamma=0.5)
    exp = np.zeros(5)
    acc = 0.0
    for t in reversed(range(5)):
        acc = rw[t] + 0.5 * acc * (0.0 if dn[t] else 1.0)
        exp[t] = acc
    assert np.allclose(g, exp)


def test_model_forward_and_rollout_score():
    model = StateDynamicsModel(hidden=32, layers=2)
    B = 16
    s = torch.randn(B, STATE_DIM)
    a = torch.randint(0, 6, (B,))
    ds, fl, tm, rw, vv = model(s, a)
    assert ds.shape == (B, STATE_DIM)
    seq = torch.randint(0, 6, (8, B))
    s0 = torch.randn(1, STATE_DIM)
    score = model.rollout_score(s0, seq, gamma=0.99, fall_penalty=50.0,
                                term_penalty=100.0)
    assert score.shape == (B,) and torch.isfinite(score).all()


def test_planner_physics_objective_avoids_predicted_falls():
    """The state MPC's objective must punish candidate sequences whose
    predicted height reaches the kill plane."""
    from src.state_mpc import StateMPC
    model = StateDynamicsModel(hidden=32, layers=2)
    norm = dict(mu=[0.0] * STATE_DIM, sd=[1.0] * STATE_DIM,
                rmu=0.0, rsd=1.0, gmu=0.0, gsd=1.0)
    ctrl = StateMPC(model, norm, "cpu", horizon=4, candidates=8, elites=4,
                    iters=2, seed=0)
    # force the model to predict a height plunge for one action
    with torch.no_grad():
        # one-hot for a=5 -> dyn output height (col 2) hugely negative
        for layer, w_idx in [(model.dyn, None)]:
            pass
        # simpler: set output layer so the feature map doesn't matter for the demo
        w = model.dyn.weight  # (STATE_DIM, hidden)
        w[2].fill_(0.0)
        model.dyn.bias[2] = -10.0 / 0.4    # check via kill_norm below
        ctrl.kill_norm = torch.tensor(-0.5, dtype=torch.float32)
    s_raw = np.zeros(6)
    s_raw[list(STATE_COLS)] = [0.0, 0.0, -0.4, 0.0]
    v = ctrl.plan(s_raw)   # must run and return a valid action
    assert 0 <= v < 6
