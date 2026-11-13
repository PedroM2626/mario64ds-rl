"""Replay buffer bookkeeping and one update of the BTR-style agent (no emulator)."""
import numpy as np
import pytest
import torch

from src.btr import BTRAgent, IQNNet, PrioritizedStepBuffer


def _fill(buf, steps, done_at=None):
    """Env e at step t sees a frame full of 10*e + t%100 (+1 so that 0 means 'blank'); reward 1 per step."""
    done_at = done_at or {}
    for t in range(steps):
        frames = np.stack([np.full((84, 84), 10 * e + (t % 100) + 1, np.uint8) for e in range(buf.E)])
        dones = np.array([t in done_at.get(e, ()) for e in range(buf.E)])
        buf.add(frames, np.full(buf.E, t % 4), np.ones(buf.E, np.float32), dones)


def test_stack_blanks_frames_before_the_episode_start():
    buf = PrioritizedStepBuffer(1, 64, n_step=3)
    _fill(buf, 12, done_at={0: (5,)})          # the first episode is steps 0..5, the second starts at 6
    # steps 4 and 5 belong to the old episode and are blanked; steps 6 and 7 show frames 7 and 8
    assert [int(f[0, 0]) for f in buf._stack_at(0, 7)] == [0, 0, 7, 8]
    assert [int(f[0, 0]) for f in buf._stack_at(0, 3)] == [1, 2, 3, 4]


def test_stack_values_in_the_middle_of_an_episode():
    buf = PrioritizedStepBuffer(2, 64, n_step=3)
    _fill(buf, 20)
    assert [int(f[0, 0]) for f in buf._stack_at(1, 10)] == [10 + 8, 10 + 9, 10 + 10, 10 + 11]
    assert [int(f[0, 0]) for f in buf._stack_at(0, 1)] == [0, 0, 1, 2]


def test_nstep_return_and_terminal_masking():
    buf = PrioritizedStepBuffer(1, 64, n_step=3, gamma=0.5)
    _fill(buf, 20, done_at={0: (7,)})
    buf.prio[:] = 0.0
    for t in (5, 10):
        buf.prio[0, t] = 1.0
    seen = {}
    for _ in range(40):
        b = buf.sample(2, beta=0.4)
        for i in range(2):
            seen[int(b["obs"][i, -1, 0, 0])] = (float(b["ret"][i]), float(b["boot"][i]))
    # step 5 (frame 6): rewards 1, 1, 1 -> done at step 7 stops after two steps (5, 6, 7 -> third reward is step 7's)
    assert seen[6] == pytest.approx((1 + 0.5 + 0.25, 0.0))
    assert seen[11] == pytest.approx((1 + 0.5 + 0.25, 0.125))


def test_only_transitions_with_n_following_steps_are_sampled():
    buf = PrioritizedStepBuffer(1, 32, n_step=3)
    _fill(buf, 10)
    valid = np.flatnonzero(buf._valid()[0])
    assert valid.max() == 10 - 1 - 3
    _fill(buf, 50)                              # wraps: only the newest C - stack + 1 ... steps stay valid
    t = buf.tabs[0][buf._valid()[0]]
    assert t.min() >= buf.pos[0] - buf.C + buf.stack and t.max() == buf.pos[0] - 1 - 3


def test_iqn_shapes_and_noise_toggle():
    net = IQNNet(10)
    x = torch.randint(0, 255, (2, 4, 84, 84), dtype=torch.uint8)
    q = net(x, torch.rand(2, 8))
    assert q.shape == (2, 8, 10)
    net.eval()
    a, b = net(x, torch.ones(2, 8) * 0.5), net(x, torch.ones(2, 8) * 0.5)
    assert torch.allclose(a, b)                 # no noise in eval mode


def test_one_update_runs_and_returns_priorities():
    agent = BTRAgent(10, device="cpu")
    buf = PrioritizedStepBuffer(2, 64, n_step=3)
    _fill(buf, 30)
    b = buf.sample(8, beta=0.4)
    loss, td = agent.learn(b)
    assert np.isfinite(loss) and td.shape == (8,)
    buf.update(b["e"], b["s"], td)
    assert agent.act(b["obs"][:3]).shape == (3,)


def test_ref10_action_set_matches_the_reference_buttons():
    from src.env import ACTION_SETS
    names, buttons = ACTION_SETS["ref10"]
    assert len(names) == len(buttons) == 10
    assert buttons[0] == () and buttons[1] == ("up",) and buttons[2] == ("up", "right")
    assert buttons[9] == ("B",) and names[9] == "jump"          # the DS jump button
    assert set(map(frozenset, buttons)) == set(map(frozenset, buttons)) and len(set(map(frozenset, buttons))) == 10


def test_goal_reward_is_off_by_default_and_passed_through():
    from scripts.train_btr import ref_env_kwargs
    assert ref_env_kwargs("ds3")["goal_reward"] == 0.0
    assert ref_env_kwargs("ds3", goal_reward=20.0)["goal_reward"] == 20.0
    assert ref_env_kwargs("ds1", goal_reward=20.0)["ram_coin_reward"] == 1.0


def test_ttc_pays_for_waypoints_and_the_star():
    from scripts.train_btr import ref_env_kwargs
    k = ref_env_kwargs("ds8")
    assert k["ram_coin_reward"] == 0.0 and k["dist_reward_mode"] == "waypoint" and k["star_reward"] == 100.0
    assert k["waypoint_reward"] == 1.0
    assert ref_env_kwargs("ds3")["ram_coin_reward"] == 1.0


def test_route_counts_waypoints_passed():
    import numpy as np
    from src.route import Route
    r = Route(np.array([[0, 0, 0], [0, 100, 0], [0, 200, 0], [0, 300, 0]], float))   # waypoints at arc 0, 100, 200, 300
    assert r.passed(0.0, 0.0) == 0 and r.passed(0.0, 99.0) == 0
    assert r.passed(0.0, 100.0) == 1 and r.passed(0.0, 250.0) == 2 and r.passed(150.0, 300.0) == 2


def test_ttc_fall_drop_and_stall_are_passed_to_the_env():
    from scripts.train_btr import ref_env_kwargs
    kw = ref_env_kwargs("ds8", stall_steps=600, fall_drop=1500.0)
    assert kw["stall_steps"] == 600 and kw["fall_drop"] == 1500.0
    assert ref_env_kwargs("ds8")["fall_drop"] == 0.0


def test_ttc_start_states_are_training_only_and_ds8_only():
    from scripts.train_btr import ref_env_kwargs
    assert "start_states" not in ref_env_kwargs("ds8")
    k = ref_env_kwargs("ds8", start_states=["a.dst"], full_start_prob=0.5)
    assert k["start_states"] == ["a.dst"] and k["full_start_prob"] == 0.5
    assert "start_states" not in ref_env_kwargs("ds3", start_states=["a.dst"])


def test_extended_agent_keeps_its_greedy_actions(tmp_path):
    import numpy as np
    import torch
    from src.btr import BTRAgent, checkpoint_actions
    from src.env import ACTION_SETS, ACTION_SET_EXTENDS
    torch.manual_seed(0)
    old = BTRAgent(10, device="cpu")
    old.save(str(tmp_path / "old.pt"))
    new = BTRAgent(13, device="cpu")
    new.load(str(tmp_path / "old.pt"), sources=ACTION_SET_EXTENDS["ref13"][1])
    new.save(str(tmp_path / "new.pt"))
    assert checkpoint_actions(str(tmp_path / "old.pt")) == 10 and checkpoint_actions(str(tmp_path / "new.pt")) == 13
    stacks = np.random.RandomState(1).randint(0, 255, (24, 4, 84, 84)).astype(np.uint8)
    a_old = old.act(stacks, noisy=False, fixed_taus=True)
    a_new = new.act(stacks, noisy=False, fixed_taus=True)
    assert (a_old == a_new).all() and (a_new < 10).all()
    assert len(ACTION_SETS["ref13"][0]) == 13 and ACTION_SETS["ref13"][1][:10] == ACTION_SETS["ref10"][1]


def test_extension_needs_sources(tmp_path):
    import pytest
    from src.btr import BTRAgent
    BTRAgent(10, device="cpu").save(str(tmp_path / "o.pt"))
    with pytest.raises(ValueError):
        BTRAgent(13, device="cpu").load(str(tmp_path / "o.pt"))
