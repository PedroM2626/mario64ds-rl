"""Route progress, action sets, start-state curriculum, recorder key mapping and
demo prefill -- all without the emulator or ROM."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.route import Route, resample_polyline


def _make_env(**kwargs):
    with patch("src.env.DeSmuME") as mock_emu_cls:
        mock_emu_cls.return_value = MagicMock()
        from src.env import Mario64DSEnv
        return Mario64DSEnv(rom_path="fake.nds", state_path="fake.ds8", **kwargs)


def _route_file(tmp_path, waypoints, **extra):
    p = tmp_path / "route.json"
    p.write_text(json.dumps({"schema": 1, "waypoints": waypoints, **extra}))
    return str(p)


def test_resample_spacing_and_endpoints():
    pts = [[0, 0, 0], [0, 0, 0], [100, 0, 0], [100, 100, 0]]  # includes a stationary sample
    w = resample_polyline(pts, 30.0)
    assert np.allclose(w[0], [0, 0, 0]) and np.allclose(w[-1], [100, 100, 0])
    assert np.all(np.linalg.norm(np.diff(w, axis=0), axis=1) <= 30.0 + 1e-9)


def test_progress_ratchets_within_window_and_corridor():
    r = Route([[0, 0, 0], [1000, 0, 0], [1000, 1000, 0]], corridor_radius=100, lookahead=300,
              lookbehind=100)
    assert r.length == pytest.approx(2000)
    assert r.progress([200, 50, 0], 0.0) == pytest.approx(200)
    assert r.progress([200, 150, 0], 0.0) is None            # outside the corridor
    # The far end is on the route but outside the lookahead window.
    assert r.progress([1000, 900, 0], 0.0) is None
    assert r.locate([1000, 900, 0]) == pytest.approx(1900)    # global search at reset


def test_action_sets_keep_legacy_indices():
    from src.env import ACTION_SETS
    legacy, compound = ACTION_SETS["legacy6"], ACTION_SETS["compound12"]
    assert compound[1][:6] == legacy[1] and len(compound[1]) == 12
    assert _make_env().action_space.n == 6
    env = _make_env(action_set="compound12")
    assert env.action_space.n == 12
    with pytest.raises(ValueError):
        _make_env(action_set="nope")


def test_compound_action_presses_all_buttons():
    from src.env import BUTTON_KEYS, keymask
    env = _make_env(action_set="compound12", use_ram_death=False)
    env.ram = MagicMock()
    env.ram.get_pos_3d.return_value = np.zeros(3)
    env._get_obs = lambda: np.full((84, 84, 1), 128, np.uint8)
    env.step(env.action_names.index("up+B (running jump)"))
    added = [c.args[0] for c in env.emu.input.keypad_add_key.call_args_list]
    removed = [c.args[0] for c in env.emu.input.keypad_rm_key.call_args_list]
    expected = sorted([keymask(BUTTON_KEYS["up"]), keymask(BUTTON_KEYS["B"])])
    assert sorted(added) == expected and sorted(removed) == expected


def test_route_reward_pays_only_new_progress(tmp_path):
    route = _route_file(tmp_path, [[0, 0, 0], [1000, 0, 0]], corridor_radius=100, lookahead=400)
    env = _make_env(dist_reward_mode="route", route_path=route, route_total_reward=1000.0,
                    flow_weight=0.0, step_penalty=0.0, use_ram_death=False, track_id="ds8")
    env.ram = MagicMock()
    env.ram.get_star_count.return_value = 0
    env._get_obs = lambda: np.full((84, 84, 1), 128, np.uint8)
    env.ram.get_pos_3d.return_value = np.array([100.0, 0, 0])
    env.reset()
    assert env.route_best_s == pytest.approx(100)
    rewards = []
    for x in (300.0, 200.0, 300.0, 400.0):  # forward, back, back to record, new record
        env.ram.get_pos_3d.return_value = np.array([x, 0, 0])
        rewards.append(env.step(0)[1])
    assert rewards == pytest.approx([200.0, 0.0, 0.0, 100.0])


def test_route_mode_requires_route():
    with pytest.raises(ValueError):
        _make_env(dist_reward_mode="route")


def test_start_state_curriculum_mixes_full_starts():
    env = _make_env(start_states=["a.dst", "b.dst"], full_start_prob=0.25, track_id="ds8")
    env.ram = MagicMock()
    env.ram.get_pos_3d.return_value = np.zeros(3)
    env._get_obs = lambda: np.zeros((84, 84, 1), np.uint8)
    seen = []
    env.reset(seed=0)
    for _ in range(400):
        env.reset()
        seen.append(env.current_start)
    frac_full = seen.count("fake.ds8") / len(seen)
    assert 0.15 < frac_full < 0.35 and {"a.dst", "b.dst"} <= set(seen)
    env.emu.savestate.load_file.assert_called_with(env.current_start)


def test_info_reports_flow_reward():
    env = _make_env(use_ram_death=False)
    env.ram = MagicMock()
    env.ram.get_pos_3d.return_value = np.zeros(3)
    info = env._get_info()
    assert "flow_reward" in info and "died" in info


def test_recorder_key_mapping():
    from scripts.record_human import keys_to_action
    from src.env import ACTION_SETS
    names, buttons = ACTION_SETS["compound12"]
    def act(*held):
        i, exact = keys_to_action(held, buttons)
        return names[i], exact
    assert act() == ("noop", True)
    assert act("up", "B") == ("up+B (running jump)", True)
    assert act("up", "Y") == ("up+Y (run)", True)
    assert act("left", "right") == ("noop", False)            # opposites cancel
    assert act("up", "Y", "B") == ("up+B (running jump)", False)  # Y dropped, nothing added
    i, exact = keys_to_action(("up", "B"), ACTION_SETS["legacy6"][1])
    assert ACTION_SETS["legacy6"][0][i] in ("up (accelerate)", "B (jump)") and not exact


def test_prefill_groups_by_n_envs():
    from src.demos import prefill_replay_buffer
    buf = MagicMock()
    buf.n_envs = 4
    model = MagicMock(replay_buffer=buf)
    t = {"obs": np.zeros((1, 4, 84, 84), np.uint8), "next_obs": np.zeros((1, 4, 84, 84), np.uint8),
         "action": np.array([[3]]), "reward": np.array([1.0]), "done": np.array([False]), "info": {}}
    assert prefill_replay_buffer(model, [t] * 10) == 8
    assert buf.add.call_count == 2
    obs = buf.add.call_args.args[0]
    assert obs.shape == (4, 4, 84, 84)


def test_per_frame_masks_press_and_release_on_the_right_frames():
    from src.env import BUTTON_KEYS, buttons_to_mask, keymask
    env = _make_env(use_ram_death=False)
    env.ram = MagicMock()
    env.ram.get_pos_3d.return_value = np.zeros(3)
    env._get_obs = lambda: np.full((84, 84, 1), 128, np.uint8)
    events = []
    env.emu.input.keypad_add_key.side_effect = lambda k: events.append(("add", k))
    env.emu.input.keypad_rm_key.side_effect = lambda k: events.append(("rm", k))
    env.emu.cycle.side_effect = lambda: events.append(("cycle",))
    up, ub = buttons_to_mask(["up"]), buttons_to_mask(["up", "B"])
    env.step(np.array([up, ub, ub, up]))  # Up, then B pressed one frame later
    U, B = keymask(BUTTON_KEYS["up"]), keymask(BUTTON_KEYS["B"])
    assert events == [("add", U), ("cycle",), ("add", B), ("cycle",), ("cycle",), ("rm", B),
                      ("cycle",), ("rm", U)]
    assert env.last_frame_masks == [up, ub, ub, up]
    with pytest.raises(ValueError):
        env.step(np.array([up, up]))


def test_live_callable_is_sampled_every_frame():
    env = _make_env(use_ram_death=False)
    env.ram = MagicMock()
    env.ram.get_pos_3d.return_value = np.zeros(3)
    env._get_obs = lambda: np.full((84, 84, 1), 128, np.uint8)
    calls = []
    env.step(lambda: calls.append(1) or 0)
    assert len(calls) == env.frameskip


def test_label_window_exact_only_when_constant():
    from scripts.record_human import label_window
    from src.env import ACTION_SETS, buttons_to_mask
    names, buttons = ACTION_SETS["compound12"]
    up, ub, r = buttons_to_mask(["up"]), buttons_to_mask(["up", "B"]), buttons_to_mask(["R"])
    a, exact = label_window([ub] * 4, buttons)
    assert names[a] == "up+B (running jump)" and exact
    assert label_window([up, ub, ub, ub], buttons)[1] is False   # B pressed mid-window
    assert label_window([r] * 4, buttons) == (0, False)           # crouch is not an action


def test_prefill_skips_inexact_decisions():
    from src.demos import prefill_replay_buffer
    buf = MagicMock()
    buf.n_envs = 2
    t = {"obs": np.zeros((1, 4, 84, 84), np.uint8), "next_obs": np.zeros((1, 4, 84, 84), np.uint8),
         "action": np.array([[0]]), "reward": np.array([0.0]), "done": np.array([False]), "info": {}}
    trs = [dict(t, exact=e) for e in (True, False, True, False, False)]
    assert prefill_replay_buffer(MagicMock(replay_buffer=buf), trs) == 2
