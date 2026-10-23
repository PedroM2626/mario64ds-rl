"""Offline tests for the evaluation summary/provenance helpers.

The important property is that three identical deterministic rollouts do not
count as three independent samples. These tests do not need an emulator.
"""

from src.eval import summarize_rows, wilson_ci


def _row(track, reward, steps, survived):
    return {"savestate": track, "episode": 0, "reward": reward,
            "steps": steps, "survived": survived}


def test_wilson_interval_bounds():
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0
    assert 0.2 < hi < 0.3
    lo, hi = wilson_ci(10, 10)
    assert 0.7 < lo < 0.8
    assert hi == 1.0
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_summary_separates_episodes_from_unique_trajectories():
    rows = [_row("ds1", 10.0, 1350, True) for _ in range(3)]
    rows += [_row("ds3", -5.0, 100, False) for _ in range(3)]
    summary = summarize_rows(rows)
    assert summary["tracks"]["ds1"]["n"] == 3
    assert summary["tracks"]["ds1"]["n_unique"] == 1
    assert summary["tracks"]["ds1"]["survived"] == 3
    assert summary["tracks"]["ds3"]["n_unique"] == 1
    assert summary["overall"]["n"] == 6
    assert summary["overall"]["n_unique"] == 2
    assert summary["overall"]["survived"] == 3


def test_summary_counts_distinct_rollouts():
    rows = [
        _row("ds1", 1.0, 10, True),
        _row("ds1", 2.0, 20, False),
        _row("ds1", 2.0, 20, False),
    ]
    summary = summarize_rows(rows)
    assert summary["tracks"]["ds1"]["n_unique"] == 2
    assert summary["tracks"]["ds1"]["survived"] == 1
