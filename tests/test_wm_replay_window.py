"""Regression tests for the allocation-light replay window and schema header."""

import numpy as np

from src.wm_replay import EpisodeBuffer, STACK, _pad_frames, _window


def _reference_window(frames, i):
    padded = _pad_frames(frames)
    ref = padded[i:i + STACK]
    if len(ref) < STACK:
        ref = np.concatenate([ref, np.zeros((STACK - len(ref),) + ref.shape[1:],
                                            ref.dtype)])
    return ref


def test_window_matches_padded_slice():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, size=(7, 84, 84), dtype=np.uint8)
    for i in range(0, len(frames) + 2):
        got = _window(frames, i)
        ref = _reference_window(frames, i)
        assert got.shape == (STACK, 84, 84)
        assert np.array_equal(got, ref), i


def test_buffer_schema_and_dedupe(tmp_path):
    frames = np.zeros((5, 84, 84), dtype=np.uint8)
    actions = np.arange(5, dtype=np.int64)
    rewards = np.ones(5, dtype=np.float32)
    continues = np.ones(5, dtype=np.float32)

    buf = EpisodeBuffer(meta={"source": "test"})
    buf.add(frames, actions, rewards, continues, source="ppo")
    buf.add(frames, actions, rewards, continues, source="ppo")
    assert len(buf) == 2
    assert buf.total_steps == 10

    removed = buf.dedupe()
    assert removed == 1
    assert len(buf) == 1
    assert buf.total_steps == 5

    path = tmp_path / "buffer.pkl"
    buf.save(str(path))
    loaded = EpisodeBuffer()
    loaded.load(str(path))
    assert loaded.version == 2
    assert loaded.meta["source"] == "test"
    assert loaded.total_steps == 5
    assert len(loaded) == 1
