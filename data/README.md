# data/

Local, user-supplied inputs and generated replay buffers. Nothing in this
directory is committed except this README.

## Required inputs

- `Super Mario 64 DS (USA) (Rev 1).nds` -- legally dumped ROM.
- `Super Mario 64 DS (USA) (Rev 1).ds{1,2,3}` -- savestates for the three slide
  tracks (`ds1` Course 3, `ds2` Peach Slide, `ds3` Monkey Slide).

The RAM reader (`src/ram_state.py`) is verified for **USA Rev 1** only. A
different ROM revision may have a different struct address; the boot-time check
warns but does not fail, so verify it before trusting state-based results.

## Replay buffers

The world-model buffers are Python pickles produced by `src/collect_data.py` and
combined by `scripts/merge_buffers.py`. As of buffer schema version 2 the pickle
carries `version` and `meta` fields; older files still load (version 0). Each
episode stores single grayscale frames plus actions, rewards, continues and
optional probe metadata (`probe_at`, `probe_id`, `branch_cont`).

The base buffer (`wm_buffer.pkl`) is 35k real transitions / 48 episodes / 248 MB.
The probe/merge variants grow much larger because each probe branch repeats the
shared prefix; use deduplication when merging:

```bash
python scripts/merge_buffers.py --out data/wm_buffer_all.pkl data/wm_buffer.pkl data/wm_probe_hold.pkl
```

The merge script removes exact duplicate episodes by default; pass
`--keep-duplicates` if a deliberate repetition is required. See
[models/ASSETS.json](../models/ASSETS.json) for the hash of the base buffer.
