# data/

Local, user-supplied inputs and generated replay buffers. Nothing in this
directory is committed except this README.

## Required inputs

- `Super Mario 64 DS (USA) (Rev 1).nds` -- legally dumped ROM.
- `Super Mario 64 DS (USA) (Rev 1).ds{1,2,3,8,9}` -- active starting savestates:
  - `ds1`: Peach's Secret Slide (Course 11)
  - `ds2`: Tall, Tall Mountain Slide (Course 5)
  - `ds3`: Cool, Cool Mountain Slide (Course 3)
  - `ds8`: Tick Tock Clock entry - Slow setting (Course 14)
  - `ds9`: Tick Tock Clock star jump test platform (Course 14)

> [!NOTE]
> Terminus savestates (`ds4`, `ds5`, `ds6`) were historically used to identify level finish
> coordinates. Those 3D coordinates are now permanently compiled into `src/ram_state.py`
> (`TRACK_GOALS`), so `ds4`, `ds5`, and `ds6` are not required at runtime.
> Savestates are excluded from Git via `.gitignore`. Use `python scripts/manage_states.py check`
> to inspect or `python scripts/manage_states.py unpack --bundle <path>` to restore them.

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
