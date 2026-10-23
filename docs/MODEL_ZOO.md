# Model zoo and large assets

The code is in Git; the large artifacts are not. A clean clone therefore does
**not** contain the flagship checkpoints that the README commands reference.
This page is the contract for those files: their role, size and SHA256, plus how
to verify them once they are downloaded.

## Verify

```bash
python tools/verify_assets.py             # report present/missing/hash mismatch
python tools/verify_assets.py --require   # fail if a required asset is missing
```

The manifest is [models/ASSETS.json](../models/ASSETS.json). Paths are relative
to the repository root. The SHA256 values were measured locally on 2026-10-23;
publish the same bytes as GitHub Release/LFS assets so consumers can check them.

## Required for the headline results

| Asset | Role | Size | SHA256 |
|---|---|---:|---|
| `models/curriculum_flow25_r3_best.zip` | PPO + NatureCNN, flow x2.5 (main benchmark) | 20,590,768 | `c763cbf3...f0a86df6` |
| `models/wm_safe_q7_wm.pt` | RSSM bundle for expert-free CEM-MPC | 8,421,900 | `7f9e5bc3...651eb7e7` |
| `models/wm_mario64ds_wm.pt` | RSSM bundle for distillation/DAgger | 4,753,520 | `68d73ecd...cb80205d` |
| `models/wm_mario64ds_mem.pt` | Memoryless distilled head (3/3 tracks) | 3,755,738 | `203d992b...20497f66` |

The full hashes are in `models/ASSETS.json`. Optional assets are the exact-state
bundle (`models/state_wm.pt`) and the base replay buffer (`data/wm_buffer.pkl`).

## If you do not have the assets

1. Train them with the commands in the README (the PPO/curriculum path takes
   hours; the world-model path takes minutes once the data exists), or
2. download the release assets and run the verify command, or
3. collect a fresh buffer with `src.collect_data` and fit a new world model.

`data/` itself is intentionally ignored: ROMs and savestates are user-supplied
and the buffers are reproducible. See [data/README.md](../data/README.md).

## Videos and images

`videos/` is tracked but large (193 MB, no LFS at the moment). Prefer release
assets/LFS for new recordings rather than growing the Git history. See
[NOTICE.md](../NOTICE.md) for the asset/licence scope.
