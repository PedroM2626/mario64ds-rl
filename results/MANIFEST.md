# Results manifest

This directory holds the consolidated experiment artifacts referenced by the
README. The purpose of this file is to make the **provenance and the known gaps**
explicit, because some tables were assembled across several runs and the code
has changed since.

| File | What it is | Known caveat |
|---|---|---|
| `results_grid_100k.csv` | 18 configurations (PPO/Rainbow/QR-DQN x Nature/IMPALA x 3 seeds) at 100k steps | 108 rows = 36 unique trajectories: every per-track "episode" triple is the same deterministic rollout repeated 3x. No `flow_weight` column: the historical run predates the current defaults. |
| `results_grid_500k.csv` | PPO/Rainbow + Nature at 500k | Only `ds1` and `ds3` rows. The `ds2` 440/450 cell in the README is not backed by this CSV. |
| `results_curriculum_r1.csv`, `results_curriculum_r2.csv` | Curriculum phase summaries | Legacy column name `pista` (current code writes `track`). |
| `results_ds123_2m.csv` | Pre-curriculum 2M three-track experiment | Historical; single best checkpoint. |
| `device_divergence.json` | CPU vs CUDA action-sequence study | Does not record torch/CUDA versions or the model hash; current `src/eval.py` JSON output does. |
| `supervisor_parity.json` | Python vs Rust orchestration parity, budget kill, save integrity | Regenerate with `python tools/parity_supervisor.py --steps 40`. |
| `step_profile.json` | Per-stage milliseconds for one env step | Regenerate with `python tools/profile_step.py --steps 200`. |

## Regeneration

```bash
python scripts/run_grid_100k.py --flow-weight 1.0 --savestate-idxs 0,2
python -m src.eval --algo ppo --model models/curriculum_flow25_r3_best.zip \
    --n-episodes 3 --deterministic --seed 0 --device cpu --flow-weight 1.0
python tools/parity_supervisor.py --steps 40
python tools/profile_step.py --steps 200
```

The evaluation CSV now carries a `device` and `flow_weight` column and a sidecar
JSON with `n_unique`, Wilson confidence intervals and the model SHA256. Use the
JSON, not a mean over three identical deterministic rows, when quoting a result.
