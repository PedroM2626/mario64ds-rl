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

## Tick Tock Clock BTR stages (2026-11-09 snapshot)

| Files | Meaning | Provenance and limitations |
|---|---|---|
| `btr_ttc_wp_train.csv`, `btr_ttc_wp2_train.csv` through `btr_ttc_wp6_train.csv` | Periodic training statistics for six stages in one weight-transfer lineage | `steps` sums decisions over 15 environments; `minutes` and steps are the last periodic sample, not exact final counts. `ds8_coins` is mean episode return over up to 30 recent completed episodes, not a coin counter. |
| `ttc_probe_btr_ttc_wp.json`, `ttc_probe_btr_ttc_wp2.json` through `ttc_probe_btr_ttc_wp6.json` | Ten CPU greedy episodes for each stage's final checkpoint | `noisy=False` disables noisy layers, but IQN still samples quantile fractions. Probe JSONs do not themselves record seed, horizon, device, model hash or runtime versions. The documented commands establish the CPU instrument and the 3,000 / 5,000-decision horizons. |
| `ttc_btr_stages.json` | Consolidated, derived stage metrics, per-episode route-crossing counts, source hashes, checkpoint hashes and the observed stage-6 runtime | Text-source SHA256 values normalize line endings to LF; checkpoint SHA256 values use raw binary bytes. Not a new experiment. Checkpoint hashes were measured when documenting stage 6, not recorded by the historical probes. Earlier stages' complete runtimes were not preserved. The observed local runtime differs from `requirements.txt` and `pyproject.toml`. |
| `btr_ttc_route_train.csv`, `btr_ttc_route2_train.csv`, `btr_ttc_nocoin_train.csv` and corresponding probe JSONs | Earlier length-scaled route-reward experiments | Different reward recipes from the waypoint stages; returns cannot be compared directly. |

The route is `config/routes/ttc_human_v1.json`: 371 positions, 53,245.7634
arc-length units. Counts use `Route.passed(0, route_s)`; the initial point at
zero does not pay, so at most 370 new crossings are available. The denominator
371 in the README names the number of positions, while route percentages use
arc length. Rounded probe `route_s` values are used for historical counts;
near a waypoint boundary, stored decimal rounding could affect a count.

The final stage-1 probe JSON records **6 deaths and 4 horizon endings**; an
older conversational summary stated 5 and 5. The README follows the JSON.
No ten-episode probe establishes a causal explanation for the observed
obstacles, and increasing the episode horizon was not an isolated ablation.
Star outcomes are the environment's geometry/RAM success detector, not a
separate manual audit of collection animation.

Each continuation loads policy weights and resets optimizer, replay, target
initialization, update counter and replay-importance progress. It is not an
uninterrupted optimizer/replay continuation. Stage 7 is an active continuation
through approximately 19:00 America/Sao_Paulo on 2026-11-09, and its outcomes
are pending in this snapshot. TTC model files, videos, verbose logs and local
time-budget wrappers are not part of these result commits.

To re-derive the waypoint counts from a probe (with the project interpreter):

```python
import json
from src.route import Route
route = Route.load("config/routes/ttc_human_v1.json")
with open("results/ttc_probe_btr_ttc_wp6.json", encoding="utf-8") as source:
    rows = json.load(source)["rows"]
print([route.passed(0, row["route_s"]) for row in rows])
```
