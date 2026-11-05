#!/usr/bin/env bash
# PPO on ds3 with the human run (config/routes/ds3_human_v1.json) as the shaping signal:
# the distance reward is the ratcheted progress along the route (+500 for all of it), the finish
# pays +200, training resets to the run's checkpoint savestates (25% from the real start).
# Starts from the behaviour clone of the run (the DAgger script's first model).
# Usage: bash scripts/run_ds3_human_ppo.sh [run_id] [timesteps] [start_model]
set -u
cd "$(dirname "$0")/.."
RUN=${1:-ds3_human_ppo}
STEPS=${2:-1000000}
START=${3:-models/ds3_dagger.zip}
PY=.venv/Scripts/python.exe

STARTS=$(ls data/curriculum/ds3_human_v1/*.dst | tr '\n' ',' | sed 's/,$//')
mkdir -p logs

(
  D=models/${RUN}_p1/checkpoints; seen=""
  while true; do
    for f in $D/*.zip; do
      [ -f "$f" ] || continue
      case " $seen " in *" $f "*) continue;; esac
      sleep 30; seen="$seen $f"; echo "== $(basename "$f")"
      echo -n "from start: "
      $PY scripts/eval_from_states.py --algo ppo --model "$f" --states "data/Super Mario 64 DS (USA) (Rev 1).ds3" \
        --max-steps 1200 --no-black-screen-death --respawn-jump 2000 2>/dev/null | grep -aE '^\{' | cut -c1-220
    done
    grep -aq "Curriculum completed\|Traceback" logs/train_$RUN.log 2>/dev/null && break
    sleep 60
  done
  echo "watcher done"
) > logs/watch_$RUN.log 2>&1 &

PYTHONIOENCODING=utf-8 $PY -u -m src.train_curriculum --run-id "$RUN" --start-model "$START" \
  --action-set compound12 --states ds3,ds3,ds3,ds3 --n-envs 4 --start-states "$STARTS" --full-start-prob 0.25 \
  --phases "$STEPS:2e-5" --target-kl 0.02 --clip-range 0.1 --ent-coef 0.0 --value-warmup-steps 100000 \
  --flow-weight 0 --eval-flow-weight 0 --eval-freq 100000 --max-steps 1200 \
  --route config/routes/ds3_human_v1.json --route-reward 500 --goal-reward 200 --no-black-screen-death --respawn-jump 2000 \
  --checkpoint-every 100000 --results-csv results/results_$RUN.csv > logs/train_$RUN.log 2>&1
wait
