#!/usr/bin/env bash
# PPO fine-tune of the behaviour-cloned ds3 policy (scripts/clone_demo.py -> models/ds3_bc_c12.zip).
# The clone passes the red arrow but dies at decision ~671 of the route and is not robust;
# RL from the route savestates (extra weight on decisions 600-690) should fix and robustify it.
# Usage: bash scripts/run_ds3_finetune.sh [run_id] [timesteps]
set -u
cd "$(dirname "$0")/.."
RUN=${1:-ds3_finetune}
STEPS=${2:-800000}
PY=.venv/Scripts/python.exe
GOAL=6432,-11263,-11499

until [ "$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe' or Name='Dolphin.exe'\" | Where-Object { \$_.CommandLine -match 'smg_rl' }).Count" | tr -d '\r')" = "0" ]; do sleep 30; done

ROUTE=$(ls data/curriculum/ds3_route/route-*.dst | tr '\n' ',' | sed 's/,$//')
# two extra copies of the states around the clone's failure (route decisions 600-690)
HARD=$(ls data/curriculum/ds3_route/route-06[0-9]0.dst data/curriculum/ds3_route/route-06[0-9]5.dst 2>/dev/null | grep -E 'route-0(600|615|630|645|660|675|690)' | tr '\n' ',' | sed 's/,$//')
STARTS="$ROUTE,$HARD,$HARD"
mkdir -p logs

(
  D=models/${RUN}_p1/checkpoints; seen=""
  while true; do
    for f in $D/*.zip; do
      [ -f "$f" ] || continue
      case " $seen " in *" $f "*) continue;; esac
      sleep 30; seen="$seen $f"; echo "== $(basename "$f")"
      echo -n "arrow-states: "
      $PY scripts/eval_from_states.py --algo ppo --model "$f" --states "data/curriculum/ds3_route/route-03[0-4][05].dst" \
        --max-steps 400 --goal-pos $GOAL --goal-radius 600 2>/dev/null | grep -aE '^\{' | $PY -c "
import sys,json
rows=[json.loads(l) for l in sys.stdin]
print(' | '.join(f\"{r['state'][-8:-4]}: {r['end'][:4]} +{r['coins_gained']}c d={r['min_dist']:.0f}\" for r in rows))"
      echo -n "hard spot:    "
      $PY scripts/eval_from_states.py --algo ppo --model "$f" --states "data/curriculum/ds3_route/route-06[3-7][05].dst" \
        --max-steps 200 --goal-pos $GOAL --goal-radius 600 2>/dev/null | grep -aE '^\{' | $PY -c "
import sys,json
rows=[json.loads(l) for l in sys.stdin]
print(' | '.join(f\"{r['state'][-8:-4]}: {r['end'][:4]} d={r['min_dist']:.0f}\" for r in rows))"
      echo -n "from start:   "
      $PY scripts/eval_from_states.py --algo ppo --model "$f" --states "data/Super Mario 64 DS (USA) (Rev 1).ds3" \
        --max-steps 1200 --goal-pos $GOAL --goal-radius 600 2>/dev/null | grep -aE '^\{' | cut -c1-200
    done
    grep -aq "Curriculum completed\|Traceback" logs/train_$RUN.log 2>/dev/null && break
    sleep 60
  done
  echo "watcher done"
) > logs/watch_$RUN.log 2>&1 &

PYTHONIOENCODING=utf-8 $PY -u -m src.train_curriculum --run-id "$RUN" --start-model models/ds3_bc_c12.zip \
  --action-set compound12 --states ds3,ds3,ds3,ds3 --n-envs 4 --start-states "$STARTS" --full-start-prob 0.25 \
  --phases "$STEPS:3e-5" --flow-weight 0 --eval-flow-weight 0 --eval-freq 100000 --max-steps 1200 \
  --no-dist-reward --coin-reward 1.0 --goal-reward 500 --goal-pos $GOAL --goal-radius 600 \
  --checkpoint-every 100000 --results-csv results/results_$RUN.csv > logs/train_$RUN.log 2>&1
wait
