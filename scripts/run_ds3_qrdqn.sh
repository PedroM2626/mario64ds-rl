#!/usr/bin/env bash
# QR-DQN + IMPALA on ds3 from the route found by search, with the route as a replay demo.
# Usage: bash scripts/run_ds3_qrdqn.sh [run_id] [timesteps]
# Optional: WAIT_FOR=<log containing 'Curriculum completed'> waits for another training first.
# Needs (build from config/routes/ds3_search_route_v1.json with --search):
#        data/curriculum/ds3_route/ (scripts/make_route_states.py --upto 720 --every 15) and
#        data/curriculum/ds3_route_demo/ (scripts/route_to_demo.py).
set -u
cd "$(dirname "$0")/.."
RUN=${1:-qrdqn_ds3_route}
STEPS=${2:-1000000}
PY=.venv/Scripts/python.exe
GOAL=6432,-11263,-11499

if [ -n "${WAIT_FOR:-}" ]; then
  until grep -aq "Curriculum completed\|Traceback" "$WAIT_FOR" 2>/dev/null; do sleep 30; done
fi
until [ "$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe' or Name='Dolphin.exe'\" | Where-Object { \$_.CommandLine -match 'smg_rl' }).Count" | tr -d '\r')" = "0" ]; do sleep 30; done

ROUTE=$(ls data/curriculum/ds3_route/route-*.dst | tr '\n' ',' | sed 's/,$//')
mkdir -p logs

# milestone watcher: arrow-states (decisions 300-345) and the real start, per checkpoint
(
  D=models/$RUN/checkpoints; seen=""
  while true; do
    for f in $D/*.zip; do
      [ -f "$f" ] || continue
      case " $seen " in *" $f "*) continue;; esac
      sleep 30; seen="$seen $f"; echo "== $(basename "$f")"
      echo -n "arrow-states: "
      $PY scripts/eval_from_states.py --algo qrdqn --model "$f" --states "data/curriculum/ds3_route/route-03[0-4][05].dst" \
        --max-steps 400 --goal-pos $GOAL --goal-radius 600 2>/dev/null | grep -aE '^\{' | $PY -c "
import sys,json
rows=[json.loads(l) for l in sys.stdin]
print(' | '.join(f\"{r['state'][-8:-4]}: {r['end'][:4]} +{r['coins_gained']}c d={r['min_dist']:.0f}\" for r in rows))"
      echo -n "from start:   "
      $PY scripts/eval_from_states.py --algo qrdqn --model "$f" --states "data/Super Mario 64 DS (USA) (Rev 1).ds3" \
        --max-steps 1200 --goal-pos $GOAL --goal-radius 600 2>/dev/null | grep -aE '^\{' | cut -c1-200
    done
    grep -aq "Training complete\|Traceback" logs/train_$RUN.log 2>/dev/null && break
    sleep 60
  done
  echo "watcher done"
) > logs/watch_$RUN.log 2>&1 &

PYTHONIOENCODING=utf-8 $PY -u -m src.train_qrdqn --run-id "$RUN" --states ds3 --features impala --n-envs 4 \
  --timesteps "$STEPS" --buffer-size 100000 --exploration-fraction 0.3 --exploration-final-eps 0.05 \
  --action-set compound12 --flow-weight 0 --no-dist-reward --coin-reward 1.0 \
  --goal-reward 500 --goal-pos $GOAL --goal-radius 600 --max-steps 1200 \
  --start-states "$ROUTE" --full-start-prob 0.25 \
  --demo data/curriculum/ds3_route_demo/attempt-000001 --demo-copies 10 \
  --demo-refresh-every 10000 --demo-refresh-copies 1 --hold-max 16 \
  --eval-freq 25000 --checkpoint-every 100000 --device cuda > logs/train_$RUN.log 2>&1
wait
