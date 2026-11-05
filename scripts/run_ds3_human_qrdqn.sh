#!/usr/bin/env bash
# QR-DQN + IMPALA on ds3 with the human run as the shaping signal and as a replay demo
# (the value-based counterpart of scripts/run_ds3_human_ppo.sh): route-progress reward,
# +200 at the finish, resets to the run's checkpoint savestates (25% from the real start).
# Usage: bash scripts/run_ds3_human_qrdqn.sh [run_id] [timesteps] [attempt_dir]
set -u
cd "$(dirname "$0")/.."
RUN=${1:-qrdqn_ds3_human}
STEPS=${2:-1000000}
DEMO=${3:-runs/human-demonstrations/20261105-134615/completed/attempt-000001}
PY=.venv/Scripts/python.exe
mkdir -p logs

(
  D=models/$RUN/checkpoints; seen=""
  while true; do
    for f in $D/*.zip; do
      [ -f "$f" ] || continue
      case " $seen " in *" $f "*) continue;; esac
      sleep 30; seen="$seen $f"; echo "== $(basename "$f")"
      echo -n "from start: "
      $PY scripts/eval_from_states.py --algo qrdqn --model "$f" --states "data/Super Mario 64 DS (USA) (Rev 1).ds3" \
        --max-steps 1200 --no-black-screen-death --respawn-jump 2000 2>/dev/null | grep -aE '^\{' | cut -c1-220
    done
    grep -aq "Training complete\|Traceback" logs/train_$RUN.log 2>/dev/null && break
    sleep 60
  done
  echo "watcher done"
) > logs/watch_$RUN.log 2>&1 &

PYTHONIOENCODING=utf-8 $PY -u -m src.train_qrdqn --run-id "$RUN" --states ds3 --features impala --n-envs 4 \
  --timesteps "$STEPS" --buffer-size 100000 --exploration-fraction 0.3 --exploration-final-eps 0.05 \
  --action-set compound12 --flow-weight 0 --dist-reward --dist-mode route \
  --route config/routes/ds3_human_v1.json --route-total-reward 500 --goal-reward 200 --max-steps 1200 \
  --start-states route --full-start-prob 0.25 --no-black-screen-death --respawn-jump 2000 \
  --demo "$DEMO" --demo-copies 10 --hold-max 4 \
  --eval-freq 25000 --checkpoint-every 100000 --device cuda > logs/train_$RUN.log 2>&1
wait
