#!/bin/bash
# Regenerates every evaluation quoted in the README (20 episodes per slide) and the perturbed-start one.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
run() { $PY scripts/eval_btr.py --model "$1" --tracks ds1,ds2,ds3 --episodes 20 --out "$2" "${@:3}" > "logs/$(basename $2 .json).log" 2>&1; }
run models/btr_goal20/step_1000005.pt results/btr_goal20_eval_1M_x20.json &
run models/btr_goal20/step_1000005.pt results/btr_goal20_eval_1M_perturbed15_x20.json --start-random 15 &
run models/btr_goal20/step_750000.pt results/btr_goal20_eval_750k_x20.json &
wait
run models/btr_goal20/step_500010.pt results/btr_goal20_eval_500k_x20.json &
run models/btr_ref10/step_2500005.pt results/btr_ref10_eval_2500005_x20.json &
run models/btr_ref10/step_4500000.pt results/btr_ref10_eval_4500000_x20.json &
wait
echo ALL DONE
