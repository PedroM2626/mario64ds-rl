#!/bin/bash
# Control with the original recipe: coin-only (no finish reward), --steps 30M schedule, 60 minutes, then 20 greedy episodes per slide.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
echo "== orig: start $(date)" | tee -a logs/ablation.log
$PY -u scripts/train_btr.py --run abl_orig --steps 30000000 --max-minutes 60 > logs/train_abl_orig.log 2>&1
$PY scripts/eval_btr.py --model models/abl_orig/final.pt --tracks ds1,ds2,ds3 --episodes 20 --out results/abl_orig_eval_x20.json > logs/eval_abl_orig.log 2>&1
echo "== orig: done $(date)" | tee -a logs/ablation.log
