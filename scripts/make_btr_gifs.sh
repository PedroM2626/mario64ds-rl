#!/bin/bash
# One finishing episode per slide of the final BTR agent -> mp4 -> small GIF (README).
cd "$(dirname "$0")/.."
M=${1:-models/btr_goal20/step_1000005.pt}
mkdir -p videos/btr
for t in ds1 ds2 ds3; do
  .venv/Scripts/python.exe scripts/record_btr.py --model $M --track $t --attempts 10 --fps 15 --out videos/btr/btr_$t.mp4 2>&1 | grep -a "decisions, coins" | tee -a logs/make_btr_gifs.log
done
ls -la videos/btr
