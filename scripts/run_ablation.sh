#!/bin/bash
# Algorithm / action-set ablation, one run at a time, 60 minutes of training each, then 20 greedy episodes per slide.
#   control : BTR-style, ref10 actions, coin + finish reward (same recipe as the final agent, trained from scratch)
#   ppo     : PPO,       ref10 actions, same reward and episode rules
#   legacy6 : BTR-style, the original 6 actions, same reward and episode rules
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
MIN=${MIN:-60}
mem_ok() {
  free_kb=$(powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory" | tr -d '\r')
  echo "free memory: $((free_kb / 1048576)) GB" | tee -a logs/ablation.log
  [ "$free_kb" -gt 8388608 ] || { echo "ABORT: less than 8 GB free" | tee -a logs/ablation.log; exit 1; }
  busy=$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'train_btr|train_ppo_ref' }).Count" | tr -d '\r')
  [ "$busy" = "0" ] || { echo "ABORT: another training is running" | tee -a logs/ablation.log; exit 1; }
}
for cfg in control ppo legacy6; do
  mem_ok
  echo "== $cfg: start $(date)" | tee -a logs/ablation.log
  case $cfg in
    control) $PY -u scripts/train_btr.py --run abl_control --goal-reward 20 --max-minutes $MIN --action-set ref10 > logs/train_abl_control.log 2>&1
             $PY scripts/eval_btr.py --model models/abl_control/final.pt --tracks ds1,ds2,ds3 --episodes 20 --action-set ref10 --out results/abl_control_eval_x20.json > logs/eval_abl_control.log 2>&1 ;;
    ppo)     $PY -u scripts/train_ppo_ref.py --run abl_ppo --goal-reward 20 --max-minutes $MIN --action-set ref10 > logs/train_abl_ppo.log 2>&1
             $PY scripts/eval_btr.py --model models/abl_ppo/final.zip --algo ppo --tracks ds1,ds2,ds3 --episodes 20 --action-set ref10 --out results/abl_ppo_eval_x20.json > logs/eval_abl_ppo.log 2>&1 ;;
    legacy6) $PY -u scripts/train_btr.py --run abl_legacy6 --goal-reward 20 --max-minutes $MIN --action-set legacy6 > logs/train_abl_legacy6.log 2>&1
             $PY scripts/eval_btr.py --model models/abl_legacy6/final.pt --tracks ds1,ds2,ds3 --episodes 20 --action-set legacy6 --out results/abl_legacy6_eval_x20.json > logs/eval_abl_legacy6.log 2>&1 ;;
  esac
  echo "== $cfg: done $(date)" | tee -a logs/ablation.log
done
echo ABLATION DONE | tee -a logs/ablation.log
