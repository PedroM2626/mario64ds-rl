"""Sequential execution queue: waits for Rainbow 500k (background process) to finish,
then executes PPO 3-way continuation up to 2M total steps and evaluates across all 3 tracks.

Termination signal for Rainbow: models/grid_rainbow_nature_s0_500k_final.pth
(saved strictly upon training completion). Once detected, waits 120s to ensure
process termination and release of emulator resources, then launches PPO 2M and evaluations.

Usage:
    python queue_ds123_2m.py
    Start-Process ... python queue_ds123_2m.py   (background daemon)
"""

import csv
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

RAINBOW_FINAL = os.path.join(BASE, "models", "grid_rainbow_nature_s0_500k_final.pth")
PPO2M_LOG = os.path.join(BASE, "logs", "ppo_ds123_2m.log")
QUEUE_LOG = os.path.join(BASE, "logs", "queue_ds123_2m.log")
RESULTS = os.path.join(BASE, "results_ds123_2m.csv")

MAX_WAIT_SEC = 14 * 3600


def log(msg):
    with open(QUEUE_LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)


def wait_for_rainbow():
    # Detection by mtime: final.pth is only valid if created AFTER queue start time
    # (prevents false positives from stale artifacts from prior runs).
    t0 = time.time()
    while True:
        fresh = (os.path.exists(RAINBOW_FINAL)
                 and os.path.getmtime(RAINBOW_FINAL) > t0)
        if fresh:
            break
        if time.time() - t0 > MAX_WAIT_SEC:
            log(f"Timeout of {MAX_WAIT_SEC}s waiting for Rainbow; proceeding regardless")
            return
        time.sleep(60)
    log("Rainbow final detected (grid_rainbow_nature_s0_500k_final.pth); "
        "waiting 120s for process termination and emulator release")
    time.sleep(120)


def run_ppo_2m():
    model = os.path.join(BASE, "models", "ppo_ds123_2m_best.zip")
    final = os.path.join(BASE, "models", "ppo_ds123_2m_final.zip")
    if os.path.exists(final):
        log("ppo_ds123_2m_final already exists; skipping training")
        return
    cmd = [PY, "-m", "src.train_ppo",
           "--resume", "models/ppo_ds123_1m_final.zip",
           "--run-id", "ppo_ds123_2m", "--features", "nature",
           "--states", "ds1,ds2,ds3", "--n-envs", "3",
           "--timesteps", "1000000", "--seed", "0"]
    log(f"RUN PPO 2M: {' '.join(cmd)}")
    with open(PPO2M_LOG, "w") as f:
        subprocess.run(cmd, cwd=BASE, stdout=f, stderr=subprocess.STDOUT)
    log(f"PPO 2M finished (best={os.path.exists(model)}, final={os.path.exists(final)})")


def run_evals():
    rows = []
    for model_key, model_path in [("best", "models/ppo_ds123_2m_best.zip"),
                                  ("final", "models/ppo_ds123_2m_final.zip")]:
        full = os.path.join(BASE, model_path)
        if not os.path.exists(full):
            log(f"EVAL {model_key}: model not found, skipping")
            continue
        for sidx, ds in [(0, "ds1"), (1, "ds2"), (2, "ds3")]:
            out_csv = os.path.join(BASE, f"eval_ds123_2m_{model_key}_{ds}.csv")
            cmd = [PY, "-m", "src.eval", "--algo", "ppo", "--model", model_path,
                   "--n-episodes", "3", "--deterministic", "--seed", "0",
                   "--savestate-idx", str(sidx), "--out", out_csv]
            log(f"EVAL {model_key} {ds}")
            with open(PPO2M_LOG, "a") as f:
                f.write(f"\n=== EVAL {model_key} {ds} ===\n")
                f.flush()
                r = subprocess.run(cmd, cwd=BASE, stdout=f, stderr=subprocess.STDOUT)
            if os.path.exists(out_csv):
                with open(out_csv) as fh:
                    for row in csv.DictReader(fh):
                        rows.append({"model": model_key, "savestate": ds,
                                     "episode": row["episode"], "reward": row["reward"],
                                     "steps": row["steps"], "survived": row["survived"]})
    if rows:
        new_file = not os.path.exists(RESULTS)
        with open(RESULTS, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "savestate", "episode",
                                              "reward", "steps", "survived"])
            if new_file:
                w.writeheader()
            w.writerows(rows)
        log(f"Results consolidated in {RESULTS} ({len(rows)} rows)")


if __name__ == "__main__":
    log("Queue started: waiting for Rainbow 500k to finish...")
    wait_for_rainbow()
    run_ppo_2m()
    run_evals()
    log("Queue finished.")
