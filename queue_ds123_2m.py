"""Fila sequencial: espera o Rainbow 500k (background) terminar, depois roda
a continuação do PPO 3-way até 2M total e avalia as 3 pistas.

Sinal de término do Rainbow: models/grid_rainbow_nature_s0_500k_final.pth
(só é salvo no fim do treino). Após detectar, espera 120s para o processo
encerrar e libera o emulador, então dispara o PPO 2M e os evals.

Uso:
    python queue_ds123_2m.py            (bloqueia até terminar a fila)
    Start-Process ... python queue_ds123_2m.py   (background, sobrevive à sessão)
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
    # Detecção por mtime: o final.pth só conta se for criado DEPOIS do início
    # da fila (existe um final antigo/stale de tentativa prévia que não conta).
    t0 = time.time()
    while True:
        fresh = (os.path.exists(RAINBOW_FINAL)
                 and os.path.getmtime(RAINBOW_FINAL) > t0)
        if fresh:
            break
        if time.time() - t0 > MAX_WAIT_SEC:
            log(f"Timeout de {MAX_WAIT_SEC}s esperando o Rainbow; seguindo mesmo assim")
            return
        time.sleep(60)
    log("Rainbow final detectado (grid_rainbow_nature_s0_500k_final.pth); "
        "aguardando 120s para o processo encerrar e liberar o emulador")
    time.sleep(120)


def run_ppo_2m():
    model = os.path.join(BASE, "models", "ppo_ds123_2m_best.zip")
    final = os.path.join(BASE, "models", "ppo_ds123_2m_final.zip")
    if os.path.exists(final):
        log("ppo_ds123_2m_final já existe; pulando treino")
        return
    cmd = [PY, "-m", "src.train_ppo",
           "--resume", "models/ppo_ds123_1m_final.zip",
           "--run-id", "ppo_ds123_2m", "--features", "nature",
           "--states", "ds1,ds2,ds3", "--n-envs", "3",
           "--timesteps", "1000000", "--seed", "0"]
    log(f"RUN PPO 2M: {' '.join(cmd)}")
    with open(PPO2M_LOG, "w") as f:
        subprocess.run(cmd, cwd=BASE, stdout=f, stderr=subprocess.STDOUT)
    log(f"PPO 2M terminou (best={os.path.exists(model)}, final={os.path.exists(final)})")


def run_evals():
    rows = []
    for model_key, model_path in [("best", "models/ppo_ds123_2m_best.zip"),
                                  ("final", "models/ppo_ds123_2m_final.zip")]:
        full = os.path.join(BASE, model_path)
        if not os.path.exists(full):
            log(f"EVAL {model_key}: modelo não encontrado, pulando")
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
        log(f"Resultados consolidados em {RESULTS} ({len(rows)} linhas)")


if __name__ == "__main__":
    log("Fila iniciada: esperando Rainbow 500k terminar...")
    wait_for_rainbow()
    run_ppo_2m()
    run_evals()
    log("Fila concluída.")
