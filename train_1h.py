"""Treina por um tempo limitado e depois visualiza.

Substitui a versão antiga que usava ``wmic`` (Windows-only) + ``sleep(3600)``
hardcoded. Agora é cross-platform, com timeout configurável e encerramento
limpo via ``Popen.terminate()``.

Exemplos:
    python train_1h.py --timesteps 500000 --algo ppo
    python train_1h.py --timeout-sec 3600 --run-id rainbow_mario64ds_1h --algo rainbow
"""

import argparse
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Train with a time budget, then visualize")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "rainbow", "qrdqn"])
    parser.add_argument("--run-id", type=str, default="ppo_mario64ds_1h")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--timeout-sec", type=int, default=3600, help="Tempo máximo de treino (default: 1h)")
    parser.add_argument("--n-envs", type=int, default=4)
    args = parser.parse_args()

    if args.algo == "ppo":
        cmd = [sys.executable, "-m", "src.train_ppo", "--run-id", args.run_id,
               "--timesteps", str(args.timesteps), "--n-envs", str(args.n_envs)]
    elif args.algo == "rainbow":
        cmd = [sys.executable, "-m", "src.train", "--run-id", args.run_id,
               "--timesteps", str(args.timesteps)]
    else:
        cmd = [sys.executable, "-m", "src.train_qrdqn", "--run-id", args.run_id,
               "--timesteps", str(args.timesteps), "--n-envs", str(args.n_envs)]

    print(f"Iniciando treinamento ({args.algo}) com timeout de {args.timeout_sec}s...")
    print("CMD:", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    try:
        proc.wait(timeout=args.timeout_sec)
        print("Treinamento terminou antes do timeout.")
    except subprocess.TimeoutExpired:
        print(f"Timeout de {args.timeout_sec}s atingido! Encerrando treinamento...")
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        # Dá tempo para os pesos serem salvos pelo callback
        time.sleep(5)

    print("Iniciando a visualização...")
    result = subprocess.run(
        [sys.executable, "-m", "src.play", "--algo", args.algo],
        capture_output=True, text=True
    )
    print("\n--- RESULTADOS DA VISUALIZAÇÃO ---")
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)


if __name__ == "__main__":
    main()
