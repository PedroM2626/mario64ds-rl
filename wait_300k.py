"""Aguarda um checkpoint atingir N timesteps (polling de best_model/final).

Substitui a versão antiga que só olhava ``models/best_model.zip`` (que agora
é por run-id: ``models/<run-id>/best_model.zip``) e não tinha timeout.

Exemplos:
    python wait_300k.py --run-id ppo_mario64ds_baseline --target-steps 300000
    python wait_300k.py --run-id ppo_mario64ds_8_envs --target-steps 500000 --timeout-sec 7200
"""

import argparse
import os
import time

from stable_baselines3 import PPO


def candidate_paths(run_id: str):
    base = os.path.dirname(os.path.abspath(__file__))
    models = os.path.join(base, "models")
    return [
        os.path.join(models, run_id, "best_model.zip"),
        os.path.join(models, f"{run_id}_best.zip"),
        os.path.join(models, f"{run_id}_final.zip"),
        os.path.join(models, "best_model.zip"),  # legado
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default="ppo_mario64ds_baseline")
    parser.add_argument("--target-steps", type=int, default=300000)
    parser.add_argument("--poll-sec", type=int, default=30)
    parser.add_argument("--timeout-sec", type=int, default=6 * 3600)
    args = parser.parse_args()

    start = time.time()
    steps = 0
    while steps < args.target_steps:
        if time.time() - start > args.timeout_sec:
            print(f"Timeout após {args.timeout_sec}s (atual: {steps} passos).")
            raise SystemExit(1)
        time.sleep(args.poll_sec)
        for path in candidate_paths(args.run_id):
            if not os.path.exists(path):
                continue
            try:
                model = PPO.load(path, device="cpu")
                steps = model.num_timesteps
                print(f"[{path}] Current steps: {steps}")
                break
            except Exception as e:
                print(f"[{path}] ainda não legível: {e}")

    print(f"Reached {args.target_steps} steps!")


if __name__ == "__main__":
    main()
