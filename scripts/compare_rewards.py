#!/usr/bin/env python3
"""
scripts/compare_rewards.py: Empirical Comparison of Reward Formulations on Slide Tracks

Compares the baseline (Old) vs proposed (New) reward systems across slide courses
(ds1: Peach Slide, ds2: TTM Slide, ds3: CCM Slide) using the flagship trained policy
models/curriculum_flow25_r3_best.zip.

Formulations:
- Old Reward:
    R = w_flow * max(0, v_y) - c_step + R_death * I(visual_death)
- New Reward:
    R = w_flow * max(0, v_y) - c_step + w_dist * (D_{t-1} - D_t)
        + R_goal * I(d <= 250) + R_death * I(ram_death)

Each track evaluation runs in a dedicated subprocess to preserve emulator stability
(DeSmuME C++ static singleton constraint).
"""

import os
import sys
import argparse
import json
import subprocess
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ROM_PATH = str(REPO_ROOT / "data" / "Super Mario 64 DS (USA) (Rev 1).nds")
DEFAULT_MODEL = str(REPO_ROOT / "models" / "curriculum_flow25_r3_best.zip")
TRACKS = ["ds1", "ds2", "ds3"]

def run_worker(track: str, reward_mode: str, model_path: str, max_steps: int = 1350):
    """Executes a single evaluation episode inside the worker process."""
    from stable_baselines3 import PPO
    from src.env import Mario64DSEnv
    from src.ram_state import TRACK_GOALS

    state_path = str(REPO_ROOT / "data" / f"Super Mario 64 DS (USA) (Rev 1).{track}")
    is_new = (reward_mode == "new")

    env = Mario64DSEnv(
        rom_path=ROM_PATH,
        state_path=state_path,
        max_steps=max_steps,
        dist_reward_enabled=is_new,
        use_ram_death=is_new,
        dist_weight=0.05,
        goal_reward=200.0,
        flow_weight=1.0,
        step_penalty=0.02,
        death_penalty=100.0,
        track_id=track
    )

    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
    vec = DummyVecEnv([lambda: env])
    vec = VecFrameStack(vec, n_stack=4)
    vec = VecTransposeImage(vec)

    model = PPO.load(model_path, device="cpu")

    vec_obs = vec.reset()
    start_pos = env.ram.get_pos_3d().copy()
    spec = env.goal_spec
    if spec is not None and "target_pos" in spec:
        initial_dist = float(np.linalg.norm(start_pos - spec["target_pos"]))
    else:
        initial_dist = 0.0

    total_reward = 0.0
    flow_sum = 0.0
    dist_reward_sum = 0.0
    step_penalty_sum = 0.0
    goal_reached = False
    death = False
    final_pos = start_pos.copy()

    for step in range(max_steps):
        action, _ = model.predict(vec_obs, deterministic=True)
        vec_obs, r, dones, infos = vec.step(action)
        info = infos[0]

        total_reward += float(r[0])
        flow_sum += info.get("flow_reward", 0.0)
        dist_reward_sum += info.get("dist_reward", 0.0)
        step_penalty_sum += env.step_penalty
        final_pos = env.ram.get_pos_3d().copy()

        if info.get("goal_reached", False):
            goal_reached = True
        if info.get("died", False) or info.get("dead", False) or dones[0]:
            if not goal_reached:
                death = True

        if dones[0]:
            break

    if spec is not None and "target_pos" in spec:
        final_dist = float(np.linalg.norm(final_pos - spec["target_pos"]))
    else:
        final_dist = 0.0
    steps_taken = step + 1
    vec.close()

    result = {
        "track": track,
        "mode": reward_mode,
        "steps": steps_taken,
        "total_reward": float(total_reward),
        "flow_sum": float(flow_sum),
        "dist_reward_sum": float(dist_reward_sum),
        "step_penalty_sum": float(step_penalty_sum),
        "goal_reached": bool(goal_reached),
        "death": bool(death),
        "initial_dist": float(initial_dist),
        "final_dist": float(final_dist),
        "progress_delta": float(initial_dist - final_dist),
        "start_pos": start_pos.tolist(),
        "final_pos": final_pos.tolist()
    }
    print(f"__RESULT__={json.dumps(result)}")

def main():
    parser = argparse.ArgumentParser(description="Compare Old vs New Reward System")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--track", choices=TRACKS, default=None, help="Specific track to evaluate")
    parser.add_argument("--mode", choices=["old", "new"], default="new")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model checkpoint path")
    parser.add_argument("--max-steps", type=int, default=1350)
    parser.add_argument("--out", default="docs/REWARD_COMPARISON_RESULTS.json")

    args = parser.parse_args()

    if args.worker:
        run_worker(args.track or "ds1", args.mode, args.model, args.max_steps)
        return

    print("=" * 80)
    print("EMPIRICAL COMPARISON: Baseline (Old) vs Proposed (New) Reward System")
    print(f"Policy: {args.model}")
    print("=" * 80)

    results = []
    tracks_to_run = [args.track] if args.track else TRACKS

    for track in tracks_to_run:
        for mode in ["old", "new"]:
            print(f"Evaluating {track.upper()} under [{mode.upper()}] reward formulation...")
            cmd = [
                sys.executable,
                __file__,
                "--worker",
                "--track", track,
                "--mode", mode,
                "--model", args.model,
                "--max-steps", str(args.max_steps)
            ]
            env_vars = os.environ.copy()
            env_vars["PYTHONPATH"] = str(REPO_ROOT)
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env_vars)

            found = False
            for line in proc.stdout.splitlines():
                if line.startswith("__RESULT__="):
                    data = json.loads(line[len("__RESULT__="):])
                    results.append(data)
                    found = True
                    break

            if not found:
                print(f"[ERROR] Failed to obtain result for {track} ({mode}):\n{proc.stderr}")

    # Output Comparative Summary Table
    print("\n" + "=" * 80)
    print("SUMMARY COMPARATIVE TABLE (Flagship Model Evaluation)")
    print("=" * 80)
    print(f"{'Track':<6} {'Mode':<6} {'Steps':<7} {'Outcome':<14} {'Total R':<10} {'Dist 0':<9} {'Dist T':<9} {'Delta D':<9}")
    print("-" * 80)
    for r in results:
        outcome = "GOAL REACHED" if r["goal_reached"] else ("DEATH" if r["death"] else "TIMEOUT")
        print(f"{r['track']:<6} {r['mode']:<6} {r['steps']:<7} {outcome:<14} {r['total_reward']:<10.2f} "
              f"{r['initial_dist']:<9.1f} {r['final_dist']:<9.1f} {r['progress_delta']:<9.1f}")
    print("=" * 80)

    # Save to JSON
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to: {out_path}")

if __name__ == "__main__":
    main()
