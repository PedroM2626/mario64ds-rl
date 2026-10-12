"""Where the time in a Mario 64 DS RL step actually goes.

This is the measurement that decides what is worth porting. Run it before
arguing about performance: it prices each Python-visible stage of env.step()
against the DeSmuME cycles the same step performs, on this machine, in one
process, with the emulator warm.

    python tools/profile_step.py --steps 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import cv2  # noqa: E402


def bench(fn, n, warmup=20):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--track", default="ds2", help="any savestate; the step shape is the same")
    ap.add_argument("--model", default="models/curriculum_flow25_r3_best.zip")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "step_profile.json"))
    args = ap.parse_args()

    from src.env import Mario64DSEnv

    env = Mario64DSEnv(
        rom_path=os.path.join(ROOT, "data", "Super Mario 64 DS (USA) (Rev 1).nds"),
        state_path=os.path.join(ROOT, "data", f"Super Mario 64 DS (USA) (Rev 1)."
                                              f"{args.track}"))
    env.reset()
    fs = env.frameskip

    def cycles_only():
        for _ in range(fs):
            env.emu.cycle()

    obs = env._get_obs()
    gray32 = cv2.resize(obs[:, :, 0], (32, 32), interpolation=cv2.INTER_AREA)
    prev32 = gray32.copy()

    def flow_only():
        cv2.calcOpticalFlowFarneback(prev32, gray32, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    rows = {
        "emulator_cycles": bench(cycles_only, args.steps),
        "get_obs": bench(env._get_obs, args.steps),
        "buffer_view": bench(lambda: np.array(env.emu.display_buffer_as_rgbx(),
                                              dtype=np.uint8).reshape((384, 256, 4)),
                             args.steps),
        "farneback_flow": bench(flow_only, args.steps),
        "ram_info": bench(env._get_info, args.steps),
        "full_step": bench(lambda: env.step(0), args.steps),
    }

    try:
        from stable_baselines3 import PPO
        model = PPO.load(args.model)
        stack = np.zeros((1, 4, 84, 84), dtype=np.uint8)
        rows["ppo_predict"] = bench(lambda: model.predict(stack, deterministic=True),
                                    args.steps)
        import torch
        rows["torch_device"] = str(next(model.policy.parameters()).device)
        rows["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as e:
        rows["ppo_predict_error"] = f"{type(e).__name__}: {e}"

    step = rows["full_step"]
    rows["cycles_share_of_step"] = round(100.0 * rows["emulator_cycles"] / step, 2)
    rows["python_share_of_step"] = round(100.0 * (
        rows["get_obs"] + rows["farneback_flow"] + rows["ram_info"]) / step, 2)
    rows["steps_per_second"] = round(1000.0 / step, 2)
    rows["frames_per_second"] = round(1000.0 * fs / step, 1)
    meta = {"n_steps": args.steps, "track": args.track, "frameskip": fs,
            "python": sys.version.split()[0], "cv2": cv2.__version__,
            "numpy": np.__version__}
    out = {"ms_per_call": {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in rows.items()}, "meta": meta}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    env.close()

    print(json.dumps(out, indent=2))
    print(f"\n[profile] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
