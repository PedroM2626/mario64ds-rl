"""Record / evaluate the world-model agent playing Mario 64 DS on the REAL game.

Real-time closed-loop test of the learned latent policy (belief-state
controller). One subprocess per track (DeSmuME crashes if two emulators share a
process). Saves an RGB MP4 per track and prints reward / steps / completion.

    python scripts/record_world_model.py --model models/wm_mario64ds_wm.pt
    python scripts/record_world_model.py --model models/wm_mario64ds_wm.pt --live
"""

import argparse
import os
import subprocess
import sys
import time
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TRACKS = ["ds1", "ds2", "ds3"]


def _record_single(args, out_dir):
    import numpy as np
    import imageio
    from src.env import Mario64DSEnv
    from src.world_model_controller import WorldModelController, load_world_model_actor

    rom = os.path.join(ROOT, args.rom)
    ss = os.path.join(ROOT, "data", f"Super Mario 64 DS (USA) (Rev 1).{args.single}")
    bundle = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)
    model, actor, meta = load_world_model_actor(bundle, device=args.device)
    ctrl = WorldModelController(model, actor, device=args.device,
                                deterministic=not args.stochastic)

    env = Mario64DSEnv(rom, ss, max_steps=args.max_steps, flow_weight=args.flow_weight)
    obs, _ = env.reset()
    action = ctrl.reset(obs[:, :, 0])
    frames = [env.get_screen_rgb()]
    steps, R, done = 0, 0.0, False
    while not done and steps < args.max_steps:
        obs, reward, terminated, truncated, _ = env.step(action)
        action = ctrl.act(obs[:, :, 0])
        R += float(reward)
        steps += 1
        done = bool(terminated or truncated)
        if args.live:
            cv = obs
            import cv2
            cv2.imshow(f"WM {args.single}", cv2.resize(cv[:, :, 0], (336, 336)))
            cv2.waitKey(1)
        frames.append(env.get_screen_rgb())
    env.close()

    out = os.path.join(out_dir, f"wm_{args.single}.mp4")
    with imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8,
                            macro_block_size=1) as w:
        for fr in frames:
            w.append_data(fr)
    survived = steps >= args.max_steps
    print(f"{args.single}: reward={R:.2f} steps={steps}/{args.max_steps} "
          f"{'COMPLETED (survived)' if survived else 'FAILED'} | {out}", flush=True)
    return survived, steps, R


def main():
    p = argparse.ArgumentParser(description="Record world-model agent on real game")
    p.add_argument("--model", required=True)
    p.add_argument("--rom", default="data/Super Mario 64 DS (USA) (Rev 1).nds")
    p.add_argument("--out-dir", default="videos")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=1350)
    p.add_argument("--flow-weight", type=float, default=2.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--live", action="store_true", help="show real-time window")
    p.add_argument("--tracks", default=",".join(TRACKS))
    p.add_argument("--single", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.single is None:
        summary = {}
        for tr in [t.strip() for t in args.tracks.split(",") if t.strip()]:
            r = subprocess.run(
                [sys.executable, __file__, "--model", args.model, "--rom", args.rom,
                 "--out-dir", args.out_dir, "--fps", str(args.fps), "--max-steps",
                 str(args.max_steps), "--flow-weight", str(args.flow_weight),
                 "--device", args.device, "--single", tr]
                + (["--live"] if args.live else [])
                + (["--stochastic"] if args.stochastic else []),
                cwd=ROOT)
            summary[tr] = r.returncode
        print("\n=== WM record summary ===")
        for k, v in summary.items():
            print(f"  {k}: exit={v}")
        return

    _record_single(args, out_dir)


if __name__ == "__main__":
    main()
