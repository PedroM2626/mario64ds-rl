"""Record a greedy rollout of a BTR checkpoint (scripts/train_btr.py) on one slide as an mp4.

Same episode rules as training and evaluation (scripts/train_btr.ref_env_kwargs). Each frame shows the
decision, the action, the coins gained, the position and the distance to the finish; the last frame is
held for two seconds.

Usage:
    python scripts/record_btr.py --model models/btr_ref10/step_500010.pt --track ds3 --out videos/btr_ds3.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import deque

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--track", default="ds3")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", required=True)
    ap.add_argument("--attempts", type=int, default=1, help="retry (the game is not deterministic) until an episode finishes")
    args = ap.parse_args()

    from scripts.train_btr import ref_env_kwargs
    from src.btr import BTRAgent
    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path

    agent = BTRAgent(10, device="cpu")
    agent.load(os.path.join(ROOT_DIR, args.model))
    env = Mario64DSEnv(rom_path=rom_path(), state_path=savestate_path(args.track), max_steps=args.max_steps,
                       track_id=args.track, **ref_env_kwargs(args.track))
    for attempt in range(args.attempts):
        obs, _ = env.reset()
        dq = deque([np.zeros((84, 84), np.uint8)] * 3 + [obs[:, :, 0]], maxlen=4)
        c0, frames, end, info = env.ram.get_coins(), [], "horizon", {}
        dmin = 1e18
        for t in range(args.max_steps):
            a = int(agent.act(np.stack(list(dq))[None], noisy=False)[0])
            obs, _, term, _, info = env.step(a)
            dq.append(obs[:, :, 0])
            dist = float(info.get("dist_to_goal", float("nan")))
            dmin = min(dmin, dist)
            img = cv2.cvtColor(env.get_screen_rgb(), cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)
            pos = env.ram.get_pos_3d()
            h = img.shape[0]
            cv2.putText(img, f"decision {t + 1}  {env.action_names[a]}  coins +{env.ram.get_coins() - c0}", (6, h - 40),
                        0, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"dist to finish {dist:.0f} (closest {dmin:.0f})", (6, h - 24), 0, 0.5,
                        (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"pos {pos[0]:.0f} {pos[1]:.0f} {pos[2]:.0f}", (6, h - 8), 0, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            frames.append(img)
            if term:
                end = "finish" if info.get("reached_goal") else "died"
                break
        if end == "finish":
            break
    coins = env.ram.get_coins() - c0
    env.close()
    n = len(frames)
    frames += [frames[-1].copy() for _ in range(args.fps * 2)]
    out = os.path.join(ROOT_DIR, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    hh, ww = frames[0].shape[:2]
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (ww, hh))
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"{end} after {n} decisions, coins +{coins}, closest {dmin:.1f}, attempt {attempt + 1} of {args.attempts} -> {args.out}")


if __name__ == "__main__":
    main()
