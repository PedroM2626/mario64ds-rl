"""Record a greedy rollout of a BTR checkpoint (scripts/train_btr.py) on one slide as an mp4.

Same episode rules as training and evaluation (scripts/train_btr.ref_env_kwargs). Each frame has two
parts:
  * left: the game screen, with the decision, the coins gained, the distance to the finish and the position;
  * right: what the agent receives and what it does at that decision: the four 84x84 grayscale frames of
    its input stack (oldest first, the newest highlighted) and the ten actions, the chosen one lit.
The frame is the state the agent sees when it chooses; the last frame is the state after the final action
and is held for two seconds.

Usage:
    python scripts/record_btr.py --model models/btr_goal20/step_1000005.pt --track ds3 --attempts 10 \
        --out videos/btr/btr_ds3.mp4
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

PANEL_W = 264
BG = (24, 24, 24)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
GREY = (150, 150, 150)


def panel(stack, action, names, height):
    """Right-hand panel: the agent's 4-frame input and the action list (BGR image, PANEL_W x height)."""
    p = np.full((height, PANEL_W, 3), BG, np.uint8)
    cv2.putText(p, "What the agent sees (4 frames, 84x84 gray)", (6, 14), 0, 0.36, WHITE, 1, cv2.LINE_AA)
    side = 126
    for i, fr in enumerate(stack):
        g = cv2.resize(np.asarray(fr, np.uint8), (side, side), interpolation=cv2.INTER_NEAREST)
        g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        x, y = 6 + (i % 2) * (side + 4), 22 + (i // 2) * (side + 4)
        p[y:y + side, x:x + side] = g
        if i == len(stack) - 1:
            cv2.rectangle(p, (x, y), (x + side - 1, y + side - 1), YELLOW, 2)
    y0 = 22 + 2 * (side + 4) + 14
    cv2.putText(p, "Action chosen", (6, y0), 0, 0.36, WHITE, 1, cv2.LINE_AA)
    for i, n in enumerate(names):
        x, y = 6 + (i % 2) * (PANEL_W // 2), y0 + 6 + (i // 2) * 16
        on = i == action
        if on:
            cv2.rectangle(p, (x - 2, y), (x + PANEL_W // 2 - 8, y + 14), YELLOW, -1)
        cv2.putText(p, n, (x + 2, y + 11), 0, 0.4, (0, 0, 0) if on else GREY, 1, cv2.LINE_AA)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--track", default="ds3")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--scale", type=int, default=2, help="game-screen scale (the panel is sized to match at 2)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deterministic", action="store_true", help="fixed quantile levels: the policy is then a deterministic function of the observation")
    ap.add_argument("--attempts", type=int, default=1, help="retry until an episode finishes (only useful without --deterministic: the sampled quantile levels make episodes differ)")
    args = ap.parse_args()

    from scripts.train_btr import ref_env_kwargs
    from src.btr import BTRAgent
    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path

    agent = BTRAgent(10, device="cpu")
    agent.load(os.path.join(ROOT_DIR, args.model))
    env = Mario64DSEnv(rom_path=rom_path(), state_path=savestate_path(args.track), max_steps=args.max_steps,
                       track_id=args.track, **ref_env_kwargs(args.track))

    def render(t, a, stack, c0, dist, dmin):
        img = cv2.cvtColor(env.get_screen_rgb(), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)
        h = img.shape[0]
        pos = env.ram.get_pos_3d()
        cv2.putText(img, f"decision {t}   coins +{env.ram.get_coins() - c0}", (6, h - 40), 0, 0.5, YELLOW, 1, cv2.LINE_AA)
        cv2.putText(img, f"dist to finish {dist:.0f} (closest {dmin:.0f})", (6, h - 24), 0, 0.5, YELLOW, 1, cv2.LINE_AA)
        cv2.putText(img, f"pos {pos[0]:.0f} {pos[1]:.0f} {pos[2]:.0f}", (6, h - 8), 0, 0.5, WHITE, 1, cv2.LINE_AA)
        return np.hstack([img, panel(stack, a, env.action_names, h)])

    for attempt in range(args.attempts):
        obs, _ = env.reset()
        dq = deque([np.zeros((84, 84), np.uint8)] * 3 + [obs[:, :, 0]], maxlen=4)
        c0, frames, end, info = env.ram.get_coins(), [], "horizon", {}
        dmin, dist = 1e18, float("nan")
        a, t = -1, 0
        for t in range(args.max_steps):
            a = int(agent.act(np.stack(list(dq))[None], noisy=False, fixed_taus=args.deterministic)[0])
            frames.append(render(t + 1, a, list(dq), c0, dist, dmin if dmin < 1e17 else dist))
            obs, _, term, _, info = env.step(a)
            dq.append(obs[:, :, 0])
            dist = float(info.get("dist_to_goal", float("nan")))
            dmin = min(dmin, dist)
            if term:
                end = "star" if info.get("star_collected") else ("finish" if info.get("reached_goal") else "died")
                break
        frames.append(render(t + 1, a, list(dq), c0, dist, dmin))      # the state after the last action
        if end in ("finish", "star"):
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
    print(f"{end} after {n - 1} decisions, coins +{coins}, closest {dmin:.1f}, attempt {attempt + 1} of {args.attempts} -> {args.out}")


if __name__ == "__main__":
    main()
