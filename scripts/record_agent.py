"""Record a greedy rollout of a PPO checkpoint on a savestate as an mp4.

The recording env uses the same rules as the ds3 trainings (no black-screen death, a displacement above
--respawn-jump in one decision is the game's respawn and ends the episode). Each frame is annotated with
the decision, the position and the distance to the finish.

Usage:
    python scripts/record_agent.py --model models/ds3_bcppo_p1/checkpoints/step_1770000_steps.zip \
        --out videos/ds3_bcppo_200k.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--state", default=None, help="savestate path (default: the ds3 start)")
    ap.add_argument("--track", default="ds3")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--respawn-jump", type=float, default=2000.0)
    ap.add_argument("--fall-speed", type=float, default=140.0,
                    help="descending faster than this for --fall-steps decisions in a row ends the episode "
                         "(0 = wait for the respawn)")
    ap.add_argument("--fall-steps", type=int, default=15)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from collections import deque

    from stable_baselines3 import PPO
    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path

    model = PPO.load(os.path.join(ROOT_DIR, args.model), device="cpu")
    n = int(model.action_space.n)
    env = Mario64DSEnv(rom_path=rom_path(), state_path=args.state or savestate_path(args.track),
                       max_steps=args.max_steps, flow_weight=0.0, dist_reward_enabled=False, track_id=args.track,
                       action_set={6: "legacy6", 12: "compound12"}[n], black_screen_death=False,
                       respawn_jump=args.respawn_jump, fall_speed=args.fall_speed, fall_steps=args.fall_steps)
    frames = []
    obs, _ = env.reset()
    dq = deque([np.zeros((84, 84), np.uint8)] * 3 + [obs[:, :, 0]], maxlen=4)
    info, end = {}, "horizon"
    for t in range(args.max_steps):
        a = int(model.predict(np.stack(list(dq))[None], deterministic=True)[0][0])
        obs, _, term, _, info = env.step(a)
        dq.append(obs[:, :, 0])
        img = cv2.cvtColor(env.get_screen_rgb(), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)
        pos = env.ram.get_pos_3d()
        label = f"decision {t + 1}  {env.action_names[a]}  dist {info.get('dist_to_goal', float('nan')):.0f}"
        cv2.putText(img, label, (6, img.shape[0] - 24), 0, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"pos {pos[0]:.0f} {pos[1]:.0f} {pos[2]:.0f}", (6, img.shape[0] - 8), 0, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(img)
        if term:
            end = "finish" if info.get("reached_goal") else "died"
            break
    env.close()
    for _ in range(args.fps * 2):                       # hold the last frame
        frames.append(frames[-1].copy())
    out = os.path.join(ROOT_DIR, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"{end} after {len(frames) - args.fps * 2} decisions, closest {info.get('dist_to_goal')} -> {args.out}")


if __name__ == "__main__":
    main()
