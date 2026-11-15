"""Record one episode of the two-policy Tick Tock Clock agent (main policy, then the transfer specialist) as an mp4.

Replays exactly the random sequence of scripts/eval_ttc_composite.py for the same --seed: episodes 0..--episode-1 are
played unrecorded, so that --episode k is the k-th episode of that evaluation run. One video frame per decision (both DS
screens); a banner shows which policy acts, the decision number and the waypoints reached.

Usage:
    python scripts/record_ttc_composite.py --main models/btr_ttc_s9/step_750000.pt --spec models/btr_ttc_spec3/final.pt \
        --seed 44 --episode 0 --max-steps 3000 --out videos/ttc_composite_star.mp4
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
    ap.add_argument("--main", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--handoff-wp", type=int, default=340)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from scripts.eval_ttc_composite import fresh_stack, load_agent
    from scripts.train_btr import ref_env_kwargs
    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path
    from src.route import Route

    torch.manual_seed(args.seed)
    route = Route.load(os.path.join(ROOT_DIR, "config/routes/ttc_human_v1.json"))
    spec, n_spec = load_agent(args.spec)
    main_agent, n_main = load_agent(args.main)
    action_set = {10: "ref10", 13: "ref13"}[max(n_spec, n_main)]
    env = Mario64DSEnv(rom_path=rom_path(), state_path=savestate_path("ds8"), max_steps=args.max_steps, track_id="ds8",
                       **ref_env_kwargs("ds8", action_set=action_set))
    writer, outcome = None, None
    for ep in range(args.episode + 1):
        rec = ep == args.episode
        obs, _ = env.reset()
        dq = fresh_stack(obs)
        acting, handoff, best_wp, name = main_agent, None, 0, "main policy (s9)"
        for t in range(args.max_steps):
            if rec:
                img = cv2.cvtColor(env.get_screen_rgb(), cv2.COLOR_RGB2BGR)
                img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)
                if writer is None:
                    os.makedirs(os.path.dirname(os.path.join(ROOT_DIR, args.out)), exist_ok=True)
                    writer = cv2.VideoWriter(os.path.join(ROOT_DIR, args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                             (img.shape[1], img.shape[0]))
                cv2.rectangle(img, (0, 0), (img.shape[1], 22), (24, 24, 24), -1)
                cv2.putText(img, f"{name} | decision {t} | waypoint {best_wp}/370", (6, 16), 0, 0.5,
                            (0, 255, 255) if acting is spec else (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(img)
            a = int(acting.act(np.stack(list(dq))[None], noisy=False, fixed_taus=False)[0])
            obs, _, term, _, info = env.step(a)
            dq.append(obs[:, :, 0])
            best_wp = max(best_wp, route.passed(0, float(info.get("route_progress") or 0.0)))
            if handoff is None and best_wp >= args.handoff_wp:
                handoff, acting, dq, name = t + 1, spec, fresh_stack(obs), "transfer specialist"
            if term:
                outcome = "star" if info.get("success") else "died"
                break
        else:
            outcome = "horizon"
        if rec:
            img = cv2.cvtColor(env.get_screen_rgb(), cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(img, (0, 0), (img.shape[1], 22), (24, 24, 24), -1)
            cv2.putText(img, f"{outcome.upper()} | decision {t + 1} | waypoint {best_wp}/370", (6, 16), 0, 0.5,
                        (0, 255, 0) if outcome == "star" else (0, 0, 255), 1, cv2.LINE_AA)
            for _ in range(2 * args.fps):
                writer.write(img)
            writer.release()
        print(f"episode {ep}: {outcome}, {t + 1} decisions, best waypoint {best_wp}, handoff {handoff}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
