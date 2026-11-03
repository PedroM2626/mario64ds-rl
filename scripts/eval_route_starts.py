"""Evaluate a TTC QR-DQN checkpoint from the full start and from route checkpoints.

A full-start evaluation (EvalCallback, scripts/eval_qrdqn_ttc.py) says nothing
about the parts of the tower the curriculum trained on. This runs the greedy
policy from `ds8` and from every --every-th checkpoint of a route file, each for
at most --max-steps decisions, and reports the route progress gained beyond the
start's own route position, how the episode ended and the height gained.

Usage:
    python scripts/eval_route_starts.py --model models/qrdqn_impala_ds8_v4_best.zip
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="QR-DQN checkpoint, or 'noop' for a do-nothing baseline (passive platform motion)")
    ap.add_argument("--route", default="config/routes/ttc_human_v1.json")
    ap.add_argument("--every", type=int, default=8, help="Use every N-th route checkpoint")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--out", default=None, help="JSON output (default results/route_starts_<model>.json)")
    args = ap.parse_args()

    from sb3_contrib import QRDQN
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path
    from src.route import Route, route_checkpoints

    route_path = os.path.join(ROOT_DIR, args.route)
    route = Route.load(route_path)
    starts = [("ds8", savestate_path("ds8"))]
    starts += [(os.path.basename(p), p) for p in route_checkpoints(route_path)[::args.every]]
    model = None if args.model == "noop" else QRDQN.load(os.path.join(ROOT_DIR, args.model), device="cpu")
    action_set = "compound12" if model is None else {6: "legacy6", 12: "compound12"}[int(model.action_space.n)]
    env = Mario64DSEnv(rom_path=rom_path(), state_path=starts[0][1], max_steps=args.max_steps,
                       flow_weight=0.0, dist_reward_mode="route", route_path=route_path,
                       action_set=action_set, track_id="ds8")
    vec = VecTransposeImage(VecFrameStack(DummyVecEnv([lambda: env]), n_stack=4))
    rows = []
    try:
        for name, path in starts:
            env.state_path = path
            obs = vec.reset()
            s0 = env.route_best_s
            y0 = float(env.ram.get_pos_3d()[1])
            ret, ymax, info = 0.0, y0, {}
            for t in range(args.max_steps):
                action = np.zeros(1, int) if model is None else model.predict(obs, deterministic=True)[0]
                obs, r, done, infos = vec.step(action)
                info = infos[0]
                ret += float(r[0])
                if info.get("pos_3d") is not None:
                    ymax = max(ymax, float(info["pos_3d"][1]))
                if done[0]:
                    break
            s1 = info.get("route_progress")
            end = ("star" if info.get("star_collected") else "died" if info.get("died")
                   else "horizon")
            row = {"start": name, "route_s_start": s0, "route_s_end": s1,
                   "route_gain": None if s0 is None or s1 is None else round(s1 - s0, 1),
                   "decisions": t + 1, "end": end, "return": round(ret, 2),
                   "height_gain": round(ymax - y0, 1)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        vec.close()
    gains = [r["route_gain"] for r in rows if r["route_gain"] is not None]
    summary = {"model": args.model, "route": args.route, "route_length": route.length,
               "max_steps": args.max_steps, "starts": len(rows),
               "died": sum(r["end"] == "died" for r in rows),
               "star": sum(r["end"] == "star" for r in rows),
               "median_route_gain": float(np.median(gains)) if gains else None, "rows": rows}
    out = args.out or os.path.join(
        ROOT_DIR, "results", f"route_starts_{os.path.splitext(os.path.basename(args.model))[0]}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"SUMMARY starts={summary['starts']} died={summary['died']} star={summary['star']} "
          f"median_route_gain={summary['median_route_gain']} -> {out}")


if __name__ == "__main__":
    main()
