"""Tick Tock Clock with two policies: the main one plays until the hand-over point, the transfer specialist from there.

Mode 1 (default): episodes from the normal ds8 start. The main policy acts until its route progress reaches
--handoff-wp waypoints; from then on the specialist acts (its frame stack restarts from zeros plus the current
frame, as in its training episodes that start from a savestate). Greedy, random quantile levels, like probe_ttc.py.

Mode 2 (--states DIR): the specialist alone, one episode from each savestate of DIR (held-out hand-over states).

Usage:
    python scripts/eval_ttc_composite.py --main models/btr_ttc_s9/step_750000.pt --spec models/btr_ttc_spec1/final.pt \
        --handoff-wp 340 --episodes 10 --max-steps 5000 --out results/ttc_composite_spec1.json
    python scripts/eval_ttc_composite.py --spec models/btr_ttc_spec1/final.pt --states data/curriculum/ttc_handoff_val \
        --max-steps 700 --out results/ttc_spec_states_spec1.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import deque

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np  # noqa: E402


def load_agent(path):
    from src.btr import BTRAgent, checkpoint_actions
    n = checkpoint_actions(os.path.join(ROOT_DIR, path))
    agent = BTRAgent(n, device="cpu")
    agent.load(os.path.join(ROOT_DIR, path))
    return agent, n


def fresh_stack(obs):
    return deque([np.zeros((84, 84), np.uint8)] * 3 + [obs[:, :, 0]], maxlen=4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", default=None, help="main policy (not needed with --states)")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--handoff-wp", type=int, default=340)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--states", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stall-handoff", type=int, default=0,
                    help="also hand over when the main policy has reached --stall-min-wp waypoints but gained none for this many decisions (0 = off)")
    ap.add_argument("--stall-min-wp", type=int, default=330)
    ap.add_argument("--idle", type=int, default=0, help="stop an episode after this many decisions without a new waypoint (0 = never)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--deterministic", action="store_true", help="fixed quantile levels: both policies are then deterministic functions of the observation")
    ap.add_argument("--shot-dir", default=None, help="save screenshots of the last ~70 decisions of every episode (every 10th) to this directory")
    ap.add_argument("--trace", default=None, help="write the per-decision action, position and route arc length of the specialist's part to this json")
    args = ap.parse_args()

    import torch
    from scripts.train_btr import ref_env_kwargs
    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path
    from src.route import Route

    torch.manual_seed(args.seed)
    route = Route.load(os.path.join(ROOT_DIR, "config/routes/ttc_human_v1.json"))
    spec, n_spec = load_agent(args.spec)
    main_agent, n_main = (load_agent(args.main) if args.main else (None, 0))
    n_act = max(n_spec, n_main)
    action_set = {10: "ref10", 13: "ref13"}[n_act]
    if n_main and n_main != n_act and n_main != 10:
        raise SystemExit("the main policy must have the first actions of the specialist's action set")
    env = Mario64DSEnv(rom_path=rom_path(), state_path=savestate_path("ds8"), max_steps=args.max_steps, track_id="ds8",
                       **ref_env_kwargs("ds8", action_set=action_set))
    files = sorted(glob.glob(os.path.join(ROOT_DIR, args.states, "*.dst"))) if args.states else []
    n_eps = len(files) if args.states else args.episodes
    rows, traces = [], []
    for ep in range(n_eps):
        trace = []
        shots = deque(maxlen=8)
        if args.states:
            env.start_states, env.full_start_prob = [files[ep]], 0.0
        obs, _ = env.reset()
        dq = fresh_stack(obs)
        acting = spec if args.states else main_agent
        handoff, best_wp, last_gain, end, info, star = (0 if args.states else None), 0, 0, "horizon", {}, False
        if args.states:
            best_wp = route.passed(0, float(env.route_best_s or 0.0))
        wp_at_handoff = best_wp if args.states else None
        star_log, last_sc = [], None       # (decision, value) every time the RAM star counter changes
        handoff_reason = None
        landed = False          # route arc length >= 51,000 (waypoint ~354) with Mario above height 5,800: on the platform after the hand
        for t in range(args.max_steps):
            a = int(acting.act(np.stack(list(dq))[None], noisy=False, fixed_taus=args.deterministic)[0])
            obs, _, term, _, info = env.step(a)
            dq.append(obs[:, :, 0])
            wp = route.passed(0, float(info.get("route_progress") or 0.0))
            sc = env.ram.get_star_count()
            if sc != last_sc:
                star_log.append((t + 1, sc))
                last_sc = sc
            if float(info.get("route_progress") or 0.0) >= 51000 and float(env.ram.get_pos_3d()[1]) > 5800:
                landed = True
            if args.shot_dir and (t + 1) % 10 == 0:
                shots.append((t + 1, np.array(env.emu.display_buffer_as_rgbx(), dtype=np.uint8).reshape(384, 256, 4)[:, :, :3].copy()))
            if args.trace and acting is spec:
                trace.append([t + 1, a] + [round(float(v), 1) for v in env.ram.get_pos_3d()] + [round(float(info.get("route_progress") or 0.0), 1)])
            if wp > best_wp:
                best_wp, last_gain = wp, t
            stalled = (args.stall_handoff and best_wp >= args.stall_min_wp and t - last_gain >= args.stall_handoff)
            if handoff is None and (best_wp >= args.handoff_wp or stalled):
                handoff_reason = "wp" if best_wp >= args.handoff_wp else "stall"
                handoff, wp_at_handoff, acting, dq, last_gain = t + 1, best_wp, spec, fresh_stack(obs), t
            if term:
                end = "star" if info.get("success") else "died"
                star = end == "star"
                break
            if args.idle and t - last_gain >= args.idle:
                end = "idle"
                break
        rows.append({"episode": ep, "start": os.path.basename(files[ep]) if args.states else "full", "handoff_decision": handoff, "handoff_reason": handoff_reason,
                     "wp_at_handoff": wp_at_handoff, "best_wp": best_wp, "landed": landed, "death_source": getattr(env, "death_source", None), "course_id_end": int(env.ram.get_course_id()),
                     "end_pos": [round(float(v), 1) for v in env.ram.get_pos_3d()], "star_counter_changes": star_log[:40], "decisions": t + 1, "end": end, "star": star})
        print(rows[-1], flush=True)
        traces.append(trace)
        if args.shot_dir:
            from PIL import Image
            os.makedirs(os.path.join(ROOT_DIR, args.shot_dir), exist_ok=True)
            tag = os.path.splitext(rows[-1]["start"])[0] + f"_{rows[-1]['end']}"
            for k, (d, img) in enumerate(shots):
                Image.fromarray(img).save(os.path.join(ROOT_DIR, args.shot_dir, f"{tag}_{k}_d{d:04d}.png"))
    env.close()
    if args.trace:
        with open(os.path.join(ROOT_DIR, args.trace), "w", encoding="utf-8") as fh:
            json.dump(traces, fh)
    wps = sorted(r["best_wp"] for r in rows)
    summary = {"episodes": len(rows), "stars": sum(r["star"] for r in rows), "median_wp": wps[len(wps) // 2],
               "min_wp": wps[0], "max_wp": wps[-1], "landed": sum(r["landed"] for r in rows)}
    print("SUMMARY", summary, flush=True)
    if args.out:
        with open(os.path.join(ROOT_DIR, args.out), "w", encoding="utf-8") as fh:
            json.dump({"main": args.main, "spec": args.spec, "handoff_wp": args.handoff_wp, "summary": summary, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
