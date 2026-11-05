"""Turn one human recording into a route file and a start-state curriculum.

1. Verify the attempt's file and source-savestate hashes (src/demos.load_demo).
2. Replay its actions through a fresh Mario64DSEnv and require every 3D
   position to match the recording (and the star event, for a completed demo).
   A recording that does not reproduce is refused: its waypoints, checkpoints
   and replay-buffer transitions would describe a different game.
3. Resample the path every --spacing units of 3D arc length into ordered
   waypoints, erase loops (falls re-climbed, laps ridden on the clock hands;
   src/route.erase_loops), and copy the attempt's checkpoint savestates to
   --checkpoint-dir with their route position.

The route JSON holds coordinates only (no game data) and can be committed;
the checkpoint savestates are game RAM dumps and stay local like data/*.ds*.

Usage:
    python scripts/prepare_human_route.py runs/human-demonstrations/<run>/completed/attempt-000001
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np  # noqa: E402

from src.demos import decision_inputs, load_demo, sha256_file  # noqa: E402
from src.envs import rom_path  # noqa: E402
from src.route import ROUTE_SCHEMA, Route, erase_loops, resample_polyline  # noqa: E402


def repo_rel(path):
    """Repo-relative POSIX path when inside the repo, else absolute (other drive)."""
    try:
        rel = os.path.relpath(path, ROOT_DIR)
    except ValueError:
        return os.path.abspath(path)
    return os.path.abspath(path) if rel.startswith("..") else rel.replace(os.sep, "/")


def verify_replay(demo, tol=1e-3):
    """Max position error of an independent replay, and whether the star fired."""
    from src.env import Mario64DSEnv
    m = demo["manifest"]
    actions = demo["actions"].astype(int)
    env = Mario64DSEnv(rom_path=rom_path(), state_path=m["source_state"], max_steps=len(actions) + 1,
                       frameskip=m["frameskip"], flow_weight=0.0, action_set=m["action_set"],
                       track_id=m["track_id"], black_screen_death=False)
    try:
        env.reset()
        err = float(np.max(np.abs(env.ram.get_pos_3d() - demo["positions"][0])))
        info = {}
        for t, step_input in enumerate(decision_inputs(demo)):
            _, _, terminated, _, info = env.step(step_input)
            err = max(err, float(np.max(np.abs(env.ram.get_pos_3d() - demo["positions"][t + 1]))))
            if err > tol:
                return err, False, t
            if terminated and t != len(actions) - 1:
                return err, False, t
        return err, bool(info.get("star_collected") or info.get("reached_goal")), len(actions)
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("attempt", help="completed/ (or partial/ with --allow-partial) attempt folder")
    ap.add_argument("--out", default="config/routes/ttc_human_v1.json")
    ap.add_argument("--checkpoint-dir", default="data/curriculum/ttc_human_v1")
    ap.add_argument("--spacing", type=float, default=150.0, help="Waypoint spacing in world units")
    ap.add_argument("--corridor", type=float, default=250.0,
                    help="Max distance from the route for progress to count")
    ap.add_argument("--lookahead", type=float, default=600.0,
                    help="Max arc length ahead of the best progress searched per step")
    ap.add_argument("--lookbehind", type=float, default=300.0)
    ap.add_argument("--loop-radius", type=float, default=150.0,
                    help="Erase loops returning within this distance of earlier waypoints (0 = keep)")
    ap.add_argument("--allow-partial", action="store_true", help="Accept a demo that did not reach the star")
    ap.add_argument("--no-verify", action="store_true", help="Skip the replay check (not recommended)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    attempt = os.path.abspath(args.attempt)
    demo = load_demo(attempt)
    m = demo["manifest"]
    if m["state"] != "completed" and not args.allow_partial:
        raise SystemExit(f"{attempt} is a {m['state']} demo; pass --allow-partial to use it")
    out = os.path.join(ROOT_DIR, args.out)
    cp_dir = os.path.join(ROOT_DIR, args.checkpoint_dir)
    if not args.force and (os.path.exists(out) or os.path.exists(cp_dir)):
        raise SystemExit(f"{args.out} or {args.checkpoint_dir} already exists (use --force)")

    verified = None
    if not args.no_verify:
        err, star, upto = verify_replay(demo)
        print(f"replay: max position error {err:.6f} over {upto} decisions, goal reproduced={star}")
        if upto != len(demo["actions"]) or (m["state"] == "completed" and not star):
            raise SystemExit("the replay does not reproduce the recording; refusing to build a route")
        verified = {"max_position_error": err, "star_reproduced": star}

    pos = demo["positions"]
    raw = resample_polyline(pos, args.spacing)
    waypoints = erase_loops(raw, args.loop_radius) if args.loop_radius > 0 else raw
    raw_len = float(np.linalg.norm(np.diff(raw, axis=0), axis=1).sum())
    route = Route(waypoints, args.corridor, args.lookahead, args.lookbehind)

    if os.path.exists(cp_dir) and args.force:
        shutil.rmtree(cp_dir)
    os.makedirs(cp_dir)
    checkpoints = []
    for cp in m["checkpoints"]:
        src = os.path.join(attempt, cp["file"])
        dst = os.path.join(cp_dir, os.path.basename(cp["file"]))
        shutil.copy2(src, dst)
        s = route.locate(cp["pos"])
        checkpoints.append({"path": repo_rel(dst),
                            "step": cp["step"], "pos": cp["pos"],
                            "route_s": None if s is None else round(s, 2),
                            "sha256": sha256_file(dst)})

    spec = {
        "schema": ROUTE_SCHEMA, "track_id": m["track_id"],
        "source": {"attempt": repo_rel(attempt),
                   "state": m["state"], "source_state_sha256": m["source_state_sha256"],
                   "rom_sha256": m["rom_sha256"], "action_set": m["action_set"],
                   "frameskip": m["frameskip"], "decisions": m["decisions"],
                   "demo_sha256": m["files_sha256"]["demo.npz"], "replay_verified": verified},
        "spacing": args.spacing, "loop_radius": args.loop_radius,
        "length_before_loop_erasure": round(raw_len, 2), "corridor_radius": args.corridor,
        "lookahead": args.lookahead, "lookbehind": args.lookbehind,
        "length": round(route.length, 2),
        "height_gain": round(float(pos[-1, 1] - pos[0, 1]), 2),
        "waypoints": np.round(waypoints, 2).tolist(),
        "checkpoints": checkpoints,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=1)

    exact = m.get("exact_fraction")
    on_route = sum(c["route_s"] is not None for c in checkpoints)
    print(f"loop erasure: {len(raw)} -> {len(waypoints)} waypoints, length {raw_len:.0f} -> {route.length:.0f}; "
          f"{on_route}/{len(checkpoints)} checkpoints inside the corridor")
    print(f"route: {len(waypoints)} waypoints, length {route.length:.1f}, "
          f"height gain {spec['height_gain']:+.1f}, {len(checkpoints)} checkpoints -> {args.out}")
    print(f"{m.get('input_mode', 'agent')} input; decisions exactly reproducible by a {m['action_set']} "
          f"agent (usable as replay-buffer demos): {'n/a' if exact is None else f'{100 * exact:.1f}%'}")
    if "buttons_used" in m:
        print("frames with each button held:", json.dumps(m["buttons_used"]))
    print("action histogram:", json.dumps(m["action_histogram"]))


if __name__ == "__main__":
    main()
